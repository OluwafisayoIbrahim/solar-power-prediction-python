"""Monitor tests: stdlib by default; real Chromium fixtures explicitly enabled in CI."""
import asyncio
import contextlib
import io
from html import escape
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_streamlit as monitor


READY = '''<div role="radiogroup" aria-label="Research dashboard"><label><input type="radio">Overview</label></div>
<h2>Key findings</h2><button>Read the Manuscript</button>'''
SLEEP = '<h1>This app has gone to sleep due to inactivity</h1>'


class ArgumentTests(unittest.TestCase):
    def test_defaults_and_restricted_urls(self):
        args = monitor.parse_args([])
        self.assertEqual(args.timeout, 180)
        self.assertEqual(args.attempts, 2)
        for url in ("file:///secret", "https://user:pass@example.com", "https://example.com/?token=x"):
            with self.subTest(url=url), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                monitor.parse_args(["--url", url])


@unittest.skipUnless(os.environ.get("RUN_HEALTH_BROWSER_TESTS") == "1", "Set RUN_HEALTH_BROWSER_TESTS=1 with Playwright/Chromium installed")
class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from playwright.async_api import async_playwright
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        self.log = monitor.Diagnostics()

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()

    async def check(self, html, timeout=1):
        await self.page.set_content(html)
        return await monitor.wait_for_app(self.page, time.monotonic() + timeout, self.log)

    async def test_healthy_and_partial_ui(self):
        result = await self.check(READY)
        self.assertEqual(result.condition, "already_awake")
        self.assertTrue(result.healthy)
        result = await self.check('<h2>Key findings</h2>')
        self.assertEqual(result.condition, "interface_not_loaded")
        self.assertFalse(result.healthy)

    async def test_embedded_app_and_delayed_initialization(self):
        await self.page.route("https://fixture.test/app", lambda route: route.fulfill(body=READY, content_type="text/html"))
        result = await self.check('<iframe title="streamlitApp" src="https://fixture.test/app"></iframe>', timeout=5)
        self.assertTrue(result.healthy)

    async def test_wake_button_and_unknown_sleep_ui(self):
        handler = escape('document.body.innerHTML=' + json.dumps(READY), quote=True)
        html = SLEEP + f'<button onclick="{handler}">Yes, get this app back up!</button>'
        result = await self.check(html, timeout=5)
        self.assertEqual(result.condition, "awakened")
        self.assertTrue(result.wake_clicked)
        result = await self.check(SLEEP + '<button>Unrecognized future wake control</button>')
        self.assertEqual(result.condition, "sleeping_unable_to_wake")
        self.assertFalse(result.wake_clicked)

    async def test_wake_click_does_not_loop(self):
        result = await self.check(SLEEP + '<button onclick="this.dataset.clicks=Number(this.dataset.clicks||0)+1">Yes, get this app back up!</button>')
        self.assertEqual(result.condition, "sleeping_unable_to_wake")
        self.assertEqual(await self.page.locator('button').get_attribute('data-clicks'), '1')

    async def test_errors_and_access_gates_override_readiness(self):
        for gate, expected in [('<div data-testid="stException">Traceback</div>', 'application_error'),
                               ('<h1>You do not have access</h1>', 'access_restricted'),
                               ('<h1>Verify you are human</h1>', 'access_restricted')]:
            with self.subTest(expected=expected):
                result = await self.check(READY + gate)
                self.assertEqual(result.condition, expected)
                self.assertFalse(result.healthy)

    async def test_http_200_shell_server_errors_and_dns(self):
        for status, html, expected in [(200, '<h1>Loading</h1>', 'interface_not_loaded'),
                                       (503, '<h1>Unavailable</h1>', 'service_unavailable'),
                                       (403, '<h1>Forbidden</h1>', 'access_restricted'),
                                       (404, '<h1>Not found</h1>', 'deployment_unavailable')]:
            async def respond(route):
                await route.fulfill(status=status, body=html, content_type='text/html')
            await self.page.route('https://fixture.test/', respond)
            result = await monitor.check_attempt(self.page, 'https://fixture.test/', time.monotonic() + 1, self.log)
            self.assertEqual(result.condition, expected)
            await self.page.unroute('https://fixture.test/')
        await self.page.route('https://fixture.test/', lambda route: route.abort('namenotresolved'))
        result = await monitor.check_attempt(self.page, 'https://fixture.test/', time.monotonic() + 1, self.log)
        self.assertEqual(result.condition, 'network_failure')

    async def test_navigation_timeout_keeps_waiting_without_refresh(self):
        from playwright.async_api import TimeoutError
        await self.page.set_content(READY)
        with patch.object(self.page, 'goto', AsyncMock(side_effect=TimeoutError('fixture timeout'))) as navigation:
            result = await monitor.check_attempt(self.page, 'https://fixture.test/', time.monotonic() + 1, self.log)
            self.assertTrue(result.healthy)
            navigation.assert_awaited_once()
        await self.page.set_content('<h1>Loading</h1>')
        with patch.object(self.page, 'goto', AsyncMock(side_effect=TimeoutError('fixture timeout'))):
            result = await monitor.check_attempt(self.page, 'https://fixture.test/', time.monotonic() + 1, self.log)
            self.assertEqual(result.condition, 'browser_navigation_timeout')

    async def test_exit_codes_retry_and_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            args = monitor.parse_args(['--artifacts-dir', folder, '--timeout', '10'])
            with patch.object(monitor, 'check_attempt', AsyncMock(side_effect=[monitor.Result('network_failure'), monitor.Result('already_awake', True)])) as attempt:
                self.assertEqual(await monitor.run(args), 0)
                self.assertEqual(attempt.await_count, 2)
                self.assertFalse(list(Path(folder).iterdir()))
            with patch.object(monitor, 'check_attempt', AsyncMock(return_value=monitor.Result('interface_not_loaded'))):
                self.assertEqual(await monitor.run(args), 1)
                self.assertTrue((Path(folder) / 'failure.png').is_file())
                report = json.loads((Path(folder) / 'diagnostics.json').read_text())
                self.assertEqual(report['result']['condition'], 'interface_not_loaded')
            with patch.object(monitor, 'check_attempt', AsyncMock(return_value=monitor.Result('sleeping_unable_to_wake', wake_clicked=True))) as attempt:
                self.assertEqual(await monitor.run(args), 1)
                self.assertEqual(attempt.await_count, 1)
            with patch.object(monitor, 'check_attempt', AsyncMock(return_value=monitor.Result('access_restricted'))):
                self.assertEqual(await monitor.run(args), 1)
                self.assertFalse((Path(folder) / 'failure.png').exists())


if __name__ == '__main__':
    unittest.main()
