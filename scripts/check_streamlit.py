"""Read-only browser health check with normal Community Cloud wake-up interaction.

No Streamlit/ML imports. Run with --help for time budgets and diagnostic options.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit


DEFAULT_URL = "https://solar-power-prediction-python.streamlit.app/"
WAKE = re.compile(r"^(?:Yes,?\s*get this app back up!?|Wake (?:up|app)|Wake this app(?: up)?)[.!]?$", re.I)
SLEEP = re.compile(r"(?:app has gone to sleep|app is sleeping|app is hibernating|sleeping due to inactivity)", re.I)
RESTRICTED = re.compile(r"(?:You do not have access|You don['’]t have access|Request access to this app|Sign in to view this app|This app is private|Verify you are human|Access denied)", re.I)
AUTH_HEADING = re.compile(r"^(?:Sign in|Log in|Login|Authentication required|Permission denied)(?: to .*)?$", re.I)
STARTUP_ERROR = re.compile(r"(?:Error running app|Application error|Oh no[!.]?|This app has encountered an error|This research view is unavailable|Verified manuscript evidence is unavailable)", re.I)
TRANSIENT = {"network_failure", "browser_navigation_timeout", "service_unavailable"}


@dataclass
class Result:
    condition: str
    healthy: bool = False
    wake_clicked: bool = False
    attempts: int = 1
    detail: str = ""


class Diagnostics:
    """Keep bounded metadata, never cookies, headers, HTML, console text or URL queries."""

    def __init__(self):
        self.events = []

    def record(self, event, **metadata):
        item = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **metadata}
        if len(self.events) < 200:
            self.events.append(item)
        print(json.dumps(item), flush=True)


async def visible(locator):
    for match in await locator.all():
        if await match.is_visible():
            return match
    return None


async def inspect_page(page):
    """Errors/access gates take precedence over partial or stale readiness UI."""
    ready, sleeping, wake = False, False, None
    for frame in list(page.frames):
        if frame.is_detached():
            continue
        try:
            if await visible(frame.get_by_text(RESTRICTED)) or await visible(frame.get_by_role("heading", name=AUTH_HEADING)):
                return "access_restricted", None
            if await visible(frame.locator('[data-testid="stException"]')) or await visible(frame.get_by_text(STARTUP_ERROR)):
                return "application_error", None
            sleeping = sleeping or bool(await visible(frame.get_by_text(SLEEP)))
            button = await visible(frame.get_by_role("button", name=WAKE))
            if button is not None:
                wake = button
                sleeping = True
            ready = ready or bool(
                await visible(frame.get_by_role("radiogroup", name="Research dashboard", exact=True))
                and await visible(frame.get_by_role("heading", name="Key findings", exact=True))
                and await visible(frame.get_by_role("button", name="Read the Manuscript", exact=True))
            )
        except Exception as exc:
            from playwright.async_api import Error
            if not isinstance(exc, Error) or not (frame.is_detached() or "Execution context was destroyed" in str(exc)):
                raise
    if sleeping:
        return "sleeping", wake
    return ("ready" if ready else "loading"), None


async def wait_for_app(page, deadline, diagnostics, *, navigation_timed_out=False):
    wake_clicked, saw_sleep = False, False
    changed = asyncio.Event()
    def notify(*_):
        changed.set()
    for event in ("domcontentloaded", "load", "frameattached", "framenavigated"):
        page.on(event, notify)
    try:
        while time.monotonic() < deadline:
            changed.clear()
            state, button = await inspect_page(page)
            if state in ("access_restricted", "application_error"):
                return Result(state, wake_clicked=wake_clicked,
                              detail="Visible access gate or application error; no bypass attempted.")
            if state == "ready":
                return Result("awakened" if wake_clicked else "already_awake", True, wake_clicked,
                              detail="Overview navigation, findings and manuscript action are visible in one frame.")
            if state == "sleeping":
                if not saw_sleep:
                    diagnostics.record("sleep_screen_detected")
                saw_sleep = True
                if button is not None and not wake_clicked and await button.is_enabled():
                    diagnostics.record("normal_wake_button_click")
                    # At most one wake click per run. Never refresh during initialization.
                    wake_clicked = True
                    try:
                        await button.click(timeout=min(5000, max(1, (deadline - time.monotonic()) * 1000)))
                    except Exception as exc:
                        from playwright.async_api import Error
                        if not isinstance(exc, Error):
                            raise
                        return Result("sleeping_unable_to_wake", wake_clicked=True,
                                      detail="Recognized wake button could not be clicked; no second click attempted.")
            # Explicit condition polling: wake on page/frame events, or recheck the
            # visible DOM every 500ms. No fixed cold-start sleep or networkidle wait.
            try:
                await asyncio.wait_for(changed.wait(), timeout=min(.5, max(.001, deadline - time.monotonic())))
            except asyncio.TimeoutError:
                pass
        if saw_sleep:
            return Result("sleeping_unable_to_wake", wake_clicked=wake_clicked,
                          detail="Sleep screen detected, but the real Overview never became ready before the deadline.")
        return Result("browser_navigation_timeout" if navigation_timed_out else "interface_not_loaded",
                      detail="Readiness not verified; this alone does not confirm an application failure.")
    finally:
        for event in ("domcontentloaded", "load", "frameattached", "framenavigated"):
            page.remove_listener(event, notify)


async def check_attempt(page, url, deadline, diagnostics):
    from playwright.async_api import Error, TimeoutError as NavigationTimeout

    navigation_timed_out = False
    try:
        response = await page.goto(url, wait_until="domcontentloaded",
                                   timeout=min(30000, max(1, (deadline - time.monotonic()) * 1000)))
        if response:
            diagnostics.record("navigation_response", status=response.status)
            if response.status in (401, 403):
                return Result("access_restricted", detail="Navigation returned HTTP 401/403.")
            if response.status >= 500:
                state, _ = await inspect_page(page)
                return Result("application_error" if state == "application_error" else "service_unavailable",
                              detail="Navigation returned a server error; a proxy/hosting error is possible.")
            if response.status >= 400:
                return Result("deployment_unavailable", detail=f"Navigation returned HTTP {response.status}.")
    except NavigationTimeout:
        # A browser navigation timeout is not proof of app failure. The DOM may
        # still initialize; use the remaining budget without reloading it.
        diagnostics.record("navigation_timeout_waiting_for_interface")
        navigation_timed_out = True
    except Error as exc:
        if any(token in str(exc) for token in ("ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION", "ERR_PROXY_CONNECTION", "ERR_NETWORK", "ERR_INTERNET", "ERR_TIMED_OUT")):
            return Result("network_failure", detail="Browser reported DNS/connectivity failure; app health is unknown.")
        return Result("browser_check_error", detail="Browser navigation failed; app health is unknown.")
    return await wait_for_app(page, deadline, diagnostics, navigation_timed_out=navigation_timed_out)


async def save_failure(page, folder, result, diagnostics):
    folder.mkdir(parents=True, exist_ok=True)
    # Do not accidentally include an older run's screenshot in this report.
    (folder / "failure.png").unlink(missing_ok=True)
    if page is not None and not page.is_closed():
        # Fresh anonymous context and public page only. Avoid capturing access gates.
        if result.condition != "access_restricted":
            try:
                await page.screenshot(path=str(folder / "failure.png"), timeout=5000)
            except Exception:
                diagnostics.record("screenshot_unavailable")
        diagnostics.record("frame_summary", frames=[
            {"scheme": urlsplit(f.url).scheme, "host": urlsplit(f.url).hostname}
            for f in page.frames
        ])
    (folder / "diagnostics.json").write_text(
        json.dumps({"result": asdict(result), "events": diagnostics.events}, indent=2), encoding="utf-8")


async def run(args):
    diagnostics = Diagnostics()
    deadline = time.monotonic() + args.timeout
    page, browser, result = None, None, Result("browser_check_error", detail="Browser could not start; app health is unknown.")
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(headless=True, timeout=min(30000, args.timeout * 1000))
                context = await browser.new_context(viewport={"width": 1280, "height": 900})
                page = await context.new_page()
                page.on("console", lambda message: diagnostics.record("browser_console", level=message.type) if message.type == "error" else None)
                page.on("pageerror", lambda _: diagnostics.record("browser_page_error"))
                page.on("requestfailed", lambda request: diagnostics.record(
                    "request_failed", host=urlsplit(request.url).hostname, resource_type=request.resource_type))
                for attempt in range(1, args.attempts + 1):
                    diagnostics.record("attempt_started", attempt=attempt)
                    result = await check_attempt(page, args.url, deadline, diagnostics)
                    result.attempts = attempt
                    diagnostics.record("attempt_finished", **asdict(result))
                    if result.healthy or result.wake_clicked or result.condition not in TRANSIENT or attempt == args.attempts:
                        break
                    backoff = 2 ** attempt
                    if time.monotonic() + backoff >= deadline:
                        break
                    diagnostics.record("transient_retry_backoff", seconds=backoff)
                    await asyncio.sleep(backoff)
                if not result.healthy:
                    await save_failure(page, args.artifacts_dir, result, diagnostics)
            except Exception as exc:
                diagnostics.record("browser_check_exception", exception_type=type(exc).__name__)
                result = Result("browser_check_error", detail="Browser check failed; this does not confirm app failure.")
                await save_failure(page, args.artifacts_dir, result, diagnostics)
            finally:
                if browser:
                    await browser.close()
    except ImportError:
        diagnostics.record("missing_playwright", install="python -m pip install -r scripts/requirements-health-check.txt")
        await save_failure(None, args.artifacts_dir, result, diagnostics)
    diagnostics.record("final_result", **asdict(result))
    return 0 if result.healthy else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=180, help="Total navigation/wake/readiness budget in seconds (default: 180).")
    parser.add_argument("--attempts", type=int, choices=(1, 2, 3), default=2, help="Maximum attempts for transient failures only, within the total budget.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("health-check-artifacts"))
    args = parser.parse_args(argv)
    target = urlsplit(args.url)
    if target.scheme not in ("http", "https") or not target.hostname or target.username or target.password or target.query or target.fragment:
        parser.error("Use an HTTP(S) public/local URL without credentials, query parameters or fragments.")
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
