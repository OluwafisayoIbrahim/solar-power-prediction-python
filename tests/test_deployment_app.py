"""Streamlit integration checks using the saved Track B model."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class Upload(BytesIO):
    name = "measurements.csv"


class AppTests(unittest.TestCase):
    def make_app(self):
        return AppTest.from_file(str(ROOT / "deployment/prediction_workspace.py"), default_timeout=90)

    def assert_healthy(self, app):
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(len(app.error), 0, str(app.error))

    def test_single_and_model(self):
        app = self.make_app().run()
        self.assert_healthy(app)
        next(button for button in app.button if button.label == "Predict power").click().run()
        self.assert_healthy(app)
        self.assertTrue(any(metric.label == "Predicted power" for metric in app.metric))
        self.assertTrue(any(metric.label == "Parameters" for metric in app.metric))

    def test_upload_prediction_chart_and_stale_result_reset(self):
        uploaded = Upload(b"timestamp,solar_irradiance,temperature,humidity,hour,power\n"
                          b"2026-10-16T13:00:00Z,850,31,65,13,0\n"
                          b"2026-10-16T12:00:00Z,720,29,70,12,0\n"
                          b"bad,500,28,105,14,200\n")
        with patch("streamlit.file_uploader", return_value=uploaded):
            app = self.make_app().run()
            self.assert_healthy(app)
            next(button for button in app.button if button.label == "Predict uploaded readings").click().run()
            self.assert_healthy(app)
            result = app.session_state["batch_result"]
            self.assertEqual(result["prediction_status"].tolist(), ["predicted", "predicted", "invalid"])
            self.assertTrue(result["predicted_power_w"].iloc[:2].notna().all())
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(metrics["MAPE"], "N/A")
            self.assertEqual(metrics["R2"], "N/A")
            app.selectbox(key="plot_axis").select("timestamp").run()
            self.assert_healthy(app)
            self.assertTrue(any("ISO dates" in warning.value for warning in app.warning))
        with patch("streamlit.file_uploader", return_value=Upload(b"wrong\n1\n")):
            app.run()
            self.assertTrue(len(app.error))
            with self.assertRaises(KeyError):
                app.session_state["batch_result"]

    def test_excel_upload(self):
        output = BytesIO()
        pd.DataFrame({"solar_irradiance": [850], "temperature": [31],
                      "humidity": [65], "hour": [12]}).to_excel(output, index=False)
        uploaded = Upload(output.getvalue())
        uploaded.name = "measurements.xlsx"
        with patch("streamlit.file_uploader", return_value=uploaded):
            app = self.make_app().run()
            self.assert_healthy(app)
            next(button for button in app.button if button.label == "Predict uploaded readings").click().run()
            self.assert_healthy(app)
            self.assertEqual(app.session_state["batch_result"]["prediction_status"].iloc[0], "predicted")


if __name__ == "__main__":
    unittest.main()
