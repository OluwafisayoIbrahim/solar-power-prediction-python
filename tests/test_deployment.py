"""Regression checks for spreadsheet parsing and row alignment."""

from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deployment")]
from batch_prediction import (FEATURES, chart_data, download_csv, measured_pairs,
                              read_sheet, run_batch)


class SpreadsheetTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([[850, 31, 65, 12], [720, 29, 70, 13]], columns=FEATURES)

    def test_csv_and_selected_excel_sheet(self):
        csv = self.frame.rename(columns={"solar_irradiance": " Solar Irradiance "}).to_csv(index=False).encode()
        pd.testing.assert_frame_equal(read_sheet(csv, "data.csv"), self.frame)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"wrong": [1]}).to_excel(writer, sheet_name="Other", index=False)
            self.frame.to_excel(writer, sheet_name="Readings", index=False)
        pd.testing.assert_frame_equal(read_sheet(buffer.getvalue(), "data.xlsx", "Readings"), self.frame)

    def test_missing_duplicate_empty_and_reserved_columns(self):
        for data in [b"temperature\n31\n", b"hour, Hour \n12,13\n",
                     b"solar_irradiance,temperature,humidity,hour\n",
                     self.frame.assign(predicted_power_w=1).to_csv(index=False).encode()]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                read_sheet(data, "data.csv")

    def test_invalid_rows_are_retained_without_predictions(self):
        frame = pd.concat([self.frame] * 3, ignore_index=True).astype(object)
        frame.loc[1, "humidity"] = 101
        frame.loc[2, "hour"] = 24
        frame.loc[3, "temperature"] = "bad"
        frame.loc[4, "solar_irradiance"] = np.inf
        frame["power"] = [100, 200, 300, 400, 500, "bad"]
        with patch("batch_prediction.predict_measurements", return_value=np.array([11., 22.])) as predictor:
            result = run_batch(frame, object())
        self.assertEqual(predictor.call_args.args[1].index.tolist(), [0, 5])
        self.assertEqual(result["source_row"].tolist(), list(range(2, 8)))
        self.assertEqual(result.loc[[0, 5], "predicted_power_w"].tolist(), [11, 22])
        self.assertTrue(result.loc[1:4, "predicted_power_w"].isna().all())
        self.assertEqual(len(measured_pairs(result)), 1)
        self.assertEqual(len(pd.read_csv(BytesIO(download_csv(result)))), len(frame))

    def test_chunked_inference_and_all_invalid(self):
        frame = pd.concat([self.frame] * 1100, ignore_index=True)
        with patch("batch_prediction.predict_measurements", side_effect=lambda _, rows: rows["hour"].to_numpy()) as predictor:
            result = run_batch(frame, None)
            self.assertEqual(predictor.call_count, 3)
            np.testing.assert_allclose(result["predicted_power_w"], frame["hour"])
        with patch("batch_prediction.predict_measurements") as predictor:
            result = run_batch(self.frame.assign(hour=-1), None)
            predictor.assert_not_called()
            self.assertTrue(result["predicted_power_w"].isna().all())

    def test_timestamp_order_does_not_change_download_order(self):
        frame = self.frame.assign(timestamp=["2026-10-17T12:00:00Z", "2026-10-16T13:00:00Z"])
        with patch("batch_prediction.predict_measurements", return_value=np.array([11., 22.])):
            result = run_batch(frame, None)
        chart, x = chart_data(result, "timestamp")
        self.assertEqual(chart["Predicted (W)"].tolist(), [22, 11])
        self.assertEqual(result["predicted_power_w"].tolist(), [11, 22])
        self.assertEqual(x, "Timestamp (UTC)")
        with self.assertRaises(ValueError):
            chart_data(result.assign(timestamp="bad"), "timestamp")
        with self.assertRaises(ValueError):
            chart_data(result.assign(timestamp="12:00"), "timestamp")


if __name__ == "__main__":
    unittest.main()
