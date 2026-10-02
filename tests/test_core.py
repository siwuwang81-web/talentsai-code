from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from probe_calibration.core import CalibrationResult, endpoint_candidates, load_points


class CoreTests(unittest.TestCase):
    def test_load_points_keeps_xyz_and_removes_non_finite_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "points.txt"
            input_path.write_text(
                "0 0 0 10\n1 2 3 11\nnan 4 5 12\n6 7 8 13\n",
                encoding="utf-8",
            )

            points = load_points(input_path)

        np.testing.assert_array_equal(
            points,
            np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),
        )

    def test_load_points_rejects_rows_without_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "points.txt"
            input_path.write_text("0 0\n1 1\n2 2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "X、Y、Z"):
                load_points(input_path)

    def test_endpoint_candidates_returns_original_opposite_points(self) -> None:
        points = np.array(
            [
                [-10.0, 0.0, 0.0],
                [-1.0, 0.2, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -0.2, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )

        first, second = endpoint_candidates(points)

        self.assertEqual({first, second}, {0, 4})
        np.testing.assert_array_equal(points[[first, second], 1:], 0.0)

    def test_endpoint_candidates_rejects_collapsed_cloud(self) -> None:
        with self.assertRaisesRegex(ValueError, "空间范围"):
            endpoint_candidates(np.zeros((3, 3)))

    def test_calibration_result_writes_utf8_json(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            result = CalibrationResult.from_selection(
                temp_path / "测杆.extract.txt", points, selected_index=1, other_index=0
            )
            output_path = temp_path / "标定结果.json"
            result.write_json(output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["contact_point"], [2.0, 0.0, 0.0])
        self.assertEqual(payload["other_endpoint_candidate"], [0.0, 0.0, 0.0])
        self.assertEqual(payload["distance_between_endpoint_candidates"], 2.0)
        self.assertIs(payload["contact_point_is_original_scan_point"], True)


if __name__ == "__main__":
    unittest.main()
