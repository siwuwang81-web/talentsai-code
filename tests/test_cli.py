import tempfile
import unittest
from pathlib import Path

from probe_calibration.cli import find_default_input


class CliTests(unittest.TestCase):
    def test_find_default_input_prefers_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            expected = temp_path / "测杆.extract.txt"
            expected.write_text("0 0 0\n", encoding="utf-8")

            self.assertEqual(find_default_input(temp_path), expected)

    def test_find_default_input_reports_missing_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "测杆.extract"):
                find_default_input(Path(directory))


if __name__ == "__main__":
    unittest.main()
