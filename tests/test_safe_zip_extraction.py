"""Tests for safe model-bundle ZIP extraction."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from HiTMicTools.pipelines.base_pipeline import _safe_extract_zip


class TestSafeZipExtraction(unittest.TestCase):
    """Validate that model bundle extraction rejects unsafe paths."""

    def test_rejects_parent_directory_member(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "bad_bundle.zip"
            extract_dir = Path(tmp_dir) / "extract"
            outside_path = Path(tmp_dir) / "outside.txt"

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("config.yml", "{}")
                zf.writestr("../outside.txt", "should not be written")

            with zipfile.ZipFile(zip_path, "r") as zf:
                with self.assertRaisesRegex(ValueError, "Unsafe path"):
                    _safe_extract_zip(zf, str(extract_dir))

            self.assertFalse(outside_path.exists())

    def test_extracts_normal_bundle_members(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "bundle.zip"
            extract_dir = Path(tmp_dir) / "extract"

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("config.yml", "{}")
                zf.writestr("models/model.pth", "weights")
                zf.writestr("metadata/model.json", "{}")

            with zipfile.ZipFile(zip_path, "r") as zf:
                _safe_extract_zip(zf, str(extract_dir))

            self.assertTrue((extract_dir / "config.yml").exists())
            self.assertTrue((extract_dir / "models" / "model.pth").exists())
            self.assertTrue((extract_dir / "metadata" / "model.json").exists())


if __name__ == "__main__":
    unittest.main()
