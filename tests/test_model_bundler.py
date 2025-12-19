"""Tests for model bundle creation functionality."""

import os
import sys
import unittest
import tempfile
import shutil
import zipfile
import yaml
import json
import datetime
from pathlib import Path

from HiTMicTools.model_bundler import create_model_bundle


class TestModelBundler(unittest.TestCase):
    """Test model bundle creation functionality."""

    def setUp(self):
        """Create temporary directories for test fixtures."""
        self.test_dir = tempfile.mkdtemp(prefix="test_bundle_")
        self.models_dir = os.path.join(self.test_dir, "models")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.models_dir)
        os.makedirs(self.output_dir)

        # Create dummy model files
        self.dummy_model_path = os.path.join(self.models_dir, "dummy_model.pth")
        with open(self.dummy_model_path, 'w') as f:
            f.write("dummy model weights")

        self.dummy_pi_model_path = os.path.join(self.models_dir, "pi_model.pkl")
        with open(self.dummy_pi_model_path, 'w') as f:
            f.write("dummy pi classifier")

        # Create dummy metadata file
        self.dummy_metadata_path = os.path.join(self.models_dir, "metadata.json")
        with open(self.dummy_metadata_path, 'w') as f:
            json.dump({"model_type": "test", "version": "1.0"}, f)

        # Create minimal config YAML
        self.config_path = os.path.join(self.test_dir, "models_info.yml")
        config_data = {
            'bf_focus': {
                'model_path': self.dummy_model_path,
                'model_metadata': self.dummy_metadata_path,
                'inferer_args': {'patch_size': 256}
            },
            'pi_classification': {
                'model_path': self.dummy_pi_model_path,
                'model_metadata': self.dummy_metadata_path
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_basic_bundle_creation(self):
        """Test basic bundle creation without auto-dating."""
        output_path = os.path.join(self.output_dir, "test_bundle.zip")

        result_path = create_model_bundle(
            self.config_path,
            output_path,
            auto_date=False
        )

        # Check that bundle was created
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(zipfile.is_zipfile(result_path))

    def test_auto_dating(self):
        """Test that auto-dating inserts date into filename."""
        output_path = os.path.join(self.output_dir, "test_bundle.zip")

        result_path = create_model_bundle(
            self.config_path,
            output_path,
            auto_date=True
        )

        # Check that date was inserted
        today = datetime.date.today().strftime("%Y%m%d")
        expected_path = os.path.join(self.output_dir, f"test_bundle_{today}.zip")
        self.assertEqual(result_path, expected_path)
        self.assertTrue(os.path.exists(result_path))

    def test_bundle_structure(self):
        """Test that bundle has correct internal structure."""
        output_path = os.path.join(self.output_dir, "test_bundle.zip")

        create_model_bundle(
            self.config_path,
            output_path,
            auto_date=False
        )

        # Inspect zip contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            namelist = zf.namelist()

            # Check for required directories/files
            self.assertIn('config.yml', namelist)

            # Check for models directory
            model_files = [n for n in namelist if n.startswith('models/')]
            self.assertGreater(len(model_files), 0)

            # Check for metadata directory
            metadata_files = [n for n in namelist if n.startswith('metadata/')]
            self.assertGreater(len(metadata_files), 0)

    def test_creation_date_in_config(self):
        """Test that creation date metadata is added to config.yml."""
        output_path = os.path.join(self.output_dir, "test_bundle.zip")

        create_model_bundle(
            self.config_path,
            output_path,
            auto_date=False
        )

        # Extract and read config.yml
        with zipfile.ZipFile(output_path, 'r') as zf:
            config_data = yaml.safe_load(zf.read('config.yml'))

            # Check for bundle metadata
            self.assertIn('_bundle_metadata', config_data)
            self.assertIn('creation_date', config_data['_bundle_metadata'])
            self.assertIn('source_config', config_data['_bundle_metadata'])

            # Verify creation date format
            creation_date = config_data['_bundle_metadata']['creation_date']
            # Should be in format "YYYY-MM-DD HH:MM:SS"
            self.assertRegex(creation_date, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

    def test_parent_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        nested_output = os.path.join(self.output_dir, "nested", "dirs", "bundle.zip")

        result_path = create_model_bundle(
            self.config_path,
            nested_output,
            auto_date=False
        )

        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.exists(os.path.dirname(result_path)))

    def test_missing_config_file(self):
        """Test that missing config file raises FileNotFoundError."""
        output_path = os.path.join(self.output_dir, "bundle.zip")
        missing_config = os.path.join(self.test_dir, "nonexistent.yml")

        with self.assertRaises(FileNotFoundError):
            create_model_bundle(missing_config, output_path)

    def test_invalid_output_extension(self):
        """Test that non-.zip output path raises ValueError."""
        output_path = os.path.join(self.output_dir, "bundle.tar.gz")

        with self.assertRaises(ValueError):
            create_model_bundle(self.config_path, output_path)

    def test_pi_classifier_extension_preserved(self):
        """Test that PIClassifier keeps original extension (.pkl)."""
        output_path = os.path.join(self.output_dir, "test_bundle.zip")

        create_model_bundle(
            self.config_path,
            output_path,
            auto_date=False
        )

        # Check that PI classifier has .pkl extension in bundle
        with zipfile.ZipFile(output_path, 'r') as zf:
            namelist = zf.namelist()
            pi_files = [n for n in namelist if 'PIClassifier' in n and n.startswith('models/')]
            self.assertEqual(len(pi_files), 1)
            self.assertTrue(pi_files[0].endswith('.pkl'))

    def test_model_metadata_preserved(self):
        """Test that model metadata is correctly copied to bundle."""
        output_path = os.path.join(self.output_dir, "test_bundle.zip")

        create_model_bundle(
            self.config_path,
            output_path,
            auto_date=False
        )

        # Check metadata files
        with zipfile.ZipFile(output_path, 'r') as zf:
            # Read bf_focus metadata
            metadata_content = zf.read('metadata/NAFNet-bf_focus_restoration.json')
            metadata = json.loads(metadata_content)

            # Should have original_name and loaded metadata
            self.assertIn('original_name', metadata)
            self.assertEqual(metadata['original_name'], 'dummy_model.pth')
            self.assertIn('model_type', metadata)
            self.assertEqual(metadata['model_type'], 'test')


if __name__ == "__main__":
    unittest.main()
