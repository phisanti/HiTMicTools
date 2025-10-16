import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from HiTMicTools.model_components.oof_detector import OofDetector


class TestOofDetectorInit(unittest.TestCase):
    """Test OofDetector initialization and parameter handling."""

    @patch('HiTMicTools.model_components.oof_detector.RFDETRBase')
    @patch('HiTMicTools.model_components.oof_detector.torch.compile')
    def test_init_with_class_dict(self, mock_compile, mock_rfdetr):
        """Test initialization with class_dict parameter."""
        mock_model = MagicMock()
        mock_rfdetr.return_value = mock_model
        mock_compile.return_value = mock_model.model.model

        detector = OofDetector(
            model_path="/fake/path.pth",
            patch_size=560,
            class_dict={"oof": 0, "in_focus": 1},
            model_type="rfdetrbase"
        )

        # Verify RFDETRBase was called with num_classes=2
        mock_rfdetr.assert_called_once()
        call_kwargs = mock_rfdetr.call_args[1]
        self.assertEqual(call_kwargs['num_classes'], 2)

        # Verify torch.compile was called
        mock_compile.assert_called_once()

    @patch('HiTMicTools.model_components.oof_detector.RFDETRBase')
    @patch('HiTMicTools.model_components.oof_detector.torch.compile')
    @patch('HiTMicTools.model_components.oof_detector.torch.load')
    def test_init_without_class_dict(self, mock_load, mock_compile, mock_rfdetr):
        """Test initialization without class_dict (infers from checkpoint)."""
        # Mock checkpoint with 2 classes (class_bias has shape [3] -> 2 classes + background)
        mock_checkpoint = {
            'model': {
                'class_embed.bias': torch.zeros(3)
            }
        }
        mock_load.return_value = mock_checkpoint

        mock_model = MagicMock()
        mock_rfdetr.return_value = mock_model
        mock_compile.return_value = mock_model.model.model

        detector = OofDetector(
            model_path="/fake/path.pth",
            patch_size=560,
        )

        # Verify num_classes was inferred as 2 (3 - 1)
        call_kwargs = mock_rfdetr.call_args[1]
        self.assertEqual(call_kwargs['num_classes'], 2)

    @patch('HiTMicTools.model_components.oof_detector.RFDETRBase')
    @patch('HiTMicTools.model_components.oof_detector.torch.compile')
    def test_parameters_stored_correctly(self, mock_compile, mock_rfdetr):
        """Test that parameters are stored as attributes."""
        mock_model = MagicMock()
        mock_rfdetr.return_value = mock_model
        mock_compile.return_value = mock_model.model.model

        detector = OofDetector(
            model_path="/fake/path.pth",
            patch_size=640,
            overlap_ratio=0.3,
            score_threshold=0.6,
            nms_iou=0.4,
            class_dict={"oof": 0}
        )

        self.assertEqual(detector.tile_size, 640)
        self.assertEqual(detector.overlap_ratio, 0.3)
        self.assertEqual(detector.score_threshold, 0.6)
        self.assertEqual(detector.nms_iou, 0.4)


class TestOofDetectorValidation(unittest.TestCase):
    """Test OofDetector parameter validation."""

    @patch('HiTMicTools.model_components.oof_detector.RFDETRBase')
    def test_invalid_model_type(self, mock_rfdetr):
        """Test that invalid model_type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            OofDetector(
                model_path="/fake/path.pth",
                model_type="invalid_type",
                class_dict={"oof": 0}
            )
        self.assertIn("Unsupported detector type", str(context.exception))

    @patch('HiTMicTools.model_components.oof_detector.RFDETRBase')
    def test_invalid_overlap_ratio(self, mock_rfdetr):
        """Test that invalid overlap_ratio raises AssertionError."""
        with self.assertRaises(AssertionError):
            OofDetector(
                model_path="/fake/path.pth",
                overlap_ratio=1.5,  # > 1.0
                class_dict={"oof": 0}
            )

    @patch('HiTMicTools.model_components.oof_detector.RFDETRBase')
    def test_invalid_patch_size(self, mock_rfdetr):
        """Test that invalid patch_size raises AssertionError."""
        with self.assertRaises(AssertionError):
            OofDetector(
                model_path="/fake/path.pth",
                patch_size=-100,
                class_dict={"oof": 0}
            )


if __name__ == '__main__':
    unittest.main()
