"""Tests for ScSegmenter: clump merge, priority overwrite, bundle round-trip, backward compat.

All tests are CPU-safe (no GPU required) and mock the model backend so they can
run on a login node without downloading weights.
"""

import os
import tempfile
import unittest
import zipfile

import numpy as np
import torch
import yaml
from unittest.mock import patch, MagicMock

from HiTMicTools.model_components.scsegmenter import ScSegmenter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_segmenter(**kwargs):
    """Instantiate a ScSegmenter with BacDETRSeg mocked away."""
    defaults = dict(
        model_path="/fake/model.pth",
        class_dict={0: "single-cell", 1: "clump", 2: "debris"},
        compile_mode=False,
    )
    defaults.update(kwargs)
    with patch("HiTMicTools.model_components.scsegmenter.BacDETRSeg") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.model.model.eval = MagicMock()
        mock_cls.return_value = mock_instance
        seg = ScSegmenter(**defaults)
    return seg


# ---------------------------------------------------------------------------
# Initialization & parameter storage
# ---------------------------------------------------------------------------

class TestScSegmenterInit(unittest.TestCase):
    """Constructor stores parameters and validates inputs."""

    def test_tuned_defaults(self):
        """Verify that constructor defaults match tuned production values."""
        seg = _mock_segmenter()
        self.assertEqual(seg.tile_size, 256)
        self.assertAlmostEqual(seg.overlap_ratio, 0.33)
        self.assertAlmostEqual(seg.score_threshold, 0.4)
        self.assertAlmostEqual(seg.nms_iou, 0.4)
        self.assertEqual(seg.clump_merge_min_overlap, 250)
        self.assertAlmostEqual(seg.priority_overlap_fraction, 0.5)
        self.assertEqual(seg.temporal_buffer_size, 8)
        self.assertEqual(seg.batch_size, 128)
        self.assertAlmostEqual(seg.mask_threshold, 0.5)

    def test_custom_parameters_stored(self):
        seg = _mock_segmenter(
            patch_size=512,
            overlap_ratio=0.25,
            score_threshold=0.6,
            nms_iou=0.3,
            clump_merge_min_overlap=100,
            priority_overlap_fraction=0.7,
            temporal_buffer_size=4,
            batch_size=64,
            mask_threshold=0.4,
        )
        self.assertEqual(seg.tile_size, 512)
        self.assertAlmostEqual(seg.overlap_ratio, 0.25)
        self.assertAlmostEqual(seg.score_threshold, 0.6)
        self.assertAlmostEqual(seg.nms_iou, 0.3)
        self.assertEqual(seg.clump_merge_min_overlap, 100)
        self.assertAlmostEqual(seg.priority_overlap_fraction, 0.7)
        self.assertEqual(seg.temporal_buffer_size, 4)
        self.assertEqual(seg.batch_size, 64)
        self.assertAlmostEqual(seg.mask_threshold, 0.4)

    def test_model_type_stored(self):
        seg = _mock_segmenter()
        self.assertEqual(seg.model_type, "rfdetrsegpreview")

    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _mock_segmenter(model_type="nonexistent")
        self.assertIn("Unsupported segmenter type", str(ctx.exception))

    def test_invalid_overlap_ratio_raises(self):
        with self.assertRaises(AssertionError):
            _mock_segmenter(overlap_ratio=1.5)

    def test_invalid_priority_overlap_fraction_raises(self):
        with self.assertRaises(AssertionError):
            _mock_segmenter(priority_overlap_fraction=0.0)
        with self.assertRaises(AssertionError):
            _mock_segmenter(priority_overlap_fraction=1.5)


class TestScSegmenterValidation(unittest.TestCase):
    """Validate input parameter edge cases."""

    def test_negative_patch_size(self):
        with self.assertRaises(AssertionError):
            _mock_segmenter(patch_size=-1)

    def test_zero_batch_size(self):
        with self.assertRaises(AssertionError):
            _mock_segmenter(batch_size=0)

    def test_mask_threshold_boundary(self):
        with self.assertRaises(AssertionError):
            _mock_segmenter(mask_threshold=0.0)
        with self.assertRaises(AssertionError):
            _mock_segmenter(mask_threshold=1.0)


# ---------------------------------------------------------------------------
# Backward compatibility: old bundles without new fields
# ---------------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    """Old bundle configs that lack new fields should still load successfully."""

    def test_old_config_without_priority_overlap_fraction(self):
        """Bundles created before priority_overlap_fraction was a kwarg should
        default to 0.5 (the tuned production value)."""
        old_inferer_args = {
            "patch_size": 256,
            "overlap_ratio": 0.25,
            "score_threshold": 0.5,
            "nms_iou": 0.5,
            "clump_merge_min_overlap": 10,
            "temporal_buffer_size": 8,
            "batch_size": 32,
            "mask_threshold": 0.5,
            "class_dict": {0: "single-cell", 1: "clump", 2: "debris"},
        }
        seg = _mock_segmenter(**old_inferer_args)
        # Should use new default
        self.assertAlmostEqual(seg.priority_overlap_fraction, 0.5)
        # Old values should be respected
        self.assertAlmostEqual(seg.overlap_ratio, 0.25)
        self.assertAlmostEqual(seg.score_threshold, 0.5)

    def test_new_config_with_all_fields(self):
        """Bundles with all new fields should propagate them correctly."""
        new_inferer_args = {
            "patch_size": 256,
            "overlap_ratio": 0.33,
            "score_threshold": 0.4,
            "nms_iou": 0.4,
            "clump_merge_min_overlap": 250,
            "priority_overlap_fraction": 0.5,
            "temporal_buffer_size": 8,
            "batch_size": 128,
            "mask_threshold": 0.5,
            "class_dict": {0: "single-cell", 1: "clump", 2: "debris"},
        }
        seg = _mock_segmenter(**new_inferer_args)
        self.assertEqual(seg.clump_merge_min_overlap, 250)
        self.assertAlmostEqual(seg.priority_overlap_fraction, 0.5)
        self.assertEqual(seg.batch_size, 128)


# ---------------------------------------------------------------------------
# Clump merge behavior (_union_merge_clumps)
# ---------------------------------------------------------------------------

class TestClumpMerge(unittest.TestCase):
    """Test global mask-based clump merging logic."""

    def test_no_clumps_returns_empty(self):
        seg = _mock_segmenter()
        boxes = torch.empty((0, 4), dtype=torch.float32)
        scores = torch.empty((0,), dtype=torch.float32)
        merged_boxes, merged_scores, merged_masks, merged_offsets = seg._union_merge_clumps(
            boxes, scores, None, None
        )
        self.assertEqual(merged_boxes.shape[0], 0)

    def test_single_clump_unchanged(self):
        seg = _mock_segmenter(clump_merge_min_overlap=5)
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        scores = torch.tensor([0.9])
        mask = torch.ones((1, 40, 40), dtype=torch.float32)
        offsets = torch.tensor([[10, 10]], dtype=torch.long)

        merged_boxes, merged_scores, merged_masks, merged_offsets = seg._union_merge_clumps(
            boxes, scores, mask, offsets
        )
        self.assertEqual(merged_boxes.shape[0], 1)
        self.assertEqual(len(merged_masks), 1)

    def test_overlapping_clumps_merged(self):
        """Two clumps with sufficient mask overlap should be merged into one."""
        seg = _mock_segmenter(clump_merge_min_overlap=5, mask_threshold=0.5)

        # Two 40x40 boxes that overlap by 20px in x
        boxes = torch.tensor([
            [0.0, 0.0, 40.0, 40.0],
            [20.0, 0.0, 60.0, 40.0],
        ])
        scores = torch.tensor([0.8, 0.9])

        # Both masks are fully filled — they overlap in [20:40, 0:40] in global coords
        mask1 = torch.ones((40, 40), dtype=torch.float32)
        mask2 = torch.ones((40, 40), dtype=torch.float32)
        masks = torch.stack([mask1, mask2])
        offsets = torch.tensor([[0, 0], [20, 0]], dtype=torch.long)

        merged_boxes, merged_scores, merged_masks, merged_offsets = seg._union_merge_clumps(
            boxes, scores, masks, offsets
        )
        # Should merge into 1 group
        self.assertEqual(merged_boxes.shape[0], 1)
        self.assertEqual(len(merged_masks), 1)
        # Union box should span [0, 0, 60, 40]
        self.assertAlmostEqual(merged_boxes[0, 0].item(), 0.0)
        self.assertAlmostEqual(merged_boxes[0, 2].item(), 60.0)

    def test_non_overlapping_clumps_separate(self):
        """Two clumps that don't overlap should remain separate."""
        seg = _mock_segmenter(clump_merge_min_overlap=5, mask_threshold=0.5)

        boxes = torch.tensor([
            [0.0, 0.0, 30.0, 30.0],
            [100.0, 100.0, 130.0, 130.0],
        ])
        scores = torch.tensor([0.8, 0.9])
        masks = torch.ones((2, 30, 30), dtype=torch.float32)
        offsets = torch.tensor([[0, 0], [100, 100]], dtype=torch.long)

        merged_boxes, merged_scores, merged_masks, merged_offsets = seg._union_merge_clumps(
            boxes, scores, masks, offsets
        )
        self.assertEqual(merged_boxes.shape[0], 2)

    def test_merge_respects_min_overlap(self):
        """Overlap below threshold should not trigger merging."""
        seg = _mock_segmenter(clump_merge_min_overlap=100, mask_threshold=0.5)

        boxes = torch.tensor([
            [0.0, 0.0, 40.0, 40.0],
            [39.0, 0.0, 79.0, 40.0],
        ])
        scores = torch.tensor([0.8, 0.9])
        mask1 = torch.ones((40, 40), dtype=torch.float32)
        mask2 = torch.ones((40, 40), dtype=torch.float32)
        masks = torch.stack([mask1, mask2])
        offsets = torch.tensor([[0, 0], [39, 0]], dtype=torch.long)

        merged_boxes, merged_scores, merged_masks, merged_offsets = seg._union_merge_clumps(
            boxes, scores, masks, offsets
        )
        # Only 40 pixels of overlap (1 column * 40 rows), less than threshold of 100
        self.assertEqual(merged_boxes.shape[0], 2)


# ---------------------------------------------------------------------------
# Priority overwrite behavior (_stitch_masks)
# ---------------------------------------------------------------------------

class TestPriorityOverwrite(unittest.TestCase):
    """Test class-priority overwrite during mask stitching."""

    def test_higher_priority_overwrites_when_above_threshold(self):
        """A clump (priority 3) should overwrite a single-cell (priority 1)
        when overlap >= priority_overlap_fraction of the existing label."""
        seg = _mock_segmenter(priority_overlap_fraction=0.5, mask_threshold=0.5)

        height, width = 100, 100
        # Single-cell mask at (10, 10)–(30, 30)
        sc_mask = torch.ones((20, 20), dtype=torch.float32)

        # Clump mask that overlaps >50% of the single-cell
        clump_mask = torch.ones((30, 30), dtype=torch.float32)

        # Place single-cell first (lower priority, class 0)
        # Then place clump overlapping it (higher priority, class 1)
        masks = [sc_mask, clump_mask]
        offsets = [(10, 10), (5, 5)]  # clump overlaps the sc region
        classes = [0, 1]

        labeled_mask, detection_map = seg._stitch_masks(
            masks, offsets, classes, (height, width)
        )

        # The clump should have taken over since it overlaps >50% of the single-cell
        unique_labels = np.unique(labeled_mask)
        self.assertIn(0, unique_labels)
        # Overwritten overlap region should be filled by the incoming clump, not background.
        self.assertEqual(np.count_nonzero(labeled_mask[10:30, 10:30] == 0), 0)

    def test_lower_priority_does_not_overwrite(self):
        """A single-cell should not overwrite a clump regardless of overlap."""
        seg = _mock_segmenter(priority_overlap_fraction=0.5, mask_threshold=0.5)

        height, width = 100, 100
        clump_mask = torch.ones((20, 20), dtype=torch.float32)
        sc_mask = torch.ones((20, 20), dtype=torch.float32)

        masks = [clump_mask, sc_mask]
        offsets = [(10, 10), (10, 10)]  # same position, complete overlap
        classes = [1, 0]  # clump first, then single-cell

        labeled_mask, detection_map = seg._stitch_masks(
            masks, offsets, classes, (height, width)
        )

        # Single-cell should get zero new pixels (blocked by clump)
        unique_non_zero = np.unique(labeled_mask[labeled_mask > 0])
        self.assertEqual(len(unique_non_zero), 1)

    def test_below_threshold_preserves_existing(self):
        """If overlap is below priority_overlap_fraction, existing label is preserved."""
        seg = _mock_segmenter(priority_overlap_fraction=0.9, mask_threshold=0.5)

        height, width = 200, 200
        sc_mask = torch.ones((50, 50), dtype=torch.float32)  # 2500 pixels
        clump_mask = torch.ones((10, 10), dtype=torch.float32)  # 100 pixels

        masks = [sc_mask, clump_mask]
        offsets = [(10, 10), (10, 10)]
        classes = [0, 1]

        labeled_mask, detection_map = seg._stitch_masks(
            masks, offsets, classes, (height, width)
        )

        # With 0.9 threshold, the 100/2500=4% overlap is far below 90%
        sc_area = (labeled_mask[10:60, 10:60] > 0).sum()
        self.assertGreater(sc_area, 2000)


# ---------------------------------------------------------------------------
# Bundle inferer_args round-trip
# ---------------------------------------------------------------------------

class TestBundleInfererArgsRoundTrip(unittest.TestCase):
    """Verify that inferer_args survive a bundle create -> extract -> load cycle."""

    def test_round_trip(self):
        """Write inferer_args into a bundle config, extract, and verify all fields."""
        inferer_args = {
            "patch_size": 256,
            "overlap_ratio": 0.33,
            "score_threshold": 0.4,
            "nms_iou": 0.4,
            "clump_merge_min_overlap": 250,
            "priority_overlap_fraction": 0.5,
            "temporal_buffer_size": 8,
            "batch_size": 128,
            "mask_threshold": 0.5,
            "class_dict": {0: "single-cell", 1: "clump", 2: "debris"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "_bundle_metadata": {"creation_date": "2025-01-01"},
                "sc_segmenter": {
                    "model_path": "models/RFDETRSegm-sc_segmenter.pth",
                    "model_metadata": "metadata/RFDETRSegm-sc_segmenter.json",
                    "inferer_args": inferer_args,
                },
            }

            bundle_path = os.path.join(tmpdir, "test_bundle.zip")
            with zipfile.ZipFile(bundle_path, "w") as zf:
                zf.writestr("config.yml", yaml.dump(config))

            with zipfile.ZipFile(bundle_path, "r") as zf:
                loaded_config = yaml.safe_load(zf.read("config.yml"))

            loaded_args = loaded_config["sc_segmenter"]["inferer_args"]

            for key, expected in inferer_args.items():
                self.assertEqual(
                    loaded_args[key], expected,
                    f"inferer_args['{key}'] mismatch: {loaded_args[key]} != {expected}",
                )

    def test_old_bundle_missing_new_fields(self):
        """An older bundle without priority_overlap_fraction should still be
        usable -- ScSegmenter will use its default."""
        old_inferer_args = {
            "patch_size": 256,
            "overlap_ratio": 0.25,
            "score_threshold": 0.5,
            "nms_iou": 0.5,
            "clump_merge_min_overlap": 10,
            "temporal_buffer_size": 8,
            "batch_size": 32,
            "mask_threshold": 0.5,
            "class_dict": {0: "single-cell", 1: "clump", 2: "debris"},
        }

        seg = _mock_segmenter(**old_inferer_args)
        self.assertAlmostEqual(seg.priority_overlap_fraction, 0.5)
        self.assertEqual(seg.batch_size, 32)
        self.assertEqual(seg.clump_merge_min_overlap, 10)


if __name__ == "__main__":
    unittest.main()
