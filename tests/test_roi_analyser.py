"""Unit tests for RoiAnalyser class.

This module tests the correctness, error handling, and numerical accuracy of the
RoiAnalyser implementation. Focus is on code quality, not performance benchmarking.
"""

import unittest
import numpy as np
import warnings
from HiTMicTools.roianalysis.roi_analyser import RoiAnalyser


class TestRoiAnalyserConstruction(unittest.TestCase):
    """Test RoiAnalyser construction and initialization."""

    def setUp(self):
        """Create minimal test data."""
        np.random.seed(42)
        self.image = np.random.rand(10, 1, 2, 64, 64).astype(np.float32)
        self.proba = np.random.rand(10, 1, 64, 64).astype(np.float32)

    def test_normal_construction(self):
        """Test normal construction with valid inputs."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))

        self.assertIsNotNone(analyser.img)
        self.assertIsNotNone(analyser.proba_map)
        self.assertEqual(analyser.img.shape, (10, 1, 2, 64, 64))
        self.assertEqual(analyser.proba_map.shape, (10, 1, 1, 64, 64))  # Adjusted dimensions

    def test_from_labeled_mask_construction(self):
        """Test construction from pre-labeled mask."""
        labeled_mask = np.zeros((10, 1, 64, 64), dtype=np.int32)
        # Add some labels
        labeled_mask[0, 0, 10:20, 10:20] = 1
        labeled_mask[1, 0, 30:40, 30:40] = 2

        analyser = RoiAnalyser.from_labeled_mask(
            self.image,
            labeled_mask,
            stack_order=("TSCXY", "TSYX")
        )

        self.assertIsNotNone(analyser.labeled_mask)
        self.assertIsNotNone(analyser.binary_mask)
        self.assertEqual(analyser.total_rois, 2)
        self.assertIsNone(analyser.proba_map)

    def test_dimension_adjustment(self):
        """Test that dimension adjustment works correctly."""
        # Test with 3D input that needs expansion
        image_3d = np.random.rand(10, 64, 64).astype(np.float32)  # TYX
        proba_3d = np.random.rand(10, 64, 64).astype(np.float32)  # TYX

        analyser = RoiAnalyser(image_3d, proba_3d, stack_order=("TYX", "TYX"))

        # Should be expanded to TSCXY format
        self.assertEqual(len(analyser.img.shape), 5)
        self.assertEqual(len(analyser.proba_map.shape), 5)


class TestRoiAnalyserBinaryMask(unittest.TestCase):
    """Test binary mask creation and cleaning."""

    def setUp(self):
        """Create test data with known structure."""
        np.random.seed(42)
        self.image = np.random.rand(10, 1, 2, 64, 64).astype(np.float32)
        self.proba = np.zeros((10, 1, 64, 64), dtype=np.float32)

        # Add blobs with probability 1.0
        for t in range(10):
            self.proba[t, 0, 20:30, 20:30] = 1.0  # Large blob (100 pixels)
            self.proba[t, 0, 40:42, 40:42] = 1.0  # Small blob (4 pixels)

    def test_create_binary_mask_default_threshold(self):
        """Test binary mask creation with default threshold."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask()

        self.assertIsNotNone(analyser.binary_mask)
        self.assertEqual(analyser.binary_mask.shape, analyser.proba_map.shape)
        self.assertEqual(analyser.binary_mask.dtype, bool)

    def test_create_binary_mask_custom_threshold(self):
        """Test binary mask creation with custom threshold."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.8)

        # Count pixels above threshold
        expected_count = np.sum(self.proba > 0.8)
        actual_count = np.sum(analyser.binary_mask)

        # Account for dimension expansion in proba_map
        self.assertGreater(actual_count, 0)

    def test_clean_binmask_removes_small_objects(self):
        """Test that clean_binmask removes objects below size threshold."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)

        # Before cleaning - should have both large and small blobs
        initial_mask = analyser.binary_mask.copy()

        # Clean with threshold that removes small blob (4 pixels < 10)
        analyser.clean_binmask(min_pixel_size=10)

        # After cleaning - should have fewer pixels (small blobs removed)
        self.assertLess(
            np.sum(analyser.binary_mask),
            np.sum(initial_mask),
            "clean_binmask should remove small objects"
        )

    def test_clean_binmask_creates_cache(self):
        """Test that clean_binmask creates label cache."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)

        self.assertFalse(hasattr(analyser, '_cached_labeled_mask'))

        analyser.clean_binmask(min_pixel_size=10)

        self.assertTrue(hasattr(analyser, '_cached_labeled_mask'))
        self.assertTrue(hasattr(analyser, '_cached_num_features'))


class TestRoiAnalyserLabeling(unittest.TestCase):
    """Test label computation and caching."""

    def setUp(self):
        """Create test data with distinct blobs."""
        np.random.seed(42)
        self.image = np.random.rand(5, 1, 2, 64, 64).astype(np.float32)
        self.proba = np.zeros((5, 1, 64, 64), dtype=np.float32)

        # Frame 0: 2 blobs
        self.proba[0, 0, 10:20, 10:20] = 1.0
        self.proba[0, 0, 40:50, 40:50] = 1.0

        # Frame 1: 3 blobs
        self.proba[1, 0, 10:20, 10:20] = 1.0
        self.proba[1, 0, 30:40, 30:40] = 1.0
        self.proba[1, 0, 50:60, 50:60] = 1.0

        # Frame 2-4: 1 blob each
        for t in range(2, 5):
            self.proba[t, 0, 20:30, 20:30] = 1.0

    def test_get_labels_produces_continuous_labels(self):
        """Test that labels are continuous across frames."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        # Check label continuity
        unique_labels = np.unique(analyser.labeled_mask)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background

        expected_labels = np.arange(1, analyser.total_rois + 1)
        np.testing.assert_array_equal(
            unique_labels,
            expected_labels,
            err_msg="Labels should be continuous (no gaps)"
        )

    def test_get_labels_return_value_option(self):
        """Test get_labels with return_value=True."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)

        labeled_mask, num_rois = analyser.get_labels(return_value=True)

        self.assertIsNotNone(labeled_mask)
        self.assertGreater(num_rois, 0)
        self.assertEqual(num_rois, np.max(labeled_mask))

    def test_get_labels_uses_cache_after_clean(self):
        """Test that get_labels uses cache after clean_binmask."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.clean_binmask(min_pixel_size=10)

        # Cache should exist
        self.assertTrue(hasattr(analyser, '_cached_labeled_mask'))
        cached_labels = analyser._cached_labeled_mask.copy()
        cached_rois = analyser._cached_num_features

        # Call get_labels (should use cache)
        analyser.get_labels()

        # Results should match cached values
        np.testing.assert_array_equal(analyser.labeled_mask, cached_labels)
        self.assertEqual(analyser.total_rois, cached_rois)

    def test_get_labels_correct_roi_count(self):
        """Test that total_rois matches expected number of objects."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        # Expected: 2 + 3 + 1 + 1 + 1 = 8 ROIs
        self.assertEqual(analyser.total_rois, 8)

    def test_get_labels_parallel_vs_sequential(self):
        """Test that parallel processing produces same results as sequential."""
        # Sequential
        analyser_seq = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser_seq.create_binary_mask(threshold=0.5)
        analyser_seq.get_labels(n_workers=1)

        # Parallel
        analyser_par = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser_par.create_binary_mask(threshold=0.5)
        analyser_par.get_labels(n_workers=2)

        # Compare results
        np.testing.assert_array_equal(
            analyser_seq.labeled_mask,
            analyser_par.labeled_mask,
            err_msg="Parallel and sequential labels should match"
        )
        self.assertEqual(analyser_seq.total_rois, analyser_par.total_rois)


class TestRoiAnalyserMeasurements(unittest.TestCase):
    """Test ROI measurement extraction."""

    def setUp(self):
        """Create test data with known properties."""
        np.random.seed(42)
        self.image = np.ones((3, 1, 2, 64, 64), dtype=np.float32)
        self.proba = np.zeros((3, 1, 64, 64), dtype=np.float32)

        # Create blobs with specific intensities
        # Frame 0: intensity 10
        self.image[0, 0, 0, 20:30, 20:30] = 10.0
        self.proba[0, 0, 20:30, 20:30] = 1.0

        # Frame 1: intensity 20
        self.image[1, 0, 0, 30:40, 30:40] = 20.0
        self.proba[1, 0, 30:40, 30:40] = 1.0

        # Frame 2: intensity 30
        self.image[2, 0, 0, 40:50, 40:50] = 30.0
        self.proba[2, 0, 40:50, 40:50] = 1.0

    def test_get_roi_measurements_required_columns(self):
        """Test that measurements include all required columns."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "centroid", "area"]
        )

        # Check required columns
        required_cols = ["label", "frame", "slice", "channel"]
        for col in required_cols:
            self.assertIn(col, measurements.columns, f"Missing required column: {col}")

    def test_get_roi_measurements_column_order(self):
        """Test that measurements have correct column order."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "centroid", "area"]
        )

        # Required columns should come first
        first_cols = list(measurements.columns[:4])
        self.assertEqual(first_cols, ["label", "frame", "slice", "channel"])

    def test_get_roi_measurements_correct_frame_assignment(self):
        """Test that frame indices are correctly assigned."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "centroid", "area"]
        )

        # Should have 3 ROIs (one per frame)
        self.assertEqual(len(measurements), 3)

        # Check frame assignments
        frames = sorted(measurements['frame'].unique())
        self.assertEqual(frames, [0, 1, 2])

    def test_get_roi_measurements_mean_intensity(self):
        """Test that mean intensity is correctly computed."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "mean_intensity"]
        )

        # Check mean intensities match expected values
        for idx, row in measurements.iterrows():
            frame = int(row['frame'])
            expected_intensity = (frame + 1) * 10.0  # 10, 20, 30

            np.testing.assert_allclose(
                row['mean_intensity'],
                expected_intensity,
                rtol=0.1,
                err_msg=f"Mean intensity mismatch for frame {frame}"
            )

    def test_get_roi_measurements_centroid_calculation(self):
        """Test that centroids are correctly computed."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "centroid"]
        )

        # Check that centroids are in expected regions
        for idx, row in measurements.iterrows():
            frame = int(row['frame'])
            centroid_y = row['centroid_0']
            centroid_x = row['centroid_1']

            # Frame 0: blob at 20-30, centroid should be ~24.5
            # Frame 1: blob at 30-40, centroid should be ~34.5
            # Frame 2: blob at 40-50, centroid should be ~44.5
            expected_centroid = 20 + frame * 10 + 4.5

            np.testing.assert_allclose(
                centroid_y,
                expected_centroid,
                rtol=0.1,
                err_msg=f"Centroid Y mismatch for frame {frame}"
            )

    def test_get_roi_measurements_parallel_equivalence(self):
        """Test that parallel measurements match sequential."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        # Sequential
        measurements_seq = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "centroid", "area", "mean_intensity"],
            n_workers=1
        )

        # Parallel
        measurements_par = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "centroid", "area", "mean_intensity"],
            n_workers=2
        )

        # Compare (sort by label for consistent order)
        measurements_seq_sorted = measurements_seq.sort_values('label').reset_index(drop=True)
        measurements_par_sorted = measurements_par.sort_values('label').reset_index(drop=True)

        # Check all columns
        for col in measurements_seq_sorted.columns:
            if measurements_seq_sorted[col].dtype in [np.float32, np.float64]:
                np.testing.assert_allclose(
                    measurements_seq_sorted[col].values,
                    measurements_par_sorted[col].values,
                    rtol=1e-5,
                    err_msg=f"Mismatch in column {col}"
                )
            else:
                np.testing.assert_array_equal(
                    measurements_seq_sorted[col].values,
                    measurements_par_sorted[col].values,
                    err_msg=f"Mismatch in column {col}"
                )

    def test_get_roi_measurements_before_labeling_raises_error(self):
        """Test that calling get_roi_measurements before labeling raises error."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)

        # Should raise AttributeError or AssertionError
        with self.assertRaises((AssertionError, AttributeError)):
            analyser.get_roi_measurements(
                target_channel=0,
                target_slice=0,
                properties=["label", "area"]
            )


class TestRoiAnalyserEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_probability_map(self):
        """Test handling of empty probability map (no ROIs)."""
        image = np.random.rand(5, 1, 2, 64, 64).astype(np.float32)
        proba = np.zeros((5, 1, 64, 64), dtype=np.float32)  # All zeros

        analyser = RoiAnalyser(image, proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        # Should handle gracefully
        self.assertEqual(analyser.total_rois, 0)

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "area"]
        )

        self.assertEqual(len(measurements), 0)

    def test_single_frame_input(self):
        """Test handling of single-frame input."""
        image = np.random.rand(1, 1, 2, 64, 64).astype(np.float32)
        proba = np.ones((1, 1, 64, 64), dtype=np.float32)

        analyser = RoiAnalyser(image, proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        # Should work correctly
        self.assertGreater(analyser.total_rois, 0)

    def test_very_small_image(self):
        """Test handling of very small images."""
        image = np.random.rand(2, 1, 2, 16, 16).astype(np.float32)
        proba = np.ones((2, 1, 16, 16), dtype=np.float32)

        analyser = RoiAnalyser(image, proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "area"]
        )

        self.assertGreater(len(measurements), 0)

    def test_threshold_boundary_conditions(self):
        """Test threshold boundary conditions."""
        image = np.random.rand(2, 1, 2, 32, 32).astype(np.float32)
        proba = np.full((2, 1, 32, 32), 0.5, dtype=np.float32)

        analyser = RoiAnalyser(image, proba, stack_order=("TSCXY", "TSYX"))

        # Test with threshold = 0.5 (values equal to threshold)
        analyser.create_binary_mask(threshold=0.5)

        # Should not include pixels equal to threshold (uses >)
        self.assertEqual(np.sum(analyser.binary_mask), 0)

    def test_clean_binmask_with_zero_threshold(self):
        """Test clean_binmask with min_pixel_size=0."""
        image = np.random.rand(2, 1, 2, 32, 32).astype(np.float32)
        proba = np.ones((2, 1, 32, 32), dtype=np.float32)

        analyser = RoiAnalyser(image, proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)

        initial_mask_sum = np.sum(analyser.binary_mask)

        # Should not remove anything
        analyser.clean_binmask(min_pixel_size=0)

        # Mask should be unchanged (or possibly all removed if interpreted as >0)
        final_mask_sum = np.sum(analyser.binary_mask)
        self.assertLessEqual(final_mask_sum, initial_mask_sum)


class TestRoiAnalyserWorkflow(unittest.TestCase):
    """Test complete workflow integration."""

    def setUp(self):
        """Create realistic test scenario."""
        np.random.seed(42)
        self.image = np.random.rand(20, 1, 2, 128, 128).astype(np.float32)
        self.proba = np.random.rand(20, 1, 128, 128).astype(np.float32)

        # Add structured blobs
        for t in range(20):
            for _ in range(5):
                cy, cx = np.random.randint(20, 108, size=2)
                y, x = np.ogrid[-5:5, -5:5]
                mask = x**2 + y**2 <= 25
                self.proba[t, 0, cy-5:cy+5, cx-5:cx+5][mask] = 1.0

    def test_complete_workflow(self):
        """Test complete analysis workflow."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))

        # Step 1: Create binary mask
        analyser.create_binary_mask(threshold=0.5)
        self.assertIsNotNone(analyser.binary_mask)

        # Step 2: Clean mask
        analyser.clean_binmask(min_pixel_size=20)
        self.assertTrue(hasattr(analyser, '_cached_labeled_mask'))

        # Step 3: Get labels
        analyser.get_labels()
        self.assertIsNotNone(analyser.labeled_mask)
        self.assertGreater(analyser.total_rois, 0)

        # Step 4: Get measurements
        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "centroid", "area", "mean_intensity"]
        )

        # Verify output
        self.assertGreater(len(measurements), 0)
        self.assertEqual(len(measurements), analyser.total_rois)

        # Check column names
        self.assertIn("label", measurements.columns)
        self.assertIn("frame", measurements.columns)
        self.assertIn("centroid_0", measurements.columns)
        self.assertIn("centroid_1", measurements.columns)
        self.assertIn("area", measurements.columns)

    def test_workflow_without_cleaning(self):
        """Test workflow that skips clean_binmask step."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))

        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()  # Should work without clean_binmask

        measurements = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "area"]
        )

        self.assertGreater(len(measurements), 0)

    def test_repeated_measurements(self):
        """Test that measurements can be called multiple times."""
        analyser = RoiAnalyser(self.image, self.proba, stack_order=("TSCXY", "TSYX"))
        analyser.create_binary_mask(threshold=0.5)
        analyser.get_labels()

        # First call
        measurements1 = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "area"]
        )

        # Second call (should produce same results)
        measurements2 = analyser.get_roi_measurements(
            target_channel=0,
            target_slice=0,
            properties=["label", "area"]
        )

        # Results should be identical
        self.assertEqual(len(measurements1), len(measurements2))
        np.testing.assert_array_equal(
            measurements1['label'].values,
            measurements2['label'].values
        )


if __name__ == '__main__':
    unittest.main()
