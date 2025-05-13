import unittest
import numpy as np
import torch
import pandas as pd

# Assuming img_ops.py is in src/HiTMicTools/img_processing/
# Adjust the import path based on your project structure and how tests are run.
from HiTMicTools.img_processing.img_ops import (
    detect_and_fix_well,
    clear_background,
    convert_to_uint8,
    norm_eq_hist,
    crop_black_region,
    measure_background_intensity,
    dynamic_resize_roi,
)
from HiTMicTools.utils import round_to_odd, unit_converter # Imported by img_ops

class TestImgOps(unittest.TestCase):

    def test_detect_and_fix_well(self):
        # Image with no border
        img_no_border = np.ones((100, 100), dtype=np.float32) * 100
        result_img, has_border = detect_and_fix_well(img_no_border.copy())
        self.assertFalse(has_border)
        np.testing.assert_array_equal(result_img, img_no_border)

        # Image with a clear dark border
        img_with_border = np.ones((100, 100), dtype=np.float32) * 100
        img_with_border[0:10, :] = 10  # Top border
        img_with_border[:, -10:] = 10 # Right border
        
        result_img_fixed, has_border_fixed = detect_and_fix_well(img_with_border.copy(), darkness_threshold_factor=0.5)
        self.assertTrue(has_border_fixed)
        # Check that border pixels are changed (not 10 anymore)
        self.assertTrue(np.all(result_img_fixed[0:10, :] != 10))
        self.assertTrue(np.all(result_img_fixed[:, -10:] != 10))
        # Check that non-border pixels are largely unchanged (or changed to mean)
        # The fixed value should be close to the mean of the non-border area.
        non_border_original_mean = np.mean(img_with_border[10:, :-10])
        self.assertAlmostEqual(result_img_fixed[5,5], non_border_original_mean, delta=1.0)


    def test_clear_background(self):
        img = np.ones((50, 50), dtype=np.float32) * 100
        img[10:40, 10:40] = 200  # Brighter square

        # Test subtract method
        result_subtract = clear_background(img.copy(), sigma_r=11, method="subtract")
        self.assertEqual(result_subtract.shape, img.shape)
        self.assertTrue(np.all(result_subtract >= 0)) # Due to clip_negative=True

        # Test divide method
        result_divide = clear_background(img.copy(), sigma_r=11, method="divide")
        self.assertEqual(result_divide.shape, img.shape)
        # Values in divide method can be small, check general properties
        self.assertTrue(result_divide.mean() > 0)

        # Test unit conversion for sigma_r
        result_physical_unit = clear_background(img.copy(), sigma_r=5, unit="um", pixel_size=0.5, method="subtract")
        # Expected sigma_r in pixels = 5um / 0.5 um/pixel = 10, rounded to odd = 11
        # This implicitly tests if unit_converter and round_to_odd are working as expected by clear_background
        self.assertEqual(result_physical_unit.shape, img.shape)

        with self.assertRaises(ValueError):
            clear_background(img.copy(), sigma_r=11, method="invalid_method")
        with self.assertRaises(ValueError): # Test 3D image
            clear_background(np.zeros((2,2,2)), sigma_r=3)


    def test_convert_to_uint8(self):
        img_float = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32) * 200 # Range 0 to 200
        result = convert_to_uint8(img_float.copy())
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 255)
        np.testing.assert_array_equal(result, np.array([[0, 127], [191, 255]], dtype=np.uint8))

        img_zero_variance = np.ones((10,10)) * 50
        with self.assertRaisesRegex(ValueError, "Image has zero variance; cannot normalize."):
            convert_to_uint8(img_zero_variance)

    def test_norm_eq_hist(self):
        # Create an image with some variation
        img = np.arange(256, dtype=np.uint8).reshape(16, 16).astype(np.float32)
        img = img / img.max() * 150 + 50 # Ensure values are spread but not full 0-255 initially
        
        result = norm_eq_hist(img.copy())
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(result.mean(), 0.0, places=5)
        self.assertAlmostEqual(result.std(), 1.0, places=5)

    def test_crop_black_region(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[2:8, 3:7] = 100  # Non-black region: y from 2 to 7, x from 3 to 6
        y0, y1, x0, x1 = crop_black_region(img.copy())
        self.assertEqual((y0, y1, x0, x1), (2, 8, 3, 7))

        img_all_black = np.zeros((5,5), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Image is completely black."):
            crop_black_region(img_all_black)
            
        img_no_black = np.ones((5,5), dtype=np.uint8) * 5
        y0_nb, y1_nb, x0_nb, x1_nb = crop_black_region(img_no_black.copy())
        self.assertEqual((y0_nb, y1_nb, x0_nb, x1_nb), (0, 5, 0, 5))


    def test_measure_background_intensity(self):
        # img: [frame, slice, channel, y, x]
        img_stack = np.zeros((2, 1, 1, 10, 10), dtype=np.float32)
        img_stack[0, 0, 0, :, :] = 50  # Frame 0, all 50
        img_stack[1, 0, 0, :, :] = 100 # Frame 1, all 100

        # mask: [frame, slice, y, x]
        mask_stack = np.zeros((2, 1, 10, 10), dtype=np.uint8)
        mask_stack[0, 0, 2:5, 2:5] = 1 # Object in frame 0
        # Frame 1 has no object in mask, so background is whole image

        df_bck = measure_background_intensity(img_stack.copy(), mask_stack.copy(), target_channel=0, quantile=0.5) # Median
        
        self.assertIsInstance(df_bck, pd.DataFrame)
        self.assertEqual(len(df_bck), 2)
        self.assertAlmostEqual(df_bck.loc[df_bck['frame'] == 0, 'background'].iloc[0], 50.0)
        self.assertAlmostEqual(df_bck.loc[df_bck['frame'] == 1, 'background'].iloc[0], 100.0)

    def test_dynamic_resize_roi(self):
        min_size = 64

        # Test case 1: Image smaller than min_size (should pad)
        small_img_np = np.ones((32, 48), dtype=np.float32)
        small_img_torch = torch.from_numpy(small_img_np)
        resized_small = dynamic_resize_roi(small_img_torch.clone(), min_size)
        self.assertEqual(resized_small.shape, (min_size, min_size))
        # Check if original content is centered
        pad_h_top = (min_size - 32) // 2
        pad_w_left = (min_size - 48) // 2
        self.assertTrue(torch.allclose(resized_small[pad_h_top:pad_h_top+32, pad_w_left:pad_w_left+48], small_img_torch))


        # Test case 2: Image larger than min_size (should resize and pad if necessary)
        large_img_np = np.ones((128, 96), dtype=np.float32) # Aspect ratio 4:3
        large_img_torch = torch.from_numpy(large_img_np)
        resized_large = dynamic_resize_roi(large_img_torch.clone(), min_size)
        self.assertEqual(resized_large.shape, (min_size, min_size))
        # Max dim (128) will be scaled to min_size (64). Scale = 0.5
        # New H = 128*0.5 = 64. New W = 96*0.5 = 48.
        # Then (64,48) will be padded to (64,64).

        # Test case 3: Image is 3D (Z, H, W)
        img_3d_np = np.ones((3, 32, 48), dtype=np.float32)
        img_3d_torch = torch.from_numpy(img_3d_np)
        resized_3d = dynamic_resize_roi(img_3d_torch.clone(), min_size)
        self.assertEqual(resized_3d.shape, (min_size, min_size))
        # Content should be from the first slice and padded
        first_slice_padded_manually = dynamic_resize_roi(img_3d_torch[0].clone(), min_size)
        self.assertTrue(torch.allclose(resized_3d, first_slice_padded_manually))

        # Test case 4: Image already min_size
        exact_img_torch = torch.ones((min_size, min_size), dtype=torch.float32)
        resized_exact = dynamic_resize_roi(exact_img_torch.clone(), min_size)
        self.assertEqual(resized_exact.shape, (min_size, min_size))
        self.assertTrue(torch.allclose(resized_exact, exact_img_torch))


if __name__ == "__main__":
    unittest.main()