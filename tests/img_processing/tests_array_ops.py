import unittest
import numpy as np
import itertools # Used by stack_indexer in the source file

# Assuming array_ops.py is in src/HiTMicTools/img_processing/
# This import path needs to be correct based on how tests are run
# For example, if tests are run from project root and src is in PYTHONPATH:
from HiTMicTools.img_processing.array_ops import (
    adjust_dimensions,
    stack_indexer,
    get_bit_depth,
    convert_image,
)

class TestArrayOps(unittest.TestCase):

    def test_adjust_dimensions(self):
        # Test adding dimensions to a 2D image
        img_2d = np.zeros((10, 20)) # XY
        result_2d = adjust_dimensions(img_2d, "XY")
        self.assertEqual(result_2d.shape, (1, 1, 1, 10, 20)) # TSCXY

        # Test reordering and adding dimensions for a 3D image
        img_3d_txy = np.zeros((5, 10, 20)) # TXY
        result_3d_txy = adjust_dimensions(img_3d_txy, "TXY")
        self.assertEqual(result_3d_txy.shape, (5, 1, 1, 10, 20)) # TSCXY

        # Test complex reordering for a 5D image
        img_5d_ctsyx = np.zeros((3, 5, 2, 20, 10)) # CTSYX
        result_5d_ctsyx = adjust_dimensions(img_5d_ctsyx, "CTSYX")
        self.assertEqual(result_5d_ctsyx.shape, (5, 2, 3, 10, 20)) # TSCXY

        # Test with an invalid dimension order string
        with self.assertRaises(AssertionError):
            adjust_dimensions(img_2d, "XYZ") # Z is not in TSCXY

    def test_stack_indexer(self):
        # Default values
        np.testing.assert_array_equal(stack_indexer(), np.array([[0, 0, 0]]))

        # Lists and ranges
        expected_lists = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
        np.testing.assert_array_equal(stack_indexer([0, 1], 1, range(2)), expected_lists)

        # Error handling: negative integers
        with self.assertRaisesRegex(ValueError, "Dimensions must be positive integers or lists."):
            stack_indexer(-1, 0, 0)
        with self.assertRaisesRegex(ValueError, "All elements in the list dimensions must be positive integers."):
            stack_indexer([-1], 0, 0)
        
        # Error handling: invalid types
        with self.assertRaisesRegex(TypeError, "All dimensions must be either positive integers or lists of positive integers."):
            stack_indexer("0", 1, 2)

    def test_get_bit_depth(self):
        self.assertEqual(get_bit_depth(np.zeros((1,1), dtype=np.uint8)), 8)
        self.assertEqual(get_bit_depth(np.zeros((1,1), dtype=np.uint16)), 16)
        self.assertEqual(get_bit_depth(np.zeros((1,1), dtype=np.float32)), 32)
        self.assertEqual(get_bit_depth(np.zeros((1,1), dtype=np.int64)), 64)
        with self.assertRaises(KeyError): # For unsupported dtypes
            get_bit_depth(np.zeros((1,1), dtype=np.complex64))

    def test_convert_image(self):
        # Simple 3D image (e.g., HWC or SXY for channel scaling)
        # Using a small range to make scaling predictable
        img_float = np.array([[[0.0, 0.5], [0.25, 0.75]], [[0.5, 1.0], [0.75, 0.25]]], dtype=np.float32) # (2,2,2)
        # For channel scaling, let's assume last dim is channel if img is 3D
        # img_float shape (S,X,C) = (2,2,2) or (H,W,C)

        # Convert to uint8 with global scaling
        # Global min=0, max=1. Scaled: (arr - 0)/(1-0) = arr. Then *255.
        converted_u8_global = convert_image(img_float.copy(), np.uint8, scale_mode="global")
        self.assertEqual(converted_u8_global.dtype, np.uint8)
        self.assertEqual(converted_u8_global.min(), 0)
        self.assertEqual(converted_u8_global.max(), 255)
        np.testing.assert_array_almost_equal(converted_u8_global, (img_float * 255).astype(np.uint8))

        # Convert to uint16 with channel scaling (assuming img_float is S,X,C or H,W,C)
        # Channel 0: min=0, max=0.75. Scaled: (arr-0)/0.75. Then *65535
        # Channel 1: min=0.25, max=1.0. Scaled: (arr-0.25)/0.75. Then *65535
        converted_u16_channel = convert_image(img_float.copy(), np.uint16, scale_mode="channel")
        self.assertEqual(converted_u16_channel.dtype, np.uint16)
        # Check that each channel was scaled independently (min 0, max 65535 per channel after scaling)
        for c in range(img_float.shape[2]):
            channel_data_original = img_float[:,:,c]
            channel_data_converted = converted_u16_channel[:,:,c]
            if channel_data_original.min() < channel_data_original.max(): # Avoid division by zero if flat
                 self.assertAlmostEqual(channel_data_converted.min(), 0, delta=1) # allow for rounding
                 self.assertAlmostEqual(channel_data_converted.max(), 65535, delta=1)


        # Convert to float32 (should just scale, no type change if already float32)
        # Using a uint8 input to test initial type conversion and scaling
        img_uint8_input = np.array([[[0, 128], [64, 192]], [[128, 255], [192, 64]]], dtype=np.uint8)
        converted_f32_global = convert_image(img_uint8_input.copy(), np.float32, scale_mode="global")
        self.assertEqual(converted_f32_global.dtype, np.float32)
        self.assertAlmostEqual(converted_f32_global.min(), 0.0)
        self.assertAlmostEqual(converted_f32_global.max(), 1.0)
        
        # Test unsupported scale_mode
        with self.assertRaisesRegex(ValueError, "Unsupported scale mode: invalid_mode"):
            convert_image(img_float.copy(), np.uint8, scale_mode="invalid_mode")

        # Test unsupported target dtype
        with self.assertRaisesRegex(ValueError, "Unsupported data type: <class 'numpy.int64'>"):
            convert_image(img_float.copy(), np.int64, scale_mode="global")

if __name__ == "__main__":
    unittest.main()