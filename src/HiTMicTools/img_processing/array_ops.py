import itertools
from typing import List, Union

import numpy as np


def adjust_dimensions(img: np.ndarray, dim_order: str) -> np.ndarray:
    """
    Adjust the dimensions of an image array to match the target order 'TSCXY'.

    Args:
        img (np.ndarray): Input image array.
        dim_order (str): Current dimension order of the image. Should be a permutation or subset of 'TSCXY'.
        T=Time, S=Slice, C=Channel, X=Width, Y=Height.

    Returns:
        np.ndarray: Image array with adjusted dimensions.
    """

    target_order = "TSCXY"
    assert set(dim_order).issubset(set(target_order)), (
        "Invalid dimension order. Allowed dimensions: 'TSCXY'"
    )

    missing_dims = set(target_order) - set(dim_order)

    # Add missing dimensions
    for dim in missing_dims:
        index = target_order.index(dim)
        img = np.expand_dims(img, axis=index)
        dim_order = dim_order[:index] + dim + dim_order[index:]

    # Reorder dimensions
    order = [dim_order.index(dim) for dim in target_order]
    img = np.transpose(img, order)

    return img


def stack_indexer(
    nframes: Union[int, List[int], range] = [0],
    nslices: Union[int, List[int], range] = [0],
    nchannels: Union[int, List[int], range] = [0],
) -> np.ndarray:
    """
    Generate an index table for accessing specific frames, slices, and channels in an image stack. This aims
    to simplify the process of iterating over different combinations of frame, slice, and channel indices with
    for loops.
    Args:
        nframes (Union[int, List[int], range], optional): Frame indices. Defaults to [0].
        nslices (Union[int, List[int], range], optional): Slice indices. Defaults to [0].
        nchannels (Union[int, List[int], range], optional): Channel indices. Defaults to [0].

    Returns:
        np.ndarray: Index table with shape (n_combinations, 3), where each row represents a combination
                   of frame, slice, and channel indices.

    Raises:
        ValueError: If any dimension contains negative integers.
        TypeError: If any dimension is not an integer, list of integers, or range object.
    """
    dimensions = []
    for dimension in [nframes, nslices, nchannels]:
        if isinstance(dimension, int):
            if dimension < 0:
                raise ValueError("Dimensions must be positive integers or lists.")
            dimensions.append([dimension])
        elif isinstance(dimension, (list, range)):
            if not all(isinstance(i, int) and i >= 0 for i in dimension):
                raise ValueError(
                    "All elements in the list dimensions must be positive integers."
                )
            dimensions.append(dimension)
        else:
            raise TypeError(
                "All dimensions must be either positive integers or lists of positive integers."
            )

    combinations = list(itertools.product(*dimensions))
    index_table = np.array(combinations)
    return index_table


def get_bit_depth(img: np.ndarray) -> int:
    """Get the bit depth of an image based on its data type."""
    dtype_to_bit_depth = {
        "uint8": 8,
        "uint16": 16,
        "uint32": 32,
        "uint64": 64,
        "int8": 8,
        "int16": 16,
        "int32": 32,
        "int64": 64,
        "float32": 32,
        "float64": 64,
    }

    bit_depth = dtype_to_bit_depth[str(img.dtype)]

    return bit_depth


def convert_image(
    img: np.ndarray, dtype: np.dtype, scale_mode: str = "channel"
) -> np.ndarray:
    """
    Convert an image to the specified data type with optional re-scaling.

    Args:
        img (np.ndarray): Input image array.
        dtype (np.dtype): Target data type for conversion.
        scale_mode (str, optional): Method for scaling the image. Defaults to 'channel'.

    Returns:
        np.ndarray: Converted image array.

    scale_mode options:
    - 'global': Scale entire image
    - 'channel': Scale each channel independently (default)
    - 'frame': Scale each frame independently
    - 'channel_frame': Scale each channel and frame independently
    """

    def scale_array(arr):
        """Normalize the provided array to the [0, 1] interval."""
        return (arr - arr.min()) / (arr.max() - arr.min())

    # Scale image to [0, 1] based on scale_mode
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32)
    else:
        img_scaled = img

    if scale_mode == "global":
        img_scaled = scale_array(img)
    elif scale_mode == "channel":
        img_scaled = np.stack(
            [scale_array(img[:, :, c]) for c in range(img.shape[2])], axis=2
        )
    elif scale_mode == "frame":
        img_scaled = np.stack(
            [scale_array(img[t]) for t in range(img.shape[0])], axis=0
        )
    elif scale_mode == "channel_frame":
        img_scaled = np.stack(
            [
                [scale_array(img[t, :, c]) for c in range(img.shape[2])]
                for t in range(img.shape[0])
            ],
            axis=(0, 2),
        )
    else:
        raise ValueError(f"Unsupported scale mode: {scale_mode}")

    # Convert image to the target data type
    if dtype == np.uint8:
        img_converted = (img_scaled * 255).astype(np.uint8)
    elif dtype == np.uint16:
        img_converted = (img_scaled * 65535).astype(np.uint16)
    elif dtype == np.float16 or dtype == np.float32:
        img_converted = img_scaled.astype(dtype)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    return img_converted
