import numpy as np
import itertools
import json
import pandas as pd
import ome_types
from datetime import timedelta
from typing import List, Union, Dict, Any


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
    assert set(dim_order).issubset(
        set(target_order)
    ), "Invalid dimension order. Allowed dimensions: 'TSCXY'"

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


def unit_converter(
    value: float, conversion_factor: float = 1, to_unit: str = "pixel"
) -> float:
    """
    Convert a value between pixels and micrometers (um) using a conversion factor.

    Args:
        value (float): The value to be converted.
        conversion_factor (float, optional, default 1): The conversion factor between pixels and micrometers.
        to_unit (str, optional, default pixel): The target unit for conversion. Can be either 'pixel' or 'um'.

    Returns:
        float: The converted value in the specified unit.

    Raises:
        ValueError: If an invalid unit is provided.
    """

    if to_unit == "um":
        return value * conversion_factor
    elif to_unit == "pixel":
        return value / conversion_factor
    else:
        raise ValueError("Invalid unit. Choose either 'um' or 'pixel'.")


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


def convert_image(img: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert an image to the specified data type."""

    # Scale image to [0, 1]
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32)
        img_shifted = img - np.min(img)
        img_scaled = img_shifted / np.max(img_shifted)
    else:
        img_scaled = img

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


def get_timestamps(
    metadata: ome_types.model.OME,
    ref_channel: int = 0,
    timeformat: str = "%Y-%m-%d %H:%M:%S",
) -> pd.DataFrame:
    """
    Extract timestamps from the metadata of an OME-TIFF file.

    Args:
        metadata (ome_types.model.OME): The metadata of the OME-TIFF file.
        ref_channel (int, optional): The reference channel for timestamps. Defaults to 0.
        timeformat (str, optional): The format string for the timestamp. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        pd.DataFrame: A DataFrame containing the timestamps for each frame.
    """
    base_time = metadata.images[0].acquisition_date
    z = metadata.images[0].pixels.planes

    timestamps = []
    for i in range(len(z)):
        timestamp = z[i].delta_t
        if z[i].the_c == ref_channel:
            timepoint = base_time + timedelta(seconds=timestamp)
            formatted_timepoint = timepoint.strftime(timeformat)
            timestep = timestamp

            timestamps.append(
                {"frame": i, "date_time": formatted_timepoint, "timestep": timestep}
            )
        else:
            continue
    df = pd.DataFrame(timestamps)

    df["timestep"] = df["timestep"] - df.loc[df["frame"] == 0, "timestep"].iloc[0]
    df["timestep"] = df["timestep"] / 3600000
    return df


def measure_background_intensity(stack: np.ndarray, channel: int) -> pd.DataFrame:
    """
    Measure the background intensity for each frame in a specific channel of an image stack.

    Args:
        stack (np.ndarray): The image stack with dimensions (frames, slices, channels, height, width).
        channel (int): The index of the channel to measure the background intensity.

    Returns:
        pd.DataFrame: A DataFrame containing the background intensity for each frame.
    """
    num_frames = stack.shape[0]
    background_intensities = []

    for frame in range(num_frames):
        channel_data = stack[frame, :, channel, :, :]
        mean_intensity = np.mean(channel_data)
        background_intensities.append(
            {"frame": frame, "background_intensity": mean_intensity}
        )

    df = pd.DataFrame(background_intensities)
    return df


def round_to_odd(number: float) -> int:
    """Round a number to the nearest odd integer."""
    return int(number) if number % 2 == 1 else int(number) + 1


def read_metadata(metadata_file: str) -> Dict[str, Any]:
    """Read metadata from a JSON file."""
    with open(metadata_file) as f:
        metadata = json.load(f)
    return metadata
