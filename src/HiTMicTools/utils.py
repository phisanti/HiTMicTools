# Standard library imports
import re
import json
import platform
import itertools
from datetime import timedelta
from typing import List, Union, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd
import ome_types
import psutil
import GPUtil
import torch


def remove_file_extension(filename: str) -> str:
    """
    Remove specific file extensions from a filename.
    
    Args:
        filename (str): The input filename.
    
    Returns:
        str: Filename with the extension removed.
    """
    extensions = [
        'nd2',
        'ome\.p\.tiff',
        'p\.tiff',
        'ome\.tiff',
        'tiff'
    ]
    pattern = r'\.(?:' + '|'.join(extensions) + ')$'
    return re.sub(pattern, '', filename)

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


def convert_image(img: np.ndarray, dtype: np.dtype, scale_mode: str = 'channel') -> np.ndarray:
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
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    # Scale image to [0, 1] based on scale_mode
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32)
    else:
        img_scaled = img

    if scale_mode == 'global':
        img_scaled = scale_array(img)
    elif scale_mode == 'channel':
        img_scaled = np.stack([scale_array(img[:, :, c]) for c in range(img.shape[2])], axis=2)
    elif scale_mode == 'frame':
        img_scaled = np.stack([scale_array(img[t]) for t in range(img.shape[0])], axis=0)
    elif scale_mode == 'channel_frame':
        img_scaled = np.stack([[scale_array(img[t, :, c]) for c in range(img.shape[2])] 
                                for t in range(img.shape[0])], axis=(0, 2))
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
            timepoint = base_time + timedelta(milliseconds=timestamp)
            formatted_timepoint = timepoint.strftime(timeformat)
            timestep = timestamp

            timestamps.append(
                {"frame": z[i].the_t, "date_time": formatted_timepoint, "timestep": timestep}
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


def empty_gpu_cache(device: torch.device) -> None:
    """
    Clear the GPU cache.

    Args:
        device (torch.device): The device to clear the cache for.
    """
    # Clear the GPU cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_device() -> torch.device:
    """
    Detects the available GPU device.

    Returns:
        torch.device: The device to be used for inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_memory_usage() -> str:
    """
    Get the current memory usage of the process.
    
    Returns:
        str: Memory usage in MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"{memory_info.rss / (1024 * 1024):.2f} MB"


def get_device_memory_usage(free: bool = False, unit: str = 'MB') -> float:
    """
    Get the current memory usage or free memory for the active PyTorch device (CUDA, MPS, or CPU).
    
    Args:
        free (bool): If True, return free memory. If False, return used memory. Defaults to False.
        unit (str): Unit for memory measurement. Either 'MB' or 'GB'. Defaults to 'MB'.
    
    Returns:
        float: Memory usage or free memory in specified units.
    """
    if unit not in ['MB', 'GB']:
        raise ValueError("Unit must be either 'MB' or 'GB'")
    
    divisor = 1024 * 1024 if unit == 'MB' else 1024 * 1024 * 1024

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if free:
            memory_free = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
            return memory_free / divisor
        else:
            return torch.cuda.memory_allocated(device) / divisor
    else:
        # For both CPU and MPS (Apple Silicon), we use system memory
        system_memory = psutil.virtual_memory()
        if free:
            return system_memory.available / divisor
        else:
            return (system_memory.total - system_memory.available) / divisor


def get_system_info() -> str:
    """
    Get detailed system information including CPU, memory, disk, and GPU usage.
    
    Returns:
        str: Formatted string containing system information.
    """
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    gpus = GPUtil.getGPUs()
    cpu_model = platform.processor()

    info = f"System Information:\n"
    info += f"OS: {platform.system()} {platform.release()}\n"
    info += f"CPU Model: {cpu_model}\n"
    info += f"CPU Usage: {cpu_percent}%\n"
    info += f"Memory: {memory.percent}% used ({memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB)\n"
    info += f"Disk: {disk.percent}% used ({disk.used / (1024**3):.2f}GB / {disk.total / (1024**3):.2f}GB)\n"
    
    if gpus:
        for i, gpu in enumerate(gpus):
            info += f"GPU {i}: {gpu.name}\n"
            info += f"  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB\n"
            info += f"  GPU utilization: {gpu.load*100}%\n"
    else:
        info += "No GPUs detected\n"
    
    return info