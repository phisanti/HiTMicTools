import gc
import os
import sys
import platform
import time
from typing import Optional, Union
from logging import Logger

import psutil
import torch


def empty_gpu_cache(device: Optional[torch.device] = None) -> None:
    """
    Clear the GPU cache for the specified device.

    Args:
        device (torch.device, optional): The device to clear the cache for.
            If None, uses the current active device.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def get_device() -> torch.device:
    """
    Detects the available GPU device.

    Returns:
        torch.device: The device to be used for inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_memory_usage(
    device: Optional[torch.device] = None,
    free: bool = False,
    unit: str = "MB",
    as_string: bool = False,
) -> Union[float, str]:
    """
    Get the memory usage for the specified device or process.

    Args:
        device (torch.device, optional): The device to get memory for. If None, uses current active device.
        free (bool): If True, return free memory. If False, return used memory. Defaults to False.
        unit (str): Unit for memory measurement. Either 'MB' or 'GB'. Defaults to 'MB'.
        as_string (bool): If True, returns formatted string with units. If False, returns float. Defaults to False.

    Returns:
        Union[float, str]: Memory usage in specified units, either as float or formatted string.
    """
    if unit not in ["MB", "GB"]:
        raise ValueError("Unit must be either 'MB' or 'GB'")

    divisor = 1024 * 1024 if unit == "MB" else 1024 * 1024 * 1024

    # If no device specified, get the current active device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Get memory based on device type
    if device.type == "cuda":
        if free:
            memory_value = (
                torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_allocated(device)
            ) / divisor
        else:
            memory_value = torch.cuda.memory_allocated(device) / divisor
    else:
        # For both CPU and MPS (Apple Silicon), use system memory
        if device.type == "cpu" or device.type == "mps":
            system_memory = psutil.virtual_memory()
            if free:
                memory_value = system_memory.available / divisor
            else:
                memory_value = (system_memory.total - system_memory.available) / divisor
        else:
            # For process memory (backward compatibility with old get_memory_usage)
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_value = memory_info.rss / divisor

    if as_string:
        return f"{memory_value:.2f} {unit}"
    return memory_value


def get_system_info() -> str:
    """
    Get detailed system information including CPU, memory, disk, and GPU usage (without GPUtil).

    Returns:
        str: Formatted string containing system information.
    """
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    mem_percent = memory.used / memory.total * 100

    cpu_model = platform.processor()
    info = f"System Information:\n"
    info += f"OS: {platform.system()} {platform.release()}\n"
    info += f"Python: {platform.python_version()} ({sys.executable})\n"
    info += f"CPU Model: {cpu_model}\n"
    info += f"CPU Cores: {os.cpu_count()}\n"
    info += f"CPU Usage: {cpu_percent}%\n"

    info += f"Memory: {mem_percent:.2f}% used ({memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB)\n"

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            device = torch.device(f"cuda:{i}")
            props = torch.cuda.get_device_properties(device)
            total_mem = props.total_memory / (1024**3)
            used_mem = torch.cuda.memory_allocated(device) / (1024**3)
            free_mem = (props.total_memory - torch.cuda.memory_allocated(device)) / (
                1024**3
            )
            info += f"GPU {i}: {props.name}\n"
            info += f"  Memory: {used_mem:.2f}GB used / {total_mem:.2f}GB total ({free_mem:.2f}GB free)\n"
            # Utilization is not available via PyTorch, so we note this
            info += f"  GPU utilization: Not available via PyTorch\n"
    else:
        info += "No GPUs detected\n"

    return info


def wait_for_memory(
    required_gb: float = 4,
    check_interval: int = 5,
    max_wait: int = 60,
    logger: Optional[Logger] = None,
    raise_on_timeout: bool = False,
) -> bool:
    """
    Wait until sufficient free memory is available or timeout occurs.

    Args:
        required_gb (float): Minimum free memory required in GB.
        check_interval (int): Seconds between memory checks.
        max_wait (int): Maximum seconds to wait before timeout.
        logger (Logger, optional): Logger for status messages.
        raise_on_timeout (bool): Raise MemoryError on timeout if True.

    Returns:
        bool: True if sufficient memory became available, False if timed out.

    Raises:
        MemoryError: If raise_on_timeout is True and wait times out.
    """
    start_time = time.time()
    while True:
        free_mem = get_memory_usage(free=True, unit="GB")
        if free_mem >= required_gb:
            if logger:
                logger.info(f"Sufficient memory available: {free_mem:.2f} GB")
            return True

        elapsed = time.time() - start_time
        if elapsed >= max_wait:
            msg = f"Memory wait timeout after {max_wait}s (free: {free_mem:.2f} GB, required: {required_gb} GB)"
            if logger:
                logger.error(msg)
            if raise_on_timeout:
                raise MemoryError(msg)
            return False

        if logger:
            logger.warning(
                f"Insufficient memory: {free_mem:.2f} GB available, {required_gb} GB required "
                f"(waited {elapsed:.1f}/{max_wait}s)"
            )
        gc.collect()
        time.sleep(check_interval)
