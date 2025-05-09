import torch
import psutil
import platform
import time
import gc
from typing import Optional
from logging import Logger


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


def get_device_memory_usage(free: bool = False, unit: str = "MB") -> float:
    """
    Get the current memory usage or free memory for the active PyTorch device (CUDA, MPS, or CPU).

    Args:
        free (bool): If True, return free memory. If False, return used memory. Defaults to False.
        unit (str): Unit for memory measurement. Either 'MB' or 'GB'. Defaults to 'MB'.

    Returns:
        float: Memory usage or free memory in specified units.
    """
    if unit not in ["MB", "GB"]:
        raise ValueError("Unit must be either 'MB' or 'GB'")

    divisor = 1024 * 1024 if unit == "MB" else 1024 * 1024 * 1024

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if free:
            memory_free = torch.cuda.get_device_properties(
                device
            ).total_memory - torch.cuda.memory_allocated(device)
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
    disk = psutil.disk_usage("/")
    cpu_model = platform.processor()

    info = f"System Information:\n"
    info += f"OS: {platform.system()} {platform.release()}\n"
    info += f"CPU Model: {cpu_model}\n"
    info += f"CPU Usage: {cpu_percent}%\n"
    info += f"Memory: {memory.percent}% used ({memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB)\n"
    info += f"Disk: {disk.percent}% used ({disk.used / (1024**3):.2f}GB / {disk.total / (1024**3):.2f}GB)\n"

    if get_device().type == "cuda":
        import GPUtil

        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            info += f"GPU {i}: {gpu.name}\n"
            info += f"  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB\n"
            info += f"  GPU utilization: {gpu.load*100}%\n"
    else:
        info += "No GPUs detected\n"

    return info


def wait_for_memory(
    required_gb: float = 4,
    check_interval: int = 5,
    max_wait: int = 60,
    logger: Optional[Logger] = None,
    raise_on_timeout: bool = False
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
        free_mem = get_device_memory_usage(free=True, unit="GB")
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