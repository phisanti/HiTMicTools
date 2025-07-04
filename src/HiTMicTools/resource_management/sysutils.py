import os
import platform
import sys
import time
from typing import Literal, Optional, Union

# Platform-specific imports
if platform.system() == "Windows":
    import msvcrt
else:
    import fcntl

# Third-party imports
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

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if device.type == "cuda":
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        memory_value = free_mem / divisor if free else (total_mem - free_mem) / divisor
    elif device.type == "mps":
        system_memory = psutil.virtual_memory()
        memory_value = system_memory.available / divisor if free else system_memory.used / divisor
    else:
        system_memory = psutil.virtual_memory()
        memory_value = system_memory.available / divisor if free else system_memory.used / divisor

    return f"{memory_value:.2f} {unit}" if as_string else memory_value

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
            total_mem = props.total_memory
            free_mem, _ = torch.cuda.mem_get_info(device)
            used_mem = total_mem - free_mem

            info += f"GPU {i}: {props.name}\n"
            info += f"  Memory: {used_mem / (1024**3):.2f}GB used / {total_mem / (1024**3):.2f}GB total ({free_mem / (1024**3):.2f}GB free)\n"
            info += f"  GPU utilization: Not available via PyTorch\n"
    else:
        info += "No GPUs detected\n"

    return info


def file_lock_manager(
    file_handle, 
    operation: Literal["lock_exclusive", "lock_shared", "unlock"],
    max_retries: int = 5,
    retry_delay: float = 0.1
) -> bool:
    """
    Cross-platform file locking utility.
    
    Args:
        file_handle: Open file object
        operation: Type of operation to perform
            - "lock_exclusive": Exclusive lock (write lock)
            - "lock_shared": Shared lock (read lock) 
            - "unlock": Release lock
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
    
    Returns:
        bool: True if operation successful, False otherwise
    
    Raises:
        OSError: If locking fails after all retries
    """
    
    def _windows_lock(file_handle, operation: str, attempt: int = 0) -> bool:
        """Windows-specific file locking using msvcrt."""
        try:
            if operation == "lock_exclusive":
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
            elif operation == "lock_shared":
                # Windows doesn't have shared locks in msvcrt, use exclusive
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
            elif operation == "unlock":
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            return True
        except IOError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                return _windows_lock(file_handle, operation, attempt + 1)
            raise OSError(f"Windows file lock operation '{operation}' failed: {e}")
    
    def _unix_lock(file_handle, operation: str, attempt: int = 0) -> bool:
        """Unix-specific file locking using fcntl."""
        try:
            if operation == "lock_exclusive":
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
            elif operation == "lock_shared":
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_SH)
            elif operation == "unlock":
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
            return True
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                return _unix_lock(file_handle, operation, attempt + 1)
            raise OSError(f"Unix file lock operation '{operation}' failed: {e}")
    
    # Route to appropriate platform-specific implementation
    if platform.system() == "Windows":
        return _windows_lock(file_handle, operation)
    else:
        return _unix_lock(file_handle, operation)



def is_file_locked(file_path: str) -> bool:
    """
    Check if a file is currently locked by attempting to acquire an exclusive lock.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if file is locked, False if available
    """
    try:
        with open(file_path, 'r') as f:
            file_lock_manager(f, "lock_exclusive")
            file_lock_manager(f, "unlock")
            return False
    except (OSError, IOError):
        return True
    except FileNotFoundError:
        return False
