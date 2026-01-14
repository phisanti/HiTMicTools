import os
import platform
import re
import subprocess
import sys
import time
from typing import Dict, List, Literal, Optional, Tuple, Union

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


def run_command(cmd: str, timeout: int = 5) -> Tuple[int, str, str]:
    """
    Run a shell command and return the result.

    Args:
        cmd: Command to execute
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_nvidia_smi() -> Dict[str, any]:
    """
    Check nvidia-smi availability and GPU information.

    Returns:
        dict: Dictionary containing nvidia-smi status and GPU information
    """
    result = {
        "available": False,
        "driver_version": None,
        "cuda_version": None,
        "gpus": [],
        "error": None
    }

    ret_code, stdout, stderr = run_command("nvidia-smi")
    if ret_code != 0:
        result["error"] = f"nvidia-smi failed: {stderr if stderr else 'command not found'}"
        return result

    result["available"] = True

    # Extract driver version
    driver_match = re.search(r'Driver Version: (\S+)', stdout)
    if driver_match:
        result["driver_version"] = driver_match.group(1)

    # Extract CUDA version
    cuda_match = re.search(r'CUDA Version: (\S+)', stdout)
    if cuda_match:
        result["cuda_version"] = cuda_match.group(1)

    # Get detailed GPU information
    ret_code, gpu_info, _ = run_command(
        "nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,utilization.gpu "
        "--format=csv,noheader,nounits"
    )

    if ret_code == 0 and gpu_info:
        for line in gpu_info.split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    result["gpus"].append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": float(parts[2]),
                        "memory_free_mb": float(parts[3]),
                        "temperature_c": parts[4] if parts[4] != "N/A" else None,
                        "utilization_percent": parts[5] if parts[5] != "N/A" else None
                    })

    return result


def check_environment_variables() -> Dict[str, Optional[str]]:
    """
    Check important CUDA-related environment variables.

    Returns:
        dict: Dictionary of environment variable names and values
    """
    env_vars = {
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "CUDA_PATH": os.environ.get("CUDA_PATH"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "PATH": os.environ.get("PATH"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE"),
    }
    return env_vars


def check_pytorch_cuda() -> Dict[str, any]:
    """
    Check PyTorch CUDA availability and configuration.

    Returns:
        dict: Dictionary containing PyTorch CUDA status and details
    """
    result = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_count": 0,
        "gpus": [],
        "tensor_test": {"success": False, "error": None},
        "error": None
    }

    if result["cuda_available"]:
        result["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else None
        result["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        result["gpu_count"] = torch.cuda.device_count()

        for i in range(result["gpu_count"]):
            try:
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)

                result["gpus"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": props.total_memory / (1024**3),
                    "free_memory_gb": free_mem / (1024**3),
                    "used_memory_gb": (total_mem - free_mem) / (1024**3),
                    "multiprocessor_count": props.multi_processor_count
                })
            except Exception as e:
                result["error"] = f"Error accessing GPU {i}: {str(e)}"

        # Test tensor creation
        try:
            test_device = torch.device("cuda:0")
            x = torch.randn(100, 100, device=test_device)
            y = torch.randn(100, 100, device=test_device)
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            del x, y, z
            torch.cuda.empty_cache()
            result["tensor_test"]["success"] = True
        except Exception as e:
            result["tensor_test"]["error"] = str(e)
    else:
        result["error"] = "CUDA is not available to PyTorch"

    return result


def check_loaded_modules() -> List[str]:
    """
    Check loaded environment modules (for HPC systems using module system).

    Returns:
        List of loaded module names
    """
    modules = []

    # Check if module command is available
    ret_code, stdout, _ = run_command("module list 2>&1")
    if ret_code == 0 and stdout:
        # Parse module list output
        for line in stdout.split('\n'):
            # Skip header lines
            if 'Currently Loaded' in line or '---' in line or not line.strip():
                continue
            # Extract module names
            parts = line.strip().split()
            for part in parts:
                if '/' in part or any(keyword in part.lower() for keyword in ['cuda', 'python', 'gcc']):
                    modules.append(part)

    return modules
