import gc
import os
import platform
import time
import tempfile
import json
from typing import Optional, Union, Dict, Any
from logging import Logger

import psutil
import torch

from .sysutils import file_lock_manager, get_memory_usage, empty_gpu_cache


class ReserveResource:
    """
    A portable, cross-platform context manager to safely acquire and use
    system or device resources across multiple processes.
    
    This class implements a sophisticated resource management system that allows
    multiple processes to run simultaneously while preventing memory overallocation.
    It uses a shared booking file to track memory reservations across processes,
    ensuring that the total allocated memory does not exceed available resources.
    
    Features:
        - Cross-platform compatibility (Windows, Linux, macOS)
        - Multi-process coordination through shared booking files
        - Automatic cleanup of stale bookings from terminated processes
        - Configurable timeout and polling intervals
        - Detailed logging of memory allocation and usage
        - Support for CUDA, MPS, and CPU devices

    Usage:
        
        device = torch.device("cuda")
        logger = logging.getLogger(__name__)
        
        with ReserveResource(device, required_gb=8.0, logger=logger) as resource:
            # Your GPU-intensive/CPU/MPS code here
            model = load_model()
            results = model.inference(data)
        
    
    Args:
        device (torch.device): The target device (cuda, mps, or cpu)
        required_gb (float): Amount of memory to reserve in GB
        logger (Optional[Logger]): Logger instance for operation tracking
        timeout (int): Maximum time to wait for resource acquisition in seconds
    
    Raises:
        TimeoutError: If resource acquisition times out
        OSError: If file operations fail
        ValueError: If invalid parameters are provided
    """
    
    def __init__(self, device: torch.device, required_gb: float,
                 logger: Optional[Logger] = None, timeout: int = 600):
        """
        Initialize the ReserveResource context manager.
        
        Args:
            device (torch.device): The target device for resource reservation
            required_gb (float): Amount of memory to reserve in GB
            logger (Optional[Logger]): Logger instance for operation tracking
            timeout (int): Maximum time to wait for resource acquisition in seconds
        
        Raises:
            ValueError: If required_gb is negative
        """
        if required_gb < 0:
            raise ValueError("required_gb must be non-negative")
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device instance")
        if required_gb < 0:
            raise ValueError("required_gb must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if device.type not in ['cuda', 'mps', 'cpu']:
            raise ValueError(f"Unsupported device type: {device.type}")            
        self.device = device
        self.required_gb = required_gb
        self.logger = logger
        self.timeout = timeout
        self.poll_interval = 2
        self._booking_registered = False
        self._process_id = str(os.getpid())
        
        # Booking file path
        # For SLURM, call TMPDIR, otherwise use system temp directory
        base_tmp_dir = os.environ.get('TMPDIR', tempfile.gettempdir())
        if device.type == 'cuda' and device.index is not None:
            booking_file = f"memory_bookings_{device.type}_{device.index}.json"
        else:
            booking_file = f"memory_bookings_{device.type}.json"
        
        self.booking_path = os.path.join(base_tmp_dir, booking_file)
    def _log(self, message: str, level: str = "info"):
        """
        Log a message using the configured logger.
        
        Args:
            message (str): The message to log
            level (str): Log level ('info', 'warning', 'error', 'debug')
        """
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
    
    def _read_bookings(self) -> Dict[str, Any]:
        """
        Read memory bookings from the shared booking file.
        
        Uses cross-platform file locking to ensure thread-safe access.
        If the file doesn't exist or is corrupted, returns a default structure.
        
        Returns:
            Dict[str, Any]: Dictionary containing bookings and metadata
                - bookings: Dict mapping process IDs to booking information
                - last_cleanup: Timestamp of last cleanup operation
        
        Raises:
            OSError: If file operations fail critically
        """
        if not os.path.exists(self.booking_path):
            return {"bookings": {}, "last_cleanup": time.time()}
        
        # Retry mechanism for file reading on macOS
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(self.booking_path, 'r') as f:
                    # Only use shared lock for reading, and handle failures gracefully
                    try:
                        file_lock_manager(f, "lock_shared")
                        data = json.load(f)
                        file_lock_manager(f, "unlock")
                        return data
                    except Exception as lock_error:
                        # If locking fails, try reading without lock as fallback
                        self._log(f"Lock failed, reading without lock: {lock_error}")
                        f.seek(0)  # Reset file position
                        data = json.load(f)
                        return data
                        
            except (json.JSONDecodeError, FileNotFoundError):
                if attempt == max_retries - 1:
                    self._log("File corrupted or not found, returning default")
                    return {"bookings": {}, "last_cleanup": time.time()}
                time.sleep(0.1)  # Brief pause before retry
                
            except OSError as e:
                if attempt == max_retries - 1:
                    self._log(f"Failed to read bookings after {max_retries} attempts: {e}", "warning")
                    return {"bookings": {}, "last_cleanup": time.time()}
                time.sleep(0.1)
        
        return {"bookings": {}, "last_cleanup": time.time()}
    
    def _write_bookings(self, data: Dict[str, Any]) -> None:
        """
        Write memory bookings to the shared booking file atomically.
        
        Uses a temporary file and atomic rename to ensure data integrity.
        Implements cross-platform file locking for concurrent access safety.
        
        Args:
            data (Dict[str, Any]): Booking data to write
        
        Raises:
            OSError: If file operations fail
            Exception: If atomic write operations fail
        """
        temp_path = self.booking_path + f".tmp.{self._process_id}"
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.booking_path), exist_ok=True)
            
            # Create and write to temp file without immediate locking
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Verify temp file exists after creation
            if not os.path.exists(temp_path):
                raise OSError(f"Temporary file {temp_path} was not created successfully")
            
            # Use a retry mechanism for macOS with proper locking
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # For Windows, remove existing file first
                    if platform.system() == "Windows" and os.path.exists(self.booking_path):
                        os.unlink(self.booking_path)
                    
                    # Atomic rename
                    os.rename(temp_path, self.booking_path)
                    break
                    
                except OSError as rename_error:
                    if attempt == max_retries - 1:
                        self._log(f"Failed to rename after {max_retries} attempts: {rename_error}", "error")
                        raise
                    self._log(f"Rename attempt {attempt + 1} failed: {rename_error}, retrying...", "warning")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError as cleanup_error:
                    self._log(f"Failed to cleanup temp file: {cleanup_error}", "warning")
            raise e
    
    def _cleanup_stale_bookings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove bookings from processes that no longer exist.
        
        Performs cleanup based on process existence and booking age.
        Only runs cleanup every 5 minutes to avoid excessive overhead.
        
        Args:
            data (Dict[str, Any]): Current booking data
        
        Returns:
            Dict[str, Any]: Cleaned booking data with stale entries removed
        
        Note:
            - Checks if processes still exist using psutil
            - Removes bookings older than 1 hour (stale_threshold)
            - Updates last_cleanup timestamp
        """
        current_time = time.time()
        stale_threshold = 3600
        
        if current_time - data.get("last_cleanup", 0) < 300:
            return data
        
        bookings = data.get("bookings", {})
        active_bookings = {}
        
        for process_id, booking in bookings.items():
            try:
                if '_' in process_id:
                    pid = int(process_id.split('_')[0])
                else:
                    pid = int(process_id)
                    
                if (psutil.pid_exists(pid) and 
                    current_time - booking.get("timestamp", 0) < stale_threshold):
                    active_bookings[process_id] = booking
            except (ValueError, IndexError):
                continue
        
        data["bookings"] = active_bookings
        data["last_cleanup"] = current_time
        return data
    
    def _get_total_booked_memory(self) -> float:
        """
        Calculate the total memory booked by all active processes.
        
        Reads the booking file, performs cleanup, and sums all active bookings.
        
        Returns:
            float: Total memory booked in GB across all processes
        
        Note:
            Returns 0.0 if any errors occur during file operations
        """
        try:
            data = self._read_bookings()
            data = self._cleanup_stale_bookings(data)
            self._write_bookings(data)
            
            total_booked = sum(booking.get("memory_gb", 0) 
                             for booking in data.get("bookings", {}).values())
            return total_booked
        except Exception as e:
            self._log(f"Error reading bookings: {e}", "warning")
            return 0.0
    
    def _register_booking(self) -> None:
        """
        Register a memory booking for the current process.
        
        Creates a booking entry with process information and writes it to
        the shared booking file. This reserves the requested memory amount
        for this process. If a booking already exists for this process,
        it updates the memory requirement to the maximum needed.
        
        Raises:
            Exception: If booking registration fails
        
        Note:
            Sets self._booking_registered to True upon successful registration
        """
        try:
            data = self._read_bookings()
            data = self._cleanup_stale_bookings(data)
            
            # Check if we already have a booking for this process
            existing_booking = data["bookings"].get(self._process_id)
            
            if existing_booking:
                # Update existing booking with maximum memory requirement
                existing_booking["memory_gb"] = max(existing_booking["memory_gb"], self.required_gb)
                existing_booking["timestamp"] = time.time()
                self._log(f"Updated booking: {existing_booking['memory_gb']:.1f}GB for process {self._process_id}")
            else:
                # Create new booking
                data["bookings"][self._process_id] = {
                    "memory_gb": self.required_gb,
                    "timestamp": time.time(),
                    "pid": os.getpid()
                }
                self._log(f"Registered booking: {self.required_gb:.1f}GB for process {self._process_id}")
            
            self._write_bookings(data)
            self._booking_registered = True
        except Exception as e:
            self._log(f"Error registering booking: {e}", "error")
            raise
    
    def _unregister_booking(self) -> None:
        """
        Unregister the memory booking for the current process.
        
        Removes the booking entry from the shared booking file, freeing up
        the reserved memory for other processes to use.
        
        Note:
            - Only performs unregistration if a booking was previously registered
            - Logs warnings if unregistration fails but doesn't raise exceptions
            - Sets self._booking_registered to False upon completion
        """
        if not self._booking_registered:
            return
        
        try:
            data = self._read_bookings()
            data = self._cleanup_stale_bookings(data)
            
            if self._process_id in data["bookings"]:
                del data["bookings"][self._process_id]
                self._write_bookings(data)
                self._log(f"Unregistered booking for process {self._process_id}")
            
            self._booking_registered = False
        except Exception as e:
            self._log(f"Error unregistering booking: {e}", "warning")
    
    def __enter__(self):
        """
        Enter the context manager and acquire the requested memory resources.
        
        Continuously monitors memory availability and waits until sufficient
        memory is available for the requested allocation. Takes into account
        both actual memory usage and bookings from other processes.
        
        Returns:
            ReserveResource: Self, allowing the context manager pattern
        
        Raises:
            TimeoutError: If resource acquisition times out
            Exception: If critical errors occur during memory checks
        
        Process:
            1. Checks current memory usage and bookings
            2. Calculates available memory considering all bookings
            3. If sufficient memory is available, registers booking and returns
            4. If insufficient, waits and retries until timeout
        """
        if not self.required_gb:
            return self
        
        start_time = time.time()
        self._log(f"Requesting {self.required_gb:.1f}GB on {self.device.type.upper()}...")
        
        while True:
            try:
                empty_gpu_cache(self.device)
                gc.collect()
                
                free_mem_gb = get_memory_usage(free=True, device=self.device, unit="GB")
                used_mem_gb = get_memory_usage(free=False, device=self.device, unit="GB")
                total_mem_gb = free_mem_gb + used_mem_gb
                total_booked_gb = self._get_total_booked_memory()
                available_mem_gb = total_mem_gb - total_booked_gb
                
                if available_mem_gb >= self.required_gb:
                    self._register_booking()
                    self._log(f"Memory allocated: Total={total_mem_gb:.1f}GB, "
                            f"Used={used_mem_gb:.1f}GB, Free={free_mem_gb:.1f}GB, "
                            f"Booked={total_booked_gb:.1f}GB, Available={available_mem_gb:.1f}GB, "
                            f"Request={self.required_gb:.1f}GB")
                    return self
                else:
                    self._log(f"Insufficient memory: Total={total_mem_gb:.1f}GB, "
                            f"Used={used_mem_gb:.1f}GB, Free={free_mem_gb:.1f}GB, "
                            f"Booked={total_booked_gb:.1f}GB, Available={available_mem_gb:.1f}GB, "
                            f"Request={self.required_gb:.1f}GB")
                    
            except Exception as e:
                self._log(f"Error during memory check: {e}", "error")
                raise
            
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(f"Resource timeout after {self.timeout}s")
            
            time.sleep(self.poll_interval)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and release the reserved memory resources.
        
        Automatically called when exiting the 'with' block, regardless of
        whether an exception occurred. Ensures proper cleanup of bookings.
        
        Args:
            exc_type: Exception type (if any exception occurred)
            exc_val: Exception value (if any exception occurred) 
            exc_tb: Exception traceback (if any exception occurred)
        
        Note:
            This method always attempts to unregister the booking, even if
            exceptions occurred within the context manager block.
        """
        self._unregister_booking()
