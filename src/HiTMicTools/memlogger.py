import logging
from HiTMicTools.utils import (
    get_memory_usage,
)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class MemoryLogger(logging.Logger):
    """Enhanced logger that includes system memory and GPU usage information.
    
    Extends the standard logging.Logger to provide memory tracking capabilities.
    Can report both system RAM usage and GPU memory when available.
    """
    _gputil_imported = False
    _gputil_available = False
    def info(
        self, 
        msg: str, 
        show_memory: bool = False, 
        cuda: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Log info message with optional memory usage details.
        
        Args:
            msg: The message to log
            show_memory: Include system memory usage if True
            cuda: Include GPU memory usage if True
            *args: Additional positional arguments for Logger
            **kwargs: Additional keyword arguments for Logger
        """
        message = msg
        if show_memory:
            message += f" | Memory: {get_memory_usage()}"
        if cuda and GPUTIL_AVAILABLE:
            gpu = GPUtil.getGPUs()[0]
            message += f" | GPU: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB"
        elif cuda:
            message += " | GPU: GPUtil not installed"
        super().info(message, *args, **kwargs)
