import logging
import torch

from .sysutils import get_memory_usage

GPUTIL_AVAILABLE = torch.cuda.is_available()


class MemoryLogger(logging.Logger):
    """Enhanced logger that includes system memory and GPU usage information.

    Extends the standard logging.Logger to provide memory tracking capabilities.
    Can report both system RAM usage and GPU memory when available.
    """

    def info(
        self, msg: str, show_memory: bool = False, cuda: bool = False, *args, **kwargs
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
            try:
                cpu_device = torch.device("cpu")
                ram_used = get_memory_usage(
                    device=cpu_device, free=False, unit="GB", as_string=False
                )
                ram_free = get_memory_usage(
                    device=cpu_device, free=True, unit="GB", as_string=False
                )
                message += f" | RAM: {ram_used:.2f} GB used / {ram_free:.2f} GB free"
            except Exception as e:
                message += f" | RAM: Error ({e})"

        if cuda:
            if GPUTIL_AVAILABLE:
                try:
                    cuda_device = torch.device("cuda")
                    vram_used = get_memory_usage(
                        device=cuda_device, free=True, unit="GB", as_string=False
                    )
                    vram_free = get_memory_usage(
                        device=cuda_device, free=True, unit="GB", as_string=False
                    )
                    message += (
                        f" | VRAM: {vram_used:.2f} GB used / {vram_free:.2f} GB free"
                    )
                except Exception as e:
                    message += f" | VRAM: Error ({e})"
            else:
                message += " | VRAM: Not available"

        super().info(message, *args, **kwargs)
