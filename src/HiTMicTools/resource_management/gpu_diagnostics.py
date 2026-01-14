"""
GPU diagnostics module for detecting and troubleshooting GPU issues.

This module provides comprehensive GPU diagnostics including:
- NVIDIA driver and nvidia-smi availability
- CUDA environment configuration
- PyTorch CUDA support
- Actual GPU compute testing
- SLURM/HPC environment detection
"""

import platform
import sys
from datetime import datetime
from typing import Optional

from HiTMicTools.resource_management.sysutils import (
    check_environment_variables,
    check_loaded_modules,
    check_nvidia_smi,
    check_pytorch_cuda
)


class GPUDiagnostics:
    """
    Comprehensive GPU diagnostics and troubleshooting tool.

    Orchestrates GPU checks from sysutils module and provides formatted
    reporting with troubleshooting recommendations.
    """

    def __init__(self):
        """Initialize GPU diagnostics."""
        self.nvidia_info = {}
        self.pytorch_info = {}
        self.env_vars = {}
        self.modules = []
        self.all_checks_passed = True
        self.issues = []

    def run_checks(self) -> None:
        """Run all diagnostic checks and store results."""
        self.nvidia_info = check_nvidia_smi()
        self.env_vars = check_environment_variables()
        self.pytorch_info = check_pytorch_cuda()
        self.modules = check_loaded_modules()

        # Analyze results
        self.all_checks_passed = True
        self.issues = []

        if not self.nvidia_info["available"]:
            self.all_checks_passed = False
            self.issues.append("nvidia-smi not available - NVIDIA driver not installed or not in PATH")

        if not self.pytorch_info["cuda_available"]:
            self.all_checks_passed = False
            self.issues.append("PyTorch CUDA not available - main issue preventing GPU usage")

        if self.pytorch_info["cuda_available"] and not self.pytorch_info["tensor_test"]["success"]:
            self.all_checks_passed = False
            self.issues.append("GPU tensor operations failed - GPU may not be responsive")

    def print_report(self, verbose: bool = False) -> None:
        """
        Print diagnostic report to stdout.

        Args:
            verbose: If True, show detailed output including full environment variables
        """
        print("=" * 80)
        print("HiTMicTools GPU Diagnostics")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # System information
        print(f"\n{'System Information:':<40}")
        print(f"  OS: {platform.system()} {platform.release()}")
        print(f"  Hostname: {platform.node()}")
        print(f"  Python: {platform.python_version()}")
        print(f"  Python Path: {sys.executable}")

        # Check for HPC/SLURM environment
        print(f"\n{'HPC Environment:':<40}")
        if self.env_vars["SLURM_JOB_ID"]:
            print(f"  SLURM Job ID: {self.env_vars['SLURM_JOB_ID']}")
            print(f"  SLURM GPUs on Node: {self.env_vars['SLURM_GPUS_ON_NODE'] or 'Not set'}")
            print(f"  SLURM Job GPUs: {self.env_vars['SLURM_JOB_GPUS'] or 'Not set'}")
        else:
            print("  Not running in SLURM environment")

        # Check loaded modules
        if self.modules:
            print(f"\n{'Loaded Modules:':<40}")
            for module in self.modules:
                print(f"  - {module}")
        else:
            print(f"\n{'Loaded Modules:':<40} None detected (module command unavailable)")

        # Environment variables
        print(f"\n{'CUDA Environment Variables:':<40}")
        print(f"  CUDA_HOME: {self.env_vars['CUDA_HOME'] if self.env_vars['CUDA_HOME'] else 'Not set'}")
        print(f"  CUDA_PATH: {self.env_vars['CUDA_PATH'] if self.env_vars['CUDA_PATH'] else 'Not set'}")
        print(f"  CUDA_VISIBLE_DEVICES: {self.env_vars['CUDA_VISIBLE_DEVICES'] if self.env_vars['CUDA_VISIBLE_DEVICES'] else 'Not set (all GPUs visible)'}")

        if verbose:
            ld_path = self.env_vars["LD_LIBRARY_PATH"]
            if ld_path:
                print(f"  LD_LIBRARY_PATH:")
                for path in ld_path.split(':'):
                    if path.strip():
                        print(f"    - {path}")
            else:
                print(f"  LD_LIBRARY_PATH: Not set")

        # nvidia-smi check
        print(f"\n{'NVIDIA Driver (nvidia-smi):':<40}")
        if self.nvidia_info["available"]:
            print(f"  Status: Available")
            print(f"  Driver Version: {self.nvidia_info['driver_version']}")
            print(f"  CUDA Version (Driver): {self.nvidia_info['cuda_version']}")
            print(f"\n  Detected GPUs ({len(self.nvidia_info['gpus'])}):")

            for gpu in self.nvidia_info["gpus"]:
                print(f"\n    GPU {gpu['index']}: {gpu['name']}")
                print(f"      Memory: {gpu['memory_free_mb']/1024:.2f} GB free / "
                      f"{gpu['memory_total_mb']/1024:.2f} GB total")
                if gpu['temperature_c']:
                    print(f"      Temperature: {gpu['temperature_c']}C")
                if gpu['utilization_percent']:
                    print(f"      Utilization: {gpu['utilization_percent']}%")
        else:
            print(f"  Status: Not available")
            print(f"  Error: {self.nvidia_info['error']}")

        # PyTorch CUDA check
        print(f"\n{'PyTorch CUDA:':<40}")
        print(f"  PyTorch Version: {self.pytorch_info['pytorch_version']}")
        print(f"  CUDA Available: {self.pytorch_info['cuda_available']}")

        if self.pytorch_info["cuda_available"]:
            print(f"  CUDA Version (PyTorch): {self.pytorch_info['cuda_version']}")
            print(f"  cuDNN Version: {self.pytorch_info['cudnn_version']}")
            print(f"  GPU Count: {self.pytorch_info['gpu_count']}")

            print(f"\n  PyTorch GPU Details:")
            for gpu in self.pytorch_info["gpus"]:
                print(f"\n    GPU {gpu['index']}: {gpu['name']}")
                print(f"      Compute Capability: {gpu['compute_capability']}")
                print(f"      Total Memory: {gpu['total_memory_gb']:.2f} GB")
                print(f"      Free Memory: {gpu['free_memory_gb']:.2f} GB")
                print(f"      Used Memory: {gpu['used_memory_gb']:.2f} GB")
                print(f"      Multiprocessors: {gpu['multiprocessor_count']}")

            # Tensor test
            print(f"\n  GPU Compute Test (tensor operations):")
            if self.pytorch_info["tensor_test"]["success"]:
                print(f"    Status: PASSED - Successfully created and computed tensors on GPU")
            else:
                print(f"    Status: FAILED")
                print(f"    Error: {self.pytorch_info['tensor_test']['error']}")
        else:
            print(f"  Error: {self.pytorch_info['error']}")

        # Summary and recommendations
        self._print_summary()

    def _print_summary(self) -> None:
        """Print diagnostic summary and troubleshooting recommendations."""
        print(f"\n{'='*80}")
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)

        if self.all_checks_passed:
            print("\nSTATUS: All checks PASSED")
            print("Your GPU setup is working correctly!")
        else:
            print("\nSTATUS: Issues detected")
            print("\nIssues found:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

            self._print_troubleshooting_steps()

        print("\n" + "=" * 80)

    def _print_troubleshooting_steps(self) -> None:
        """Print context-specific troubleshooting steps."""
        print("\nTroubleshooting Steps:")

        if not self.nvidia_info["available"]:
            print("\n  For nvidia-smi issues:")
            print("    1. Verify NVIDIA driver is installed: lspci | grep -i nvidia")
            print("    2. Check if nvidia module is loaded: lsmod | grep nvidia")
            print("    3. Try loading nvidia module: sudo modprobe nvidia")

        if not self.pytorch_info["cuda_available"]:
            print("\n  For PyTorch CUDA issues:")
            print("    1. Check CUDA module is loaded: module list")
            print("    2. Verify CUDA version compatibility with PyTorch")
            print("    3. Check for conflicting conda CUDA packages: conda list | grep cuda")
            print("    4. Ensure GPU is allocated in SLURM job (--gres=gpu:1)")
            print("    5. Check CUDA_VISIBLE_DEVICES is not hiding GPUs")

            if self.env_vars["SLURM_JOB_ID"] and not self.env_vars["SLURM_GPUS_ON_NODE"]:
                print("    ! WARNING: Running in SLURM but no GPUs allocated to job")

        if self.pytorch_info["cuda_available"] and not self.pytorch_info["tensor_test"]["success"]:
            print("\n  For GPU compute issues:")
            print("    1. GPU may be locked or unresponsive - check nvidia-smi")
            print("    2. Try clearing GPU cache: torch.cuda.empty_cache()")
            print("    3. Check for other processes using GPU: nvidia-smi pmon")

    def save_report(self, output_file: str, verbose: bool = False) -> None:
        """
        Save diagnostic report to file.

        Args:
            output_file: Path to output file
            verbose: If True, show detailed output
        """
        original_stdout = sys.stdout
        try:
            with open(output_file, 'a') as f:
                sys.stdout = f
                self.print_report(verbose=verbose)
        finally:
            sys.stdout = original_stdout

    def get_exit_code(self) -> int:
        """
        Get exit code based on diagnostic results.

        Returns:
            int: 0 if all checks passed, 1 otherwise
        """
        return 0 if self.all_checks_passed else 1


def run_gpu_diagnostics(output_file: Optional[str] = None, verbose: bool = False) -> int:
    """
    Run comprehensive GPU diagnostics.

    Args:
        output_file: Optional path to save diagnostics to file
        verbose: If True, show detailed output including full environment variables

    Returns:
        int: Exit code (0 if CUDA is available, 1 otherwise)
    """
    diagnostics = GPUDiagnostics()
    diagnostics.run_checks()

    # Print to stdout
    diagnostics.print_report(verbose=verbose)

    # Save to file if requested
    if output_file:
        diagnostics.save_report(output_file, verbose=verbose)
        print(f"Diagnostics saved to: {output_file}")

    return diagnostics.get_exit_code()
