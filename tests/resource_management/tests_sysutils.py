import unittest
import torch
import psutil
import platform
import sys
import os
import time
from unittest.mock import patch, MagicMock, call
from logging import Logger

# Assuming sysutils.py is in src/HiTMicTools/resource_management/
# This import path should work if tests are run from the project root (e.g., HiTMicTools/)
# or if src/ is in PYTHONPATH.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from HiTMicTools.resource_management.sysutils import (
    empty_gpu_cache,
    get_device,
    get_memory_usage,
    get_system_info,
    wait_for_memory,
)
import HiTMicTools.resource_management.sysutils as sysutils_module  # For checking real attributes


class TestSysUtils(unittest.TestCase):
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ["cuda", "mps", "cpu"])

    @patch(f"{sysutils_module.__name__}.get_device")
    @patch(
        f"{sysutils_module.__name__}.torch.mps.empty_cache", create=True
    )  # create=True allows mocking even if attr doesn't exist
    @patch(f"{sysutils_module.__name__}.torch.cuda.empty_cache")
    def test_empty_gpu_cache(
        self, mock_cuda_empty_cache, mock_mps_empty_cache, mock_get_device_for_empty
    ):
        """Test GPU cache clearing for different devices."""
        # Test with CUDA device
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            empty_gpu_cache(cuda_device)
            mock_cuda_empty_cache.assert_called_once()
            mock_mps_empty_cache.assert_not_called()
            mock_cuda_empty_cache.reset_mock()

            # Test with device=None, get_device returns CUDA
            mock_get_device_for_empty.return_value = cuda_device
            empty_gpu_cache(None)
            mock_cuda_empty_cache.assert_called_once()
            mock_mps_empty_cache.assert_not_called()
            mock_cuda_empty_cache.reset_mock()
            mock_get_device_for_empty.reset_mock()

        # Test with MPS device
        # The SUT checks `hasattr(torch.mps, "empty_cache")`.
        # Our mock `mock_mps_empty_cache` will be called if the real `torch.mps.empty_cache` exists and is called by SUT.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            empty_gpu_cache(mps_device)
            if hasattr(
                sysutils_module.torch.mps, "empty_cache"
            ):  # Check if the real method exists
                mock_mps_empty_cache.assert_called_once()
            else:
                mock_mps_empty_cache.assert_not_called()
            mock_cuda_empty_cache.assert_not_called()
            mock_mps_empty_cache.reset_mock()

            # Test with device=None, get_device returns MPS
            mock_get_device_for_empty.return_value = mps_device
            empty_gpu_cache(None)
            if hasattr(sysutils_module.torch.mps, "empty_cache"):
                mock_mps_empty_cache.assert_called_once()
            else:
                mock_mps_empty_cache.assert_not_called()
            mock_cuda_empty_cache.assert_not_called()
            mock_mps_empty_cache.reset_mock()
            mock_get_device_for_empty.reset_mock()

        # Test with CPU device
        cpu_device = torch.device("cpu")
        empty_gpu_cache(cpu_device)
        mock_cuda_empty_cache.assert_not_called()
        mock_mps_empty_cache.assert_not_called()

        # Test with device=None, get_device returns CPU
        mock_get_device_for_empty.return_value = cpu_device
        empty_gpu_cache(None)
        mock_cuda_empty_cache.assert_not_called()
        mock_mps_empty_cache.assert_not_called()
        mock_get_device_for_empty.reset_mock()

    def test_get_memory_usage(self):
        """Test memory usage reporting for various devices and formats."""
        # Test with default device (CPU if no GPU, or first available GPU)
        mem_float = get_memory_usage(unit="MB", as_string=False)
        self.assertIsInstance(mem_float, float)
        mem_str = get_memory_usage(unit="GB", as_string=True)
        self.assertIsInstance(mem_str, str)
        self.assertTrue(mem_str.endswith(" GB"))

        # Test with explicit CPU device
        cpu_device = torch.device("cpu")
        mem_cpu_used = get_memory_usage(
            device=cpu_device, free=False, unit="MB", as_string=False
        )
        self.assertIsInstance(mem_cpu_used, float)
        self.assertGreaterEqual(mem_cpu_used, 0)
        mem_cpu_free_str = get_memory_usage(
            device=cpu_device, free=True, unit="GB", as_string=True
        )
        self.assertIsInstance(mem_cpu_free_str, str)
        self.assertTrue(mem_cpu_free_str.endswith(" GB"))

        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            mem_cuda_used = get_memory_usage(
                device=cuda_device, free=False, unit="MB", as_string=False
            )
            self.assertIsInstance(mem_cuda_used, float)
            # Cannot guarantee >=0 if no tensor is allocated, but it should be a float.
            mem_cuda_free_str = get_memory_usage(
                device=cuda_device, free=True, unit="GB", as_string=True
            )
            self.assertIsInstance(mem_cuda_free_str, str)
            self.assertTrue(mem_cuda_free_str.endswith(" GB"))

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            mem_mps_used = get_memory_usage(
                device=mps_device, free=False, unit="MB", as_string=False
            )
            self.assertIsInstance(
                mem_mps_used, float
            )  # MPS uses system memory reporting like CPU
            self.assertGreaterEqual(mem_mps_used, 0)

        # Test 'other' device type (falls back to psutil.Process().memory_info().rss)
        mock_other_device = MagicMock(spec=torch.device)
        mock_other_device.type = "other_type"  # Not cuda, mps, or cpu

        with patch(f"{sysutils_module.__name__}.psutil.Process") as mock_ps_process:
            mock_proc_instance = MagicMock()
            mock_mem_info = MagicMock()
            mock_mem_info.rss = 2 * (1024**2)  # Simulate 2 MB RSS
            mock_proc_instance.memory_info.return_value = mock_mem_info
            mock_ps_process.return_value = mock_proc_instance

            mem_other_val = get_memory_usage(
                device=mock_other_device, unit="MB", as_string=False
            )
            self.assertAlmostEqual(mem_other_val, 2.0)
            mock_ps_process.assert_called_once()

            mem_other_str_gb = get_memory_usage(
                device=mock_other_device, unit="GB", as_string=True
            )
            self.assertAlmostEqual(
                float(mem_other_str_gb.split(" ")[0]), 2.0 / 1024, places=2
            )

        with self.assertRaisesRegex(ValueError, "Unit must be either 'MB' or 'GB'"):
            get_memory_usage(unit="KB")

    def test_get_system_info(self):
        """Test system information string generation."""
        info = get_system_info()
        self.assertIsInstance(info, str)
        self.assertIn("System Information:", info)
        self.assertIn("OS:", info)
        self.assertIn(f"Python: {platform.python_version()}", info)
        self.assertIn("CPU Model:", info)
        self.assertIn(f"CPU Cores: {os.cpu_count()}", info)
        self.assertIn("CPU Usage:", info)
        self.assertIn("Memory:", info)
        if torch.cuda.is_available():
            self.assertIn("GPU ", info)
        else:
            self.assertIn("No GPUs detected", info)

    def test_wait_for_memory(self):
        """Test memory waiting logic, including timeouts and errors."""
        mock_logger = MagicMock(spec=Logger)
        test_device = get_device()  # Use a real device for total memory calculation

        # 1. required_gb = 0 (should return True immediately)
        self.assertTrue(wait_for_memory(required_gb=0, logger=mock_logger))
        mock_logger.info.assert_not_called()  # No logging if no wait

        # 2. Sufficient memory (very small requirement)
        self.assertTrue(
            wait_for_memory(
                required_gb=0.0001,
                device=test_device,
                check_interval=0,
                max_wait=1,
                logger=mock_logger,
            )
        )
        mock_logger.info.assert_any_call(
            unittest.mock.ANY
        )  # "Sufficient memory available..."
        mock_logger.reset_mock()

        # 3. Requested memory exceeds total
        if test_device.type == "cuda":
            total_gb = torch.cuda.get_device_properties(test_device).total_memory / (
                1024**3
            )
        else:  # cpu or mps
            total_gb = psutil.virtual_memory().total / (1024**3)

        exceeds_total_gb = total_gb + 1.0

        self.assertFalse(
            wait_for_memory(
                required_gb=exceeds_total_gb,
                device=test_device,
                logger=mock_logger,
                raise_on_timeout=False,
            )
        )
        expected_msg_exceeds = (
            f"Requested memory ({exceeds_total_gb:.2f} GB) exceeds total memory on {test_device.type} "
            f"({total_gb:.2f} GB)."
        )
        mock_logger.error.assert_called_once_with(expected_msg_exceeds)
        mock_logger.reset_mock()

        with self.assertRaisesRegex(
            MemoryError, expected_msg_exceeds.replace("(", r"\(").replace(")", r"\)")
        ):
            wait_for_memory(
                required_gb=exceeds_total_gb,
                device=test_device,
                logger=mock_logger,
                raise_on_timeout=True,
            )
        mock_logger.error.assert_called_once_with(expected_msg_exceeds)
        mock_logger.reset_mock()

        # 4. Timeout scenario
        # Mock get_memory_usage to consistently report low memory
        # Ensure required_gb is less than total_gb but more than mock free memory
        realistic_required_gb = min(
            total_gb * 0.8, 1.0
        )  # Request 80% of total or 1GB, whichever is smaller
        mock_free_mem_val = realistic_required_gb / 2

        with patch(
            f"{sysutils_module.__name__}.get_memory_usage",
            return_value=mock_free_mem_val,
        ) as mock_get_mem:
            self.assertFalse(
                wait_for_memory(
                    required_gb=realistic_required_gb,
                    device=test_device,
                    check_interval=0,
                    max_wait=1,
                    logger=mock_logger,
                    raise_on_timeout=False,
                )
            )
            self.assertTrue(mock_logger.warning.called)

            # Check the error log for timeout message
            # Example: "Memory wait timeout after 1s (free: 0.50 GB, required: 1.00 GB)"
            # We check for the prefix of the message.
            timeout_error_call = [
                c
                for c in mock_logger.error.call_args_list
                if "Memory wait timeout" in c[0][0]
            ]
            self.assertEqual(len(timeout_error_call), 1)
            self.assertIn(
                f"free: {mock_free_mem_val:.2f} GB", timeout_error_call[0][0][0]
            )
            self.assertIn(
                f"required: {realistic_required_gb} GB", timeout_error_call[0][0][0]
            )
            mock_logger.reset_mock()

            with self.assertRaisesRegex(MemoryError, "Memory wait timeout"):
                wait_for_memory(
                    required_gb=realistic_required_gb,
                    device=test_device,
                    check_interval=0,
                    max_wait=1,
                    logger=mock_logger,
                    raise_on_timeout=True,
                )
            self.assertTrue(mock_logger.warning.called)
            timeout_error_call_raise = [
                c
                for c in mock_logger.error.call_args_list
                if "Memory wait timeout" in c[0][0]
            ]
            self.assertEqual(len(timeout_error_call_raise), 1)
            mock_logger.reset_mock()

        # 5. Successful wait after a few tries
        with patch(f"{sysutils_module.__name__}.get_memory_usage") as mock_get_mem_prog:
            # Fails twice, then succeeds
            mock_get_mem_prog.side_effect = [
                mock_free_mem_val,
                mock_free_mem_val,
                realistic_required_gb + 0.1,
            ]
            self.assertTrue(
                wait_for_memory(
                    required_gb=realistic_required_gb,
                    device=test_device,
                    check_interval=0,
                    max_wait=5,
                    logger=mock_logger,
                    raise_on_timeout=False,
                )
            )
            self.assertEqual(mock_logger.warning.call_count, 2)

            success_info_call = [
                c
                for c in mock_logger.info.call_args_list
                if "Sufficient memory available" in c[0][0]
            ]
            self.assertEqual(len(success_info_call), 1)
            self.assertIn(
                f"{realistic_required_gb + 0.1:.2f} GB", success_info_call[0][0][0]
            )
            mock_logger.reset_mock()

        # 6. Test with no logger (should run without error)
        with patch(
            f"{sysutils_module.__name__}.get_memory_usage", return_value=0.0002
        ):  # Ensure it passes
            self.assertTrue(
                wait_for_memory(
                    required_gb=0.0001,
                    device=test_device,
                    check_interval=0,
                    max_wait=1,
                    logger=None,
                )
            )


if __name__ == "__main__":
    unittest.main()
