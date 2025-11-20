"""Shared utilities for CPU and GPU RoiAnalyser implementations.

This module provides duck-typed helper functions that work with both NumPy and CuPy arrays,
enabling code reuse between CPU and GPU implementations while maintaining performance.
"""

import numpy as np
from typing import Union, List, Any


def get_array_module(arr: Any):
    """
    Return numpy or cupy module based on array type.

    Args:
        arr: NumPy or CuPy array

    Returns:
        numpy or cupy module

    Example:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> xp = get_array_module(arr)
        >>> # xp will be numpy
    """
    if hasattr(arr, '__cuda_array_interface__'):
        try:
            import cupy as cp
            return cp
        except ImportError:
            raise ImportError("CuPy is required for GPU arrays but is not installed")
    else:
        return np


def compute_label_offsets(frame_obj_counts, xp=None):
    """
    Compute cumulative label offsets for multi-frame labeling.

    This enables vectorized label assignment across frames while
    maintaining global label continuity. Uses prefix sum to compute
    the starting label offset for each frame.

    Args:
        frame_obj_counts: Array of object counts per frame
        xp: Array module (numpy or cupy). Auto-detected if None.

    Returns:
        Array of label offsets for each frame. The i-th element contains
        the offset to add to labels in frame i.

    Example:
        >>> import numpy as np
        >>> frame_counts = np.array([10, 15, 8])
        >>> offsets = compute_label_offsets(frame_counts)
        >>> print(offsets)  # [0, 10, 25]

        This means:
        - Frame 0: labels 1-10 (offset=0)
        - Frame 1: labels 11-25 (offset=10)
        - Frame 2: labels 26-33 (offset=25)
    """
    if xp is None:
        xp = get_array_module(frame_obj_counts)

    return xp.concatenate([
        xp.array([0]),
        xp.cumsum(frame_obj_counts[:-1])
    ])


def apply_label_offsets(labeled_frames, offsets, xp=None):
    """
    Apply label offsets to a stack of labeled frames.

    This vectorizes the offset application, avoiding sequential loops
    and enabling parallel processing of frames.

    Args:
        labeled_frames: List or array of labeled frames. Each frame should
            have labels starting from 0 (background) or 1 (first object).
        offsets: Array of label offsets per frame (from compute_label_offsets)
        xp: Array module (numpy or cupy). Auto-detected if None.

    Returns:
        Labeled mask with continuous labels across frames. Background pixels
        (label=0) remain 0.

    Example:
        >>> import numpy as np
        >>> # Three frames with local labels [1,2], [1,2,3], [1]
        >>> frame1 = np.array([[0, 1], [1, 2]])
        >>> frame2 = np.array([[1, 2], [2, 3]])
        >>> frame3 = np.array([[0, 1], [1, 0]])
        >>> labeled_frames = [frame1, frame2, frame3]
        >>> offsets = np.array([0, 2, 5])  # From compute_label_offsets([2, 3, 1])
        >>> result = apply_label_offsets(labeled_frames, offsets)
        >>> # result[0] = [[0, 1], [1, 2]]  (offset=0)
        >>> # result[1] = [[3, 4], [4, 5]]  (offset=2)
        >>> # result[2] = [[0, 6], [6, 0]]  (offset=5)
    """
    if xp is None:
        xp = get_array_module(offsets)

    # Stack frames into single array
    if isinstance(labeled_frames, list):
        stacked = xp.stack(labeled_frames, axis=0)
    else:
        stacked = labeled_frames

    # Reshape offsets for broadcasting across spatial dimensions
    # offsets shape: (n_frames,) -> (n_frames, 1, 1, ...)
    offsets_bc = offsets.reshape(-1, *([1] * (stacked.ndim - 1)))

    # Vectorized offset application: only non-zero labels get offset
    return xp.where(stacked != 0, stacked + offsets_bc, 0)


def validate_measurements_equivalence(measurements1, measurements2, rtol=1e-5, atol=1e-7):
    """
    Validate that two measurement DataFrames are numerically equivalent.

    Used for testing CPU vs GPU implementations or validating optimizations
    produce identical results.

    Args:
        measurements1: First DataFrame
        measurements2: Second DataFrame
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Raises:
        AssertionError: If DataFrames differ beyond tolerance

    Example:
        >>> cpu_measurements = cpu_analyser.get_roi_measurements()
        >>> gpu_measurements = gpu_analyser.get_roi_measurements()
        >>> validate_measurements_equivalence(cpu_measurements, gpu_measurements)
        # Raises AssertionError if measurements differ
    """
    import pandas as pd

    # Check shapes match
    assert measurements1.shape == measurements2.shape, (
        f"Shape mismatch: {measurements1.shape} vs {measurements2.shape}"
    )

    # Sort by label to ensure same order (measurements may be computed in different orders)
    measurements1_sorted = measurements1.sort_values('label').reset_index(drop=True)
    measurements2_sorted = measurements2.sort_values('label').reset_index(drop=True)

    # Check each column
    for col in measurements1_sorted.columns:
        if measurements1_sorted[col].dtype in [np.float32, np.float64]:
            # Float columns: check with tolerance
            np.testing.assert_allclose(
                measurements1_sorted[col].values,
                measurements2_sorted[col].values,
                rtol=rtol,
                atol=atol,
                err_msg=f"Mismatch in column {col}"
            )
        else:
            # Integer/string columns: exact match
            np.testing.assert_array_equal(
                measurements1_sorted[col].values,
                measurements2_sorted[col].values,
                err_msg=f"Mismatch in column {col}"
            )


def get_optimal_workers(image_shape, max_workers=None):
    """
    Determine optimal number of workers for parallel processing based on data size.

    This implements adaptive worker scaling to balance parallelization benefits
    against overhead costs. Small datasets benefit less from parallelization due
    to process spawning and IPC overhead.

    Args:
        image_shape: Shape of image array (T, S, C, H, W) or (T, H, W)
        max_workers: Maximum workers to use. If None, uses 75% of CPU count.

    Returns:
        Optimal number of workers for the given data size

    Example:
        >>> # Small dataset: 20 frames, 512x512
        >>> n_workers = get_optimal_workers((20, 1, 1, 512, 512))
        >>> # Returns 4 on 16-core system (limited for small data)

        >>> # Large dataset: 500 frames, 512x512
        >>> n_workers = get_optimal_workers((500, 1, 1, 512, 512))
        >>> # Returns 12 on 16-core system (full parallelization)
    """
    from multiprocessing import cpu_count

    if max_workers is None:
        # Use 75% of available cores to avoid oversubscription
        max_workers = max(1, int(cpu_count() * 0.75))

    n_frames = image_shape[0]

    # Calculate frame size in MB (assuming float32)
    if len(image_shape) >= 5:
        # Full shape: (T, S, C, H, W)
        frame_size_mb = (np.prod(image_shape[1:]) * 4) / (1024**2)
    elif len(image_shape) >= 3:
        # Reduced shape: (T, H, W)
        frame_size_mb = (np.prod(image_shape[1:]) * 4) / (1024**2)
    else:
        # Unknown shape, use conservative estimate
        frame_size_mb = 1.0

    # Heuristics based on profiling
    if n_frames < 20:
        # Very small datasets: overhead dominates, use few workers
        return min(4, max_workers)
    elif n_frames < 50:
        # Small datasets: moderate parallelization
        return min(8, max_workers)
    elif frame_size_mb > 10:
        # Large frames: limit workers to avoid memory pressure
        # Each worker needs a copy of the frame
        return min(8, max_workers)
    else:
        # Medium to large datasets with reasonable frame size: full parallelization
        return max_workers


def to_cupy_array(arr, copy=False):
    """
    Convert various array types to CuPy array with minimal copying.

    Supports zero-copy conversion from PyTorch CUDA tensors via DLPack,
    avoiding expensive GPU->CPU->GPU round trips.

    Args:
        arr: Input array/tensor (CuPy, PyTorch, NumPy)
        copy: Force copy even for CuPy arrays

    Returns:
        CuPy array

    Raises:
        ImportError: If CuPy is not installed

    Example:
        >>> import torch
        >>> # PyTorch tensor on GPU
        >>> torch_tensor = torch.randn(200, 512, 512, device='cuda')
        >>>
        >>> # Zero-copy conversion (no data transfer!)
        >>> cupy_array = to_cupy_array(torch_tensor)
        >>> # Both share the same GPU memory
        >>> assert torch_tensor.data_ptr() == cupy_array.data.ptr
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy is required but is not installed")

    # Already CuPy
    if isinstance(arr, cp.ndarray):
        return arr.copy() if copy else arr

    # PyTorch tensor on CUDA (zero-copy via DLPack!)
    if hasattr(arr, '__cuda_array_interface__'):
        # This covers PyTorch CUDA tensors
        # cp.asarray uses DLPack internally for zero-copy
        return cp.asarray(arr)

    # PyTorch tensor on CPU
    if hasattr(arr, 'cpu') and hasattr(arr, 'numpy'):
        arr = arr.cpu().numpy()

    # NumPy array (requires CPU→GPU copy)
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr)

    # Fallback for other array-like objects
    return cp.asarray(arr)
