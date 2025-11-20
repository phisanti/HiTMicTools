"""
ROI (Region of Interest) analysis tools for microscopy images.

This package provides CPU and GPU implementations for analyzing
regions of interest in microscopy images, including measurement
extraction, property calculation, and mask operations.

The RoiAnalyser class intelligently selects the appropriate implementation
(CPU or GPU) based on environment availability, allowing pipelines to use
a single unified interface.
"""

import warnings

# GPU version is optional (requires RAPIDS libraries)
_gpu_import_error = None
try:
    from HiTMicTools.roianalysis.roi_analyser_gpu import RoiAnalyser
    __all__ = ['RoiAnalyser']
except ImportError as e:
    _gpu_import_error = e
    # RAPIDS libraries not available, fallback to CPU version
    from HiTMicTools.roianalysis.roi_analyser import RoiAnalyser
    __all__ = ['RoiAnalyser']

    # Check if GPU is available but RAPIDS libraries are missing
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    if cuda_available:
        # Diagnose which RAPIDS libraries are missing
        missing_libs = []

        try:
            import cupy
        except ImportError:
            missing_libs.append("cupy")

        try:
            import cucim
        except ImportError:
            missing_libs.append("cucim")

        try:
            import cudf
        except ImportError:
            missing_libs.append("cudf")

        if missing_libs:
            warnings.warn(
                f"\n{'='*60}\n"
                f"GPU-Accelerated ROI Analysis Not Available\n"
                f"{'='*60}\n"
                f"CUDA GPU detected but missing RAPIDS libraries: {', '.join(missing_libs)}\n"
                f"\n"
                f"Using CPU-based RoiAnalyser (slower for large datasets).\n"
                f"\n"
                f"To enable GPU acceleration, install RAPIDS:\n"
                f"  conda install -c rapidsai -c nvidia -c conda-forge \\\n"
                f"    cupy cucim cudf\n"
                f"\n"
                f"Note: Match RAPIDS version to your CUDA version.\n"
                f"See: https://rapids.ai/start.html\n"
                f"{'='*60}",
                UserWarning,
                stacklevel=2
            )
