"""
ROI (Region of Interest) analysis tools for microscopy images.

This package provides CPU and GPU implementations for analyzing
regions of interest in microscopy images, including measurement
extraction, property calculation, and mask operations.

The RoiAnalyser class intelligently selects the appropriate implementation
(CPU or GPU) based on environment availability, allowing pipelines to use
a single unified interface.
"""

# GPU version is optional (requires CuPy)
try:
    from HiTMicTools.roianalysis.roi_analyser_gpu import RoiAnalyser
    __all__ = ['RoiAnalyser']
except ImportError:
    # CuPy not available, fallback to CPU version
    from HiTMicTools.roianalysis.roi_analyser import RoiAnalyser
    __all__ = ['RoiAnalyser']
