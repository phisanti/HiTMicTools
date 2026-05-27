"""CellAsic Onix2 microfluidic device utilities.

The CellAsic B04A chamber has 6 horizontal trap rows ("lines") with
different ceiling heights. For ASCT experiments we image one target line
(typically line 5 for rod-shaped bacteria) but the full FOV captures the
silicone walls and structures of adjacent lines plus the line's ID block
(a dice-face fiducial encoding the line number).

This subpackage provides utilities to detect those fiducials, correct for
slight plate-placement tilt, and crop a FOV to its target line before
segmentation.
"""
from HiTMicTools.img_processing.cellasic.crop import (
    detect_id_blocks,
    compute_rotation,
    detect_walls_in_rotated,
    detect_chamber_x_walls,
    compute_crop_bounds,
    crop_to_target_line,
    crop_with_calibration,
    clamp_tilt,
    apply_y_calibration,
    apply_x_calibration_minmax,
    load_default_template,
    load_template,
    load_default_calibration,
    DEFAULT_LINE5_CALIBRATION,
)

__all__ = [
    "detect_id_blocks",
    "compute_rotation",
    "detect_walls_in_rotated",
    "detect_chamber_x_walls",
    "compute_crop_bounds",
    "crop_to_target_line",
    "crop_with_calibration",
    "clamp_tilt",
    "apply_y_calibration",
    "apply_x_calibration_minmax",
    "load_default_template",
    "load_template",
    "load_default_calibration",
    "DEFAULT_LINE5_CALIBRATION",
]
