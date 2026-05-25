"""Tests for the cellasic crop module.

Two layers:

1. Unit tests on the new Session-8 primitives -- Gaussian-sigma Sobel,
   ID-position extrapolation, the bottom-wall picker variants, and the
   NoIDBlockDetected hard-fail. Run on synthetic data so they don't need
   the e015 dataset.

2. Ground-truth lock-in: parametrized over the 28 empty-reference .nd2
   files (local) and 28 experimental .nd2 files (RINFsci). Asserts the
   runtime pipeline reproduces the `*_crop_info.json` records that
   Sergio visually approved at the end of Session 8. If anything in the
   detect / rotate / wall-pick / calibration-apply chain regresses, the
   matching test fails for the affected file(s).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from HiTMicTools.img_processing.cellasic import (
    crop_to_target_line,
    load_default_template,
)
from HiTMicTools.img_processing.cellasic.crop import (
    NoIDBlockDetected,
    _apply_sobel,
    _extrapolate_id_positions,
    detect_walls_in_rotated,
)


# =========================================================================
#  Paths
# =========================================================================

E015_ROOT = Path(r"C:/Users/sergi/ExperimentsWindows/e015_CellAsic_Greta")

CROPPED_LINE5 = E015_ROOT / "data" / "cropped_line5"
EMPTY_REF_DIR = CROPPED_LINE5 / "empty_reference"
EMPTY_REF_JSON_DIR = EMPTY_REF_DIR / "diagnostics"
EXPERIMENTAL_ND2_DIR = Path(r"Y:/Sergio/projects/e015_CellAsic_Greta/Data/Points_line5")
EXPERIMENTAL_JSON_DIR = CROPPED_LINE5 / "experimental"
CALIBRATION_JSON = CROPPED_LINE5 / "calibration.json"

CROP_SCRIPT_PATH = E015_ROOT / "scripts" / "crop_nd2_with_diagnostic.py"


# =========================================================================
#  Layer 1 -- unit tests on the new primitives (synthetic data)
# =========================================================================

def test_apply_sobel_gaussian_smoothing_reduces_gradient_response():
    """sigma > 0 must suppress per-pixel noise BEFORE Sobel. On a
    uniform-noise field the output should have a smaller dynamic range
    (max value) and a lower mean magnitude than sigma=0.
    """
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    no_smooth = _apply_sobel(noise, gaussian_sigma=0.0)
    smoothed = _apply_sobel(noise, gaussian_sigma=2.0)
    # Both are normalized to 0-255 internally so peak is similar; what
    # matters is the bulk distribution: mean magnitude drops.
    assert smoothed.mean() < no_smooth.mean(), (
        f"sigma=2 mean ({smoothed.mean():.2f}) not lower than "
        f"sigma=0 mean ({no_smooth.mean():.2f})"
    )


def test_extrapolate_id_positions_returns_originals_plus_extrapolated():
    """Given two strict detections, extrapolation should add 4 positions
    on each side at the detected spacing, aligned to the median Y of the
    detected centres. The 2 originals retain their own Y values."""
    detections = [(1400, 1000, 0.9), (1410, 1500, 0.9)]  # spacing 500, y=1400/1410
    positions = _extrapolate_id_positions(
        detections, image_shape=(2700, 2720), extrap_limit=4
    )
    # 2 original + 4 left + 4 right = 10
    assert len(positions) == 10
    xs = sorted(p[0] for p in positions)
    # Leftmost extrapolation: 1000 - 4*500 = -1000
    assert xs[0] == -1000
    # Rightmost extrapolation: 1500 + 4*500 = 3500
    assert xs[-1] == 3500
    # The 8 extrapolated positions all share the median Y (1405); the 2
    # originals keep their own y (1400, 1410).
    cy_ref = 1405
    n_at_cy_ref = sum(1 for (_, y) in positions if y == cy_ref)
    assert n_at_cy_ref == 8, (
        f"expected 8 extrapolated positions at y={cy_ref}, got {n_at_cy_ref}; "
        f"ys={[y for (_, y) in positions]}"
    )


def test_extrapolate_id_positions_with_single_detection_returns_only_original():
    """With one detection the spacing is unknown, so extrapolation must
    not invent positions -- returns the single detected centre only."""
    detections = [(1400, 2000, 0.9)]
    positions = _extrapolate_id_positions(detections, image_shape=(2700, 2720))
    assert positions == [(2000, 1400)]


def test_bottom_picker_first_vs_strongest_diverges_on_two_dips():
    """Synthetic image with TWO dark bands BELOW the centre, both inside
    the search window (cy + [WALL_MIN_GAP_BELOW, WALL_SEARCH_BELOW]):
       - shallow band closer to centre  (y=2250, depth 80)
       - deep band further from centre  (y=2500, depth 140)
    'first' should pick the closer (shallower) one, 'strongest' the
    farther (deeper) one. Top wall is identical in both runs.
    """
    H, W = 3000, 200
    img = np.full((H, W), 200, dtype=np.uint8)
    # Top wall (single dip well above centre)
    img[600:610, :] = 60
    # Bottom dips: shallow at 2250, deep at 2500. centre_y=1500 ->
    # search window is [2200, 2600], both inside.
    img[2245:2255, :] = 120  # shallow, depth ~80
    img[2495:2505, :] = 60   # deep, depth ~140

    _, bot_first, _ = detect_walls_in_rotated(img, centre_y=1500.0, bottom_picker="first")
    _, bot_strong, _ = detect_walls_in_rotated(img, centre_y=1500.0, bottom_picker="strongest")
    assert bot_first is not None and bot_strong is not None, (
        f"both pickers should find a wall (got first={bot_first}, strong={bot_strong})"
    )
    assert bot_first < bot_strong, (
        f"'first' should pick the closer dip (y~2250), 'strongest' the deeper one (y~2500); "
        f"got first={bot_first}, strongest={bot_strong}"
    )


def test_no_id_block_raises():
    """Blank image -- no ID block to find. Must raise NoIDBlockDetected
    (hard fail, NOT a silent full-FOV fallback)."""
    blank = np.zeros((2, 2, 500, 500), dtype=np.uint8)
    template = load_default_template(gaussian_sigma=2.0)
    with pytest.raises(NoIDBlockDetected):
        crop_to_target_line(blank, template)


# =========================================================================
#  Layer 2 -- calibration ground-truth lock-in
# =========================================================================

def _load_nd2(path: Path) -> np.ndarray:
    """Read an .nd2 into a (T, C, Y, X) array. Avoids importing the e015
    script just for this helper."""
    import nd2
    with nd2.ND2File(str(path)) as r:
        arr = r.asarray()
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        arr = arr[None, :, :, :]
    return arr


@pytest.fixture(scope="session")
def crop_script():
    """Load `scripts/crop_nd2_with_diagnostic.py` as a module so the tests
    can exercise the production orchestration (tilt clamp + calibrated
    Y + min/max X). Skips the layer-2 tests if the file is not present.
    """
    if not CROP_SCRIPT_PATH.exists():
        pytest.skip(f"crop_nd2_with_diagnostic.py not found at {CROP_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "crop_script_under_test", CROP_SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def calibration():
    if not CALIBRATION_JSON.exists():
        pytest.skip(f"calibration.json not found at {CALIBRATION_JSON}")
    return json.loads(CALIBRATION_JSON.read_text())


def _ground_truth_pairs(nd2_dir: Path, json_dir: Path):
    """Return list of (nd2_path, recorded_record) for files where both exist."""
    if not nd2_dir.exists() or not json_dir.exists():
        return []
    pairs = []
    for nd2_path in sorted(nd2_dir.glob("*.nd2")):
        json_path = json_dir / f"{nd2_path.stem}_crop_info.json"
        if json_path.exists():
            pairs.append((nd2_path, json.loads(json_path.read_text())))
    return pairs


EMPTY_PAIRS = _ground_truth_pairs(EMPTY_REF_DIR, EMPTY_REF_JSON_DIR)
EXPERIMENTAL_PAIRS = _ground_truth_pairs(EXPERIMENTAL_ND2_DIR, EXPERIMENTAL_JSON_DIR)


def _assert_record_matches(record: dict, expected: dict, atol_angle: float = 0.01) -> None:
    """Compare key fields between a fresh process_detect record and the
    recorded JSON. Numeric crop bounds must match exactly (deterministic
    NCC + integer rounding); angle has a small tolerance for FP noise."""
    keys_exact = ("n_detections", "crop_y0", "crop_y1", "crop_x0", "crop_x1", "crop_mode")
    for k in keys_exact:
        assert record[k] == expected[k], (
            f"{k}: runtime={record[k]} expected={expected[k]}"
        )
    assert abs(record["angle_deg"] - expected["angle_deg"]) < atol_angle, (
        f"angle_deg: runtime={record['angle_deg']:.4f} expected={expected['angle_deg']:.4f}"
    )


@pytest.mark.skipif(
    not EMPTY_PAIRS,
    reason="empty-reference .nd2 sources or recorded JSONs not present",
)
@pytest.mark.parametrize(
    "nd2_path,expected", EMPTY_PAIRS, ids=[p[0].name for p in EMPTY_PAIRS],
)
def test_empty_ref_walls_y_lock(nd2_path: Path, expected: dict):
    """Lock the per-frame walls-Y + tilt + ID-detection on all 28 empty
    references. These measurements ARE what built the calibration JSON,
    so any drift in detect / rotate / wall-pick code will surface here.

    X is left full-width (the recorded JSONs' X came from a stale x-only
    calibration that has since been deleted -- not re-checked here).
    """
    arr = _load_nd2(nd2_path)
    template = load_default_template(gaussian_sigma=2.0)
    _, info = crop_to_target_line(
        arr, template,
        gaussian_sigma=2.0,
        threshold=0.5,
        bottom_picker="strongest",
        crop_x_walls=False,
    )
    assert info["n_detections"] == expected["n_detections"], (
        f"n_detections: runtime={info['n_detections']} expected={expected['n_detections']}"
    )
    assert abs(info["angle_deg"] - expected["angle_deg"]) < 0.01, (
        f"angle_deg: runtime={info['angle_deg']:.4f} expected={expected['angle_deg']:.4f}"
    )
    assert info["crop_y0"] == expected["crop_y0"], (
        f"crop_y0: runtime={info['crop_y0']} expected={expected['crop_y0']}"
    )
    assert info["crop_y1"] == expected["crop_y1"], (
        f"crop_y1: runtime={info['crop_y1']} expected={expected['crop_y1']}"
    )
    assert info["crop_mode"] == "walls", (
        f"crop_mode should be 'walls', got '{info['crop_mode']}'"
    )


@pytest.mark.skipif(
    not EXPERIMENTAL_PAIRS,
    reason="experimental .nd2 sources or recorded JSONs not present",
)
@pytest.mark.parametrize(
    "nd2_path,expected", EXPERIMENTAL_PAIRS,
    ids=[p[0].name for p in EXPERIMENTAL_PAIRS],
)
def test_experimental_calibration_lock(
    nd2_path: Path, expected: dict, crop_script, calibration, tmp_path,
):
    """Lock the full production pipeline (tilt clamp + calibrated Y +
    min/max X with n_det=1 fallback) on all 28 experimental .nd2 files.
    The recorded JSONs encode Sergio's visually approved crops from the
    end of Session 8; any pipeline regression flips the matching test."""
    template = crop_script.load_default_template(gaussian_sigma=2.0)
    record = crop_script.process_detect(
        nd2_path, tmp_path, template,
        gaussian_sigma=2.0,
        threshold=0.5,
        bottom_picker="first",
        crop_x_walls=False,
        calibration=calibration,
    )
    _assert_record_matches(record, expected)
