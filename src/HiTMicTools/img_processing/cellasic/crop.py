"""Detect the ID-block fiducial in a CellAsic FOV, correct for tilt, and
crop to the target line strip.

The CellAsic Onix2 B04A chamber has 6 trap rows ("lines") whose ceilings
have different heights. The image of each line contains:

  - Silicone support pillars on a regular ~200 px grid (background)
  - An ID block: a square outline enclosing N small squares arranged in
    a dice-face pattern where N == line number. For line 5 this is the
    standard "5" face (4 corners + 1 centre).
  - Bacteria + debris (the actual signal).
  - At the top and bottom of the line, dark silicone walls separating it
    from the adjacent lines.

We detect the ID block(s) via Sobel-edge template matching (intensity-
invariant, works across chambers with different contrast). Multiple
detections per FOV (typical: 2-4) let us fit a line through their
centres and recover the small plate-placement tilt (~0.3-0.7 deg). After
tilt correction the silicone walls are horizontal bands of dark intensity
which we find via row-median projection.

Crop is top/bottom only (the full image width is kept). Left/right
chamber-wall cropping was tried but proved brittle in dense CellAsic
wells; the segmenter handles partial-FOV edge cases on its own.

Public API: see `crop_to_target_line(image, template, target_height=None)`.

Tunables are exposed as module constants; default values were calibrated
on the e015 minitest set (5 FOVs spanning 4 chambers).
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image


# --- detection (template matching) -------------------------------------------

DEFAULT_THRESHOLD = 0.25       # NCC score floor on the Sobel edge map
DEFAULT_MIN_SEP = 250          # px -- minimum centre-to-centre separation between matches
DEFAULT_TOP_K = 8              # safety cap on number of detections per frame
DEFAULT_MAX_Y_DRIFT = 60       # px -- same-row Y filter (ID blocks all share line-5 Y)

X_WALL_SPACING_EXTRAP_LIMIT = 4  # how many ID-block widths beyond the strict
                                 # detections to extrapolate when masking
                                 # partial / FOV-edge ID blocks. With chip
                                 # spacing ~500 px and FOV 2720 px wide, 4
                                 # extrapolations each side cover the whole
                                 # FOV plus a buffer.

# --- wall detection (after tilt correction) ----------------------------------

WALL_SEARCH_ABOVE = 1200       # px above ID block centre to search for top wall
WALL_SEARCH_BELOW = 1100       # px below ID block centre to search for bottom wall
WALL_MIN_GAP_ABOVE = 600       # walls must be at least this far ABOVE ID block centre
WALL_MIN_GAP_BELOW = 700       # walls must be at least this far BELOW ID block centre
WALL_SMOOTH = 31               # row-median smoothing window (px)
WALL_X_CENTRAL_FRAC = 0.6      # use only the central X fraction when computing
                                # the row signal (skips warpAffine black corners)
WALL_PROMINENCE_PCT = 0.10     # required dip depth as fraction of search-window range
WALL_LOCAL_HALFSPAN = 80       # px on each side used to score local wall dips.
                                # Prevents the bottom wall picker from jumping
                                # to darker lower-line structures.

# --- crop tightness ----------------------------------------------------------
# Insets push the top/bottom crop edge INSIDE line 5 by N px (positive
# values exclude the silicone wall itself). Values approved on the e015
# minitest after visual inspection.

WALL_INSET_TOP = 30
WALL_INSET_BOT = 20

# --- fallback geometry (used if wall detection fails) ------------------------
# Derived from p04 training crop: Y=500..2200, ID block centre at Y=1446.

CROP_OFFSET_ABOVE = 946
CROP_OFFSET_BELOW = 754


# =========================================================================
#  Template loading
# =========================================================================

def load_default_template(gaussian_sigma: float = 0.0) -> np.ndarray:
    """Load the default CellAsic line-5 ID-block template that ships with
    HiTMicTools. Returns a Sobel-preprocessed uint8 array ready to feed to
    `detect_id_blocks`. Pass `gaussian_sigma` matching the value used when
    Sobel-ing the target BF (so template and target share preprocessing)."""
    with resources.as_file(
        resources.files("HiTMicTools.img_processing.cellasic.templates") / "id_block.png"
    ) as template_path:
        raw = np.array(Image.open(template_path).convert("L"))
    return _apply_sobel(raw, gaussian_sigma=gaussian_sigma)


def load_template(path: str | Path, gaussian_sigma: float = 0.0) -> np.ndarray:
    """Load a user-supplied template from a PNG path. Applies the same
    Sobel preprocessing as the default template (with optional Gaussian
    smoothing matching the target image)."""
    raw = np.array(Image.open(path).convert("L"))
    return _apply_sobel(raw, gaussian_sigma=gaussian_sigma)


# =========================================================================
#  Preprocessing
# =========================================================================

def _to_uint8_norm(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Percentile-stretch to uint8."""
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    out = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
    return (out * 255).astype(np.uint8)


def _apply_sobel(img_u8: np.ndarray, gaussian_sigma: float = 0.0) -> np.ndarray:
    """Sobel gradient magnitude. Intensity-invariant; matches geometry
    rather than absolute contrast, so a low-contrast block in one chamber
    matches the same template as a high-contrast block in another.

    On RAW BF (raw .nd2), pillar-grid speckle noise dominates the Sobel
    signal and drowns the ID block. Set gaussian_sigma~2 to suppress
    speckle before gradient extraction; on post-NAFNet input the data is
    already denoised, so default sigma=0 is correct."""
    if gaussian_sigma > 0:
        k = int(2 * round(3 * gaussian_sigma) + 1)
        img_u8 = cv2.GaussianBlur(img_u8, (k, k), gaussian_sigma)
    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag /= (mag.max() + 1e-9)
    return (mag * 255).astype(np.uint8)


# =========================================================================
#  ID block detection
# =========================================================================

def _non_max_suppression(
    score_map: np.ndarray, threshold: float, min_sep: int, top_k: int
) -> List[Tuple[int, int, float]]:
    """Greedy NMS over a 2D score map. Returns (y, x, score) of accepted
    peaks (y, x are the top-left corner of the matched template patch).
    """
    peaks: List[Tuple[int, int, float]] = []
    sm = score_map.copy()
    H, W = sm.shape
    for _ in range(top_k):
        y, x = np.unravel_index(np.argmax(sm), sm.shape)
        s = float(sm[y, x])
        if s < threshold:
            break
        peaks.append((int(y), int(x), s))
        y0 = max(0, y - min_sep); y1 = min(H, y + min_sep + 1)
        x0 = max(0, x - min_sep); x1 = min(W, x + min_sep + 1)
        sm[y0:y1, x0:x1] = -1.0
    return peaks


def _extrapolate_id_positions(
    detections: List[Tuple[int, int, float]],
    image_shape: Tuple[int, int],
    extrap_limit: int = X_WALL_SPACING_EXTRAP_LIMIT,
) -> List[Tuple[int, int]]:
    """Given >=2 strict ID-block detections on the same line, return a
    list of (cx, cy) positions including the detected ones AND geometrically
    extrapolated positions to the left and right. The chamber chip has ID
    blocks at fixed X spacing along each line; extrapolating from the
    strict detections gives us locations for partial / FOV-edge ID blocks
    that NCC misses, so they can be masked out for X-wall detection.

    Returns positions in rotated-frame coords. If fewer than 2 detections
    are provided, just returns the detected centres unchanged.
    """
    if len(detections) < 2:
        return [(d[1], d[0]) for d in detections]
    H, W = image_shape
    centres = sorted([(d[1], d[0]) for d in detections], key=lambda p: p[0])
    xs = [p[0] for p in centres]
    spacing = float(np.median(np.diff(xs)))
    if spacing < 50:
        # detections too close; spacing unreliable
        return list(centres)
    # Use the leftmost detection's Y as the reference (all detections
    # share approximately the same Y, since they passed same-row filter)
    cy_ref = int(round(np.median([p[1] for p in centres])))
    result = list(centres)
    # Left extrapolation
    x = int(round(centres[0][0] - spacing))
    for _ in range(extrap_limit):
        result.append((x, cy_ref))
        x = int(round(x - spacing))
    # Right extrapolation
    x = int(round(centres[-1][0] + spacing))
    for _ in range(extrap_limit):
        result.append((x, cy_ref))
        x = int(round(x + spacing))
    return result


def detect_id_blocks(
    bf: np.ndarray,
    template_sobel: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    min_sep: int = DEFAULT_MIN_SEP,
    top_k: int = DEFAULT_TOP_K,
    max_y_drift: int = DEFAULT_MAX_Y_DRIFT,
    gaussian_sigma: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """Return ID block detections as a list of (centre_y, centre_x, score).

    Args:
        bf: 2D brightfield image (any dtype; converted to uint8 internally).
        template_sobel: Sobel-preprocessed template (use load_default_template()
            or load_template(path)). Must be loaded with the same gaussian_sigma
            as passed here.
        threshold: NCC score floor.
        min_sep: Minimum spacing between accepted detections.
        top_k: Max detections per frame.
        max_y_drift: Detections must lie within this Y distance of the median Y.
        gaussian_sigma: Sigma for pre-Sobel Gaussian smoothing. Use 0 on
            post-NAFNet (already denoised) input; ~2 on raw .nd2 BF where
            pillar speckle drowns the ID block.
    """
    bf_u8 = _to_uint8_norm(bf)
    bf_sobel = _apply_sobel(bf_u8, gaussian_sigma=gaussian_sigma)
    th, tw = template_sobel.shape

    score = cv2.matchTemplate(bf_sobel, template_sobel, cv2.TM_CCOEFF_NORMED)
    peaks = _non_max_suppression(score, threshold, min_sep, top_k)

    # Same-row Y filter
    if len(peaks) >= 2:
        centre_ys = np.array([p[0] + th // 2 for p in peaks])
        y_med = float(np.median(centre_ys))
        peaks = [p for p in peaks if abs((p[0] + th // 2) - y_med) <= max_y_drift]

    return [(p[0] + th // 2, p[1] + tw // 2, p[2]) for p in peaks]


# =========================================================================
#  Tilt + rotation
# =========================================================================

def _fit_line(centres_xy: np.ndarray) -> Tuple[float, float]:
    """Least squares y = slope*x + intercept. With <2 points: slope=0."""
    if len(centres_xy) < 2:
        return 0.0, float(centres_xy[0, 1])
    slope, intercept = np.polyfit(centres_xy[:, 0], centres_xy[:, 1], 1)
    return float(slope), float(intercept)


def compute_rotation(
    detections: List[Tuple[int, int, float]],
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Return (M, angle_deg, (centre_x, centre_y)). Rotation is anchored at
    the mean detected centre so that point is fixed by M."""
    centres = np.array(
        [(cx, cy) for (cy, cx, _) in detections], dtype=np.float64
    )
    centre_x = float(centres[:, 0].mean())
    centre_y = float(centres[:, 1].mean())
    slope, _ = _fit_line(centres)
    angle_deg = float(np.degrees(np.arctan(slope)))
    M = cv2.getRotationMatrix2D((centre_x, centre_y), angle_deg, 1.0)
    return M, angle_deg, (centre_x, centre_y)


# =========================================================================
#  Wall detection (in the rotated frame)
# =========================================================================

def detect_walls_in_rotated(
    rotated_bf_u8: np.ndarray,
    centre_y: float,
    bottom_picker: str = "first",
) -> Tuple[Optional[int], Optional[int], dict]:
    """Find the silicone walls above and below `centre_y` in a tilt-
    corrected BF frame. Returns (top_wall_y, bottom_wall_y, debug).
    Either side can be None if no wall passes the prominence threshold.

    Args:
        bottom_picker:
            "first"     -- pick the smallest-Y credible dip (default,
                           conservative for dense bacteria frames where a
                           lower-line smear can create a deeper minimum).
            "strongest" -- pick the deepest dip (symmetric with top wall).
                           Use on empty references where there is no
                           bacteria-smear risk.
    """
    H, W = rotated_bf_u8.shape
    cy = int(round(centre_y))

    top_search_lo = max(0, cy - WALL_SEARCH_ABOVE)
    top_search_hi = max(0, cy - WALL_MIN_GAP_ABOVE)
    bot_search_lo = min(H, cy + WALL_MIN_GAP_BELOW)
    bot_search_hi = min(H, cy + WALL_SEARCH_BELOW)

    margin = int(W * (1.0 - WALL_X_CENTRAL_FRAC) / 2.0)
    central = rotated_bf_u8[:, margin:W - margin].astype(np.float32)
    row_signal = np.median(central, axis=1)

    kernel = np.ones(WALL_SMOOTH) / WALL_SMOOTH
    row_smooth = np.convolve(row_signal, kernel, mode="same")

    def find_local_dips(lo: int, hi: int):
        if hi - lo < 10:
            return [], 0.0
        seg = row_smooth[lo:hi]
        seg_range = float(seg.max() - seg.min())
        if seg_range < 1e-6:
            return [], 0.0

        candidates = []
        for y in range(lo + 1, hi - 1):
            if row_smooth[y] > row_smooth[y - 1] or row_smooth[y] > row_smooth[y + 1]:
                continue
            left = max(lo, y - WALL_LOCAL_HALFSPAN)
            right = min(hi, y + WALL_LOCAL_HALFSPAN + 1)
            shoulder = max(
                float(row_smooth[left:y].max(initial=row_smooth[y])),
                float(row_smooth[y + 1:right].max(initial=row_smooth[y])),
            )
            depth = shoulder - float(row_smooth[y])
            if depth / seg_range >= WALL_PROMINENCE_PCT:
                candidates.append((y, depth))
        return candidates, seg_range

    top_candidates, top_range = find_local_dips(top_search_lo, top_search_hi)
    bot_candidates, bot_range = find_local_dips(bot_search_lo, bot_search_hi)

    top_y, top_depth = (
        max(top_candidates, key=lambda x: x[1]) if top_candidates else (None, 0.0)
    )
    if bottom_picker == "strongest":
        bot_y, bot_depth = (
            max(bot_candidates, key=lambda x: x[1]) if bot_candidates else (None, 0.0)
        )
    else:  # "first"
        bot_y, bot_depth = (
            min(bot_candidates, key=lambda x: x[0]) if bot_candidates else (None, 0.0)
        )

    return top_y, bot_y, {
        "top_depth": top_depth, "top_range": top_range,
        "bot_depth": bot_depth, "bot_range": bot_range,
        "top_candidates": len(top_candidates),
        "bot_candidates": len(bot_candidates),
        "row_smooth": row_smooth,
    }


# =========================================================================
#  X-wall (chamber side wall) detection
# =========================================================================

X_WALL_SEARCH_LEFT = 1400   # px to the LEFT of ID centre to scan for left wall
X_WALL_SEARCH_RIGHT = 1400  # px to the RIGHT of ID centre to scan for right wall
X_WALL_MIN_GAP = 250        # walls must be at least this far from the ID centre
X_WALL_PROMINENCE_PCT = 0.40  # require deep dips; pillar-grid background +
                              # masked-out ID blocks leave the chamber wall as
                              # the dominant col-median feature
X_WALL_SMOOTH = 31
X_WALL_INSET = 15           # px to push inward from the detected wall
X_WALL_ID_MASK_HALF = 250   # px half-width of the ID-block rectangle that
                            # gets masked out when computing col-median.
                            # Matches X_WALL_MIN_GAP so the search window
                            # never starts inside a masked-but-leaking
                            # ID-block edge.


def detect_chamber_x_walls(
    rotated_bf_u8: np.ndarray,
    centre_x: float,
    centre_y: float,
    crop_y0: int,
    crop_y1: int,
    id_block_centres_xy: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Optional[int], Optional[int], dict]:
    """Find chamber side walls (left + right) in the tilt-corrected BF.

    Strategy: take the line-5 Y band (already isolated by wall detection),
    but mask out the detected ID-block bboxes before computing col-median.
    With pillars on a regular grid and ID blocks removed, the remaining
    col-median is nearly uniform EXCEPT where chamber walls cross --
    those produce a deep, narrow dip.

    Args:
        rotated_bf_u8: tilt-corrected BF, uint8.
        centre_x, centre_y: ID block (mean) centre in rotated coords.
        crop_y0, crop_y1: line-5 Y band already determined.
        id_block_centres_xy: list of (cx, cy) for each detected ID block.
            Each gets masked out as a 2*X_WALL_ID_MASK_HALF square. If
            omitted, only the mean centre is masked.

    Returns (left_wall_x, right_wall_x, debug). Either may be None if no
    wall passes the prominence threshold within the search window.
    """
    H, W = rotated_bf_u8.shape
    cx = int(round(centre_x))

    band = rotated_bf_u8[max(0, crop_y0):min(H, crop_y1), :].astype(np.float32)
    if band.shape[0] < 20:
        return None, None, {"reason": "empty Y band"}

    # Mask out ID block bboxes. We work on a masked array so masked
    # columns are dropped from the median per-column.
    bh = band.shape[0]
    mask = np.zeros_like(band, dtype=bool)
    if id_block_centres_xy is None:
        id_block_centres_xy = [(cx, int(round(centre_y)))]
    half = X_WALL_ID_MASK_HALF
    for (icx, icy) in id_block_centres_xy:
        local_y = int(round(icy)) - max(0, crop_y0)
        if 0 <= local_y < bh:
            y_lo = max(0, local_y - half); y_hi = min(bh, local_y + half)
            x_lo = max(0, int(round(icx)) - half)
            x_hi = min(W, int(round(icx)) + half)
            mask[y_lo:y_hi, x_lo:x_hi] = True

    masked = np.ma.array(band, mask=mask)
    col_signal = np.ma.median(masked, axis=0).filled(np.median(band))

    kernel = np.ones(X_WALL_SMOOTH) / X_WALL_SMOOTH
    col_smooth = np.convolve(col_signal, kernel, mode="same")

    left_lo = max(0, cx - X_WALL_SEARCH_LEFT)
    left_hi = max(0, cx - X_WALL_MIN_GAP)
    right_lo = min(W, cx + X_WALL_MIN_GAP)
    right_hi = min(W, cx + X_WALL_SEARCH_RIGHT)

    def find_dips(lo: int, hi: int):
        if hi - lo < 10:
            return [], 0.0
        seg = col_smooth[lo:hi]
        seg_range = float(seg.max() - seg.min())
        if seg_range < 1e-6:
            return [], 0.0
        candidates = []
        for x in range(lo + 1, hi - 1):
            if col_smooth[x] > col_smooth[x - 1] or col_smooth[x] > col_smooth[x + 1]:
                continue
            left = max(lo, x - WALL_LOCAL_HALFSPAN)
            right = min(hi, x + WALL_LOCAL_HALFSPAN + 1)
            shoulder = max(
                float(col_smooth[left:x].max(initial=col_smooth[x])),
                float(col_smooth[x + 1:right].max(initial=col_smooth[x])),
            )
            depth = shoulder - float(col_smooth[x])
            if depth / seg_range >= X_WALL_PROMINENCE_PCT:
                candidates.append((x, depth))
        return candidates, seg_range

    left_cands, left_range = find_dips(left_lo, left_hi)
    right_cands, right_range = find_dips(right_lo, right_hi)

    # Post-filter: reject any candidate that falls within X_WALL_ID_MASK_HALF
    # of an extrapolated ID-block position. Masking alone leaks at mask
    # edges because col_smooth is convolved across mask boundaries.
    id_xs = [int(round(icx)) for (icx, _) in id_block_centres_xy]
    def too_close_to_id(x: int) -> bool:
        return any(abs(x - icx) < X_WALL_ID_MASK_HALF for icx in id_xs)
    left_cands = [c for c in left_cands if not too_close_to_id(c[0])]
    right_cands = [c for c in right_cands if not too_close_to_id(c[0])]

    # Strongest qualifying dip on each side. None = no wall in this FOV.
    left_x = max(left_cands, key=lambda c: c[1])[0] if left_cands else None
    right_x = max(right_cands, key=lambda c: c[1])[0] if right_cands else None

    return left_x, right_x, {
        "left_candidates": len(left_cands),
        "right_candidates": len(right_cands),
        "left_range": left_range,
        "right_range": right_range,
    }


def compute_crop_bounds(
    bf_rotated_u8: np.ndarray,
    centre_y: float,
    image_height: int,
    use_wall_detection: bool = True,
    wall_inset_top: int = WALL_INSET_TOP,
    wall_inset_bot: int = WALL_INSET_BOT,
    bottom_picker: str = "first",
) -> Tuple[int, int, str, dict]:
    """Decide [y0, y1] crop band. Tries walls first, falls back to fixed
    offsets if either wall isn't found. Returns (y0, y1, mode, debug)."""
    H = image_height
    if use_wall_detection:
        top_wall, bot_wall, dbg = detect_walls_in_rotated(
            bf_rotated_u8, centre_y, bottom_picker=bottom_picker,
        )
        if top_wall is not None and bot_wall is not None:
            y0 = max(0, top_wall + wall_inset_top)
            y1 = min(H, bot_wall - wall_inset_bot)
            return y0, y1, "walls", dbg
    y0 = max(0, int(round(centre_y - CROP_OFFSET_ABOVE)))
    y1 = min(H, int(round(centre_y + CROP_OFFSET_BELOW)))
    return y0, y1, "fixed", {}


# =========================================================================
#  Top-level entry point
# =========================================================================

class NoIDBlockDetected(RuntimeError):
    """Raised when no ID block fiducial can be located in the FOV. The
    pipeline cannot determine where the target line is, so processing
    should stop (hard fail rather than silently producing junk)."""


def crop_to_target_line(
    stack: np.ndarray,
    template_sobel: np.ndarray,
    bf_channel: int = 0,
    detect_frame: int = 0,
    use_wall_detection: bool = True,
    wall_inset_top: int = WALL_INSET_TOP,
    wall_inset_bot: int = WALL_INSET_BOT,
    gaussian_sigma: float = 0.0,
    threshold: float = DEFAULT_THRESHOLD,
    bottom_picker: str = "first",
    crop_x_walls: bool = False,
    x_wall_inset: int = X_WALL_INSET,
) -> Tuple[np.ndarray, dict]:
    """Detect, rotate, and crop a (T, C, Y, X) microscopy stack to the
    target line strip.

    Args:
        stack: (T, C, Y, X) array. Multi-frame multi-channel ok; the
            same affine is applied to every (frame, channel).
        template_sobel: Sobel-preprocessed template (see load_default_template).
        bf_channel: brightfield channel index (used for detection).
        detect_frame: frame index used to find the ID block.
        use_wall_detection: if True, locate the line-5 top/bottom silicone
            walls; else use fixed offsets from the ID block centre.
        wall_inset_top: px inset from detected top wall into line 5.
        wall_inset_bot: px inset from detected bottom wall into line 5.

    Returns:
        (cropped_stack, info)
        cropped_stack: (T, C, y1-y0, W) array, same dtype as input.
        info: dict with detection + crop diagnostics.

    Raises:
        NoIDBlockDetected: if zero ID blocks are found.
    """
    if stack.ndim != 4:
        raise ValueError(f"Expected (T, C, Y, X), got shape {stack.shape}")
    T, C, H, W = stack.shape

    bf = stack[detect_frame, bf_channel]
    detections = detect_id_blocks(
        bf, template_sobel,
        threshold=threshold,
        gaussian_sigma=gaussian_sigma,
    )
    if not detections:
        raise NoIDBlockDetected(
            f"No ID block detected on (frame={detect_frame}, channel={bf_channel}). "
            f"Cannot locate target line. Check FOV is within the chamber and the "
            f"focus restoration output is valid."
        )

    M, angle_deg, (cx, cy) = compute_rotation(detections, (H, W))

    # Rotate BF for wall detection
    bf_u8 = _to_uint8_norm(bf)
    bf_rotated = cv2.warpAffine(
        bf_u8, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    y0, y1, crop_mode, wall_dbg = compute_crop_bounds(
        bf_rotated, cy, H,
        use_wall_detection=use_wall_detection,
        wall_inset_top=wall_inset_top,
        wall_inset_bot=wall_inset_bot,
        bottom_picker=bottom_picker,
    )

    # Optional left/right chamber-wall crop. Only safe when the target
    # band (between y0 and y1) is uncluttered (e.g. empty-reference frames).
    x0 = 0
    x1 = W
    x_wall_dbg: dict = {}
    if crop_x_walls and crop_mode == "walls":
        # Geometric extrapolation of ID-block positions: with chip spacing
        # fixed along line 5, two strict detections give us the spacing
        # and we can predict where partial / FOV-edge ID blocks WOULD be.
        # All extrapolated positions get masked, so no ID artefact can be
        # mistaken for a chamber wall.
        rotated_centres: List[Tuple[int, int]] = []
        for (det_cy, det_cx, _) in detections:
            pt = M @ np.array([det_cx, det_cy, 1.0])
            rotated_centres.append((int(round(pt[0])), int(round(pt[1]))))
        # Convert (cx, cy) list back to (cy, cx, _) tuples for the
        # extrapolator's expected input format.
        rot_dets_for_extrap = [(c[1], c[0], 0.0) for c in rotated_centres]
        all_mask_centres = _extrapolate_id_positions(
            rot_dets_for_extrap, bf_rotated.shape,
        )
        lx, rx, x_wall_dbg = detect_chamber_x_walls(
            bf_rotated, cx, cy, y0, y1,
            id_block_centres_xy=all_mask_centres,
        )
        if lx is not None:
            x0 = max(0, lx + x_wall_inset)
        if rx is not None:
            x1 = min(W, rx - x_wall_inset)

    # Apply rotation + crop band to the full stack.
    out = np.zeros((T, C, y1 - y0, x1 - x0), dtype=stack.dtype)
    for t in range(T):
        for c in range(C):
            rot = cv2.warpAffine(
                stack[t, c], M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            out[t, c] = rot[y0:y1, x0:x1]

    info = {
        "n_detections": len(detections),
        "detections": detections,
        "angle_deg": angle_deg,
        "centre_xy": (cx, cy),
        "crop_y0": y0,
        "crop_y1": y1,
        "crop_x0": x0,
        "crop_x1": x1,
        "crop_height": y1 - y0,
        "crop_width": x1 - x0,
        "crop_mode": crop_mode,
        "wall_debug": {k: v for k, v in wall_dbg.items() if k != "row_smooth"},
        "x_wall_debug": x_wall_dbg,
    }
    return out, info


# =========================================================================
#  Calibration application -- pre-computed JSON instead of per-frame detect
# =========================================================================
# These helpers consume the unified calibration JSON produced by the e015
# calibration pipeline (see data/cropped_line5/calibration.json for format).
# They exist because per-frame wall + X detection is brittle on bacteria-
# heavy frames; a calibration derived from clean empty references is more
# reliable.

def clamp_tilt(angle_deg: float, max_abs: float) -> Tuple[float, bool]:
    """Clamp a tilt angle to +/- max_abs. Returns (clamped, was_clamped).

    Empty-reference chambers see plate-placement tilt of +-1 deg or less.
    Larger angles on experimental frames usually mean one ID block's NCC
    peak was shifted by surrounding bacteria, amplifying into a fake tilt
    in the 2-point line fit. Clamping rejects those.
    """
    if abs(angle_deg) <= max_abs:
        return angle_deg, False
    return (max_abs if angle_deg > 0 else -max_abs), True


def apply_y_calibration(
    centre_y: float, cal_y: dict, image_height: int,
) -> Tuple[int, int]:
    """Compute (y0, y1) from a calibration y_wall sub-dict.

    cal_y must contain `top_offset_from_id_cy` and `bot_offset_from_id_cy`
    (in pixels, in the rotated frame, relative to the detected ID centre Y).
    Values from the calibration JSON are typically ~-800 / +775.
    """
    top_off = int(cal_y["top_offset_from_id_cy"])
    bot_off = int(cal_y["bot_offset_from_id_cy"])
    y0 = max(0, int(round(centre_y + top_off)))
    y1 = min(image_height, int(round(centre_y + bot_off)))
    return y0, y1


def apply_x_calibration_minmax(
    detections: List[Tuple[int, int, float]],
    rotation_matrix: np.ndarray,
    cal_x: dict,
    image_width: int,
) -> Tuple[int, int, str]:
    """Compute (x0, x1, classification) from a calibration x_wall sub-dict.

    Strategy: rotate the detected ID centres into the rotated frame, then
    check whether min/max of those X positions sit beyond the configured
    thresholds. If yes, apply a fixed offset to derive the chamber wall.

    cal_x format (see calibration.json):
        left_wall.min_id_cx_threshold       -- min(id_cx) above this -> LEFT
        left_wall.offset_from_min_id_cx     -- wall X = min(id_cx) + offset
        left_wall.n_det_1_id_cx_threshold   -- for n_det=1 fallback
        left_wall.id_block_spacing_px       -- for n_det=1 fallback
        right_wall.* (symmetric, with max_id_cx_threshold)

    Classification labels for `crop_mode`:
        "centre"   -- no walls detected -> full FOV
        "LEFT"     -- left wall only (n_det >= 2)
        "RIGHT"    -- right wall only (n_det >= 2)
        "BOTH"     -- both walls (n_det >= 2)
        "LEFT_n1"  -- left wall via n_det=1 fallback
        "RIGHT_n1" -- right wall via n_det=1 fallback
        "BOTH_n1"  -- both walls via n_det=1 fallback (rare)
    """
    x0 = 0
    x1 = image_width
    classification = "centre"
    if not detections:
        return x0, x1, classification

    rot_xs: List[float] = []
    for (det_cy, det_cx, _) in detections:
        pt = rotation_matrix @ np.array([det_cx, det_cy, 1.0])
        rot_xs.append(float(pt[0]))

    left_cfg = cal_x.get("left_wall")
    right_cfg = cal_x.get("right_wall")
    min_x = min(rot_xs); max_x = max(rot_xs)

    if len(rot_xs) >= 2:
        if left_cfg is not None and min_x > left_cfg["min_id_cx_threshold"]:
            x0 = max(0, int(round(min_x + left_cfg["offset_from_min_id_cx"])))
            classification = "LEFT"
        if right_cfg is not None and max_x < right_cfg["max_id_cx_threshold"]:
            x1 = min(image_width, int(round(max_x + right_cfg["offset_from_max_id_cx"])))
            classification = "RIGHT" if classification == "centre" else "BOTH"
    else:
        # n_det == 1: classify by absolute position + estimate missing neighbour
        single = rot_xs[0]
        if left_cfg is not None and "n_det_1_id_cx_threshold" in left_cfg:
            if single > left_cfg["n_det_1_id_cx_threshold"]:
                spacing = left_cfg.get("id_block_spacing_px", 1033)
                est_min = single - spacing
                x0 = max(0, int(round(est_min + left_cfg["offset_from_min_id_cx"])))
                classification = "LEFT_n1"
        if right_cfg is not None and "n_det_1_id_cx_threshold" in right_cfg:
            if single < right_cfg["n_det_1_id_cx_threshold"]:
                spacing = right_cfg.get("id_block_spacing_px", 1033)
                est_max = single + spacing
                x1 = min(image_width, int(round(est_max + right_cfg["offset_from_max_id_cx"])))
                classification = "RIGHT_n1" if classification == "centre" else "BOTH_n1"
    return x0, x1, classification


def crop_with_calibration(
    stack: np.ndarray,
    template_sobel: np.ndarray,
    calibration: dict,
    bf_channel: int = 0,
    detect_frame: int = 0,
    gaussian_sigma: float = 0.0,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, dict]:
    """Detect, tilt-correct, and crop a stack using a pre-computed calibration.

    Use on bacteria-heavy / dense frames where per-frame wall detection is
    brittle. Use `crop_to_target_line` instead for empty-reference frames
    or when no calibration is available.

    Args:
        stack: (T, C, Y, X) array.
        template_sobel: Sobel-preprocessed ID-block template.
        calibration: dict with `x_wall`, `y_wall`, `tilt` sub-dicts (see
            data/cropped_line5/calibration.json for the canonical format).
        bf_channel: brightfield channel index used for detection.
        detect_frame: frame index used to detect the ID block.
        gaussian_sigma: pre-Sobel smoothing (typically 2.0 on raw .nd2 BF).
        threshold: NCC score floor for detections.

    Returns:
        (cropped_stack, info) -- info has the same keys as
        `crop_to_target_line` plus `_angle_clamped_from` if clamping fired.

    Raises:
        NoIDBlockDetected: if zero ID blocks are found.
        KeyError: if `calibration` is missing required sub-keys.
    """
    if stack.ndim != 4:
        raise ValueError(f"Expected (T, C, Y, X), got shape {stack.shape}")
    T, C, H, W = stack.shape

    bf = stack[detect_frame, bf_channel]
    detections = detect_id_blocks(
        bf, template_sobel,
        threshold=threshold,
        gaussian_sigma=gaussian_sigma,
    )
    if not detections:
        raise NoIDBlockDetected(
            f"No ID block detected on (frame={detect_frame}, channel={bf_channel})."
        )

    M, raw_angle, (cx, cy) = compute_rotation(detections, (H, W))

    # Tilt clamp -- defends against bacteria-induced spurious tilts.
    cal_tilt = calibration.get("tilt", {})
    max_abs = float(cal_tilt.get("max_abs_angle_deg", 1.0))
    clamped_angle, was_clamped = clamp_tilt(raw_angle, max_abs)
    if was_clamped:
        M = cv2.getRotationMatrix2D((cx, cy), clamped_angle, 1.0)

    # Y crop band from calibration (replaces per-frame wall detection).
    y0, y1 = apply_y_calibration(cy, calibration["y_wall"], H)

    # X crop band from min/max ID positions in the rotated frame.
    x0, x1, classification = apply_x_calibration_minmax(
        detections, M, calibration["x_wall"], W,
    )

    # Apply rotation + crop band to the full stack.
    out = np.zeros((T, C, y1 - y0, x1 - x0), dtype=stack.dtype)
    for t in range(T):
        for c in range(C):
            rot = cv2.warpAffine(
                stack[t, c], M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            out[t, c] = rot[y0:y1, x0:x1]

    info = {
        "n_detections": len(detections),
        "detections": detections,
        "angle_deg": clamped_angle,
        "centre_xy": (cx, cy),
        "crop_y0": y0,
        "crop_y1": y1,
        "crop_x0": x0,
        "crop_x1": x1,
        "crop_height": y1 - y0,
        "crop_width": x1 - x0,
        "crop_mode": f"calibrated_y+calibrated_x_{classification}",
    }
    if was_clamped:
        info["_angle_clamped_from"] = raw_angle
    return out, info
