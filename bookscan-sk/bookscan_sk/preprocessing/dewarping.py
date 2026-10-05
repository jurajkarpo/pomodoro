"""Book-page dewarping.

The goal is to flatten curved text baselines and page curl without
introducing extreme distortion. The approach:

1. Binarize the page and detect text lines (connected components grouped
   into lines by vertical proximity).
2. For each text line, fit a smooth polynomial baseline y(x).
3. Measure the curvature as the deviation of the baseline from its own
   linear (straight) fit.
4. Build a *vertical* displacement field that counteracts the curvature,
   interpolated smoothly across the whole page.
5. Bound the maximum displacement and smooth heavily so the warp is gentle.
6. Apply with cv2.remap.

If curvature cannot be reliably estimated (few lines, low confidence),
the function returns the input unchanged so the caller can fall back to
perspective correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.errors import DewarpingError
from ..core.types import DewarpStrength


@dataclass
class DewarpResult:
    image: np.ndarray
    applied: bool
    max_displacement_px: float
    curvature_score: float
    method: str


_STRENGTH_SCALE = {
    DewarpStrength.OFF: 0.0,
    DewarpStrength.LOW: 0.4,
    DewarpStrength.MEDIUM: 0.7,
    DewarpStrength.HIGH: 1.0,
    DewarpStrength.AUTO: 1.0,
}

_MAX_DISPLACEMENT_FRACTION = 0.06


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive binarization tuned for photographed pages."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=35, C=12,
    )


def _detect_text_lines(bw: np.ndarray) -> list[np.ndarray]:
    """Detect text lines as horizontal bands of connected components."""
    h, w = bw.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 80), 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, horiz_kernel, iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[np.ndarray] = []
    min_line_width = w * 0.15
    min_line_height = max(6, h * 0.008)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < min_line_width or ch < min_line_height or ch > h * 0.25:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y : y + ch, x : x + cw] = 255
        lines.append(mask)
    lines.sort(key=lambda m: int(np.argmax(m.sum(axis=1) > 0)))
    return lines


def _fit_baseline(mask: np.ndarray, degree: int = 2) -> Optional[np.ndarray]:
    """Fit a polynomial baseline y(x) to a text-line mask."""
    h, w = mask.shape
    ys: list[float] = []
    xs: list[float] = []
    col_has = np.any(mask > 0, axis=0)
    if not np.any(col_has):
        return None
    step = max(1, w // 200)
    for x in range(0, w, step):
        col = mask[:, x]
        idx = np.nonzero(col)[0]
        if len(idx) == 0:
            continue
        bottom = idx[int(len(idx) * 0.7):]
        baseline_y = float(np.median(bottom))
        xs.append(float(x))
        ys.append(baseline_y)
    if len(xs) < degree + 3:
        return None
    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)
    try:
        coeffs = np.polyfit(xs_arr, ys_arr, degree)
    except Exception:
        return None
    return coeffs


def _curvature_of(coeffs: np.ndarray, w: int) -> np.ndarray:
    """Curvature = deviation of the polynomial from its linear fit."""
    x = np.arange(w, dtype=np.float64)
    y = np.polyval(coeffs, x)
    linear = np.polyfit(x, y, 1)
    y_linear = np.polyval(linear, x)
    return y - y_linear


def estimate_curvature_field(gray: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Estimate a vertical displacement field to flatten text baselines.

    Returns (field, curvature_score, ok). The field has shape (h, w) and
    contains the y-displacement (in pixels) to apply at each point.
    """
    h, w = gray.shape
    bw = _binarize(gray)
    lines = _detect_text_lines(bw)
    if len(lines) < 3:
        return np.zeros((h, w), dtype=np.float32), 0.0, False

    curves: list[np.ndarray] = []
    for mask in lines:
        coeffs = _fit_baseline(mask)
        if coeffs is None:
            continue
        curv = _curvature_of(coeffs, w)
        if np.max(np.abs(curv)) < 1.0:
            continue
        curves.append(curv)
    if not curves:
        return np.zeros((h, w), dtype=np.float32), 0.0, False

    min_len = min(len(c) for c in curves)
    stacked = np.stack([c[:min_len] for c in curves])
    mean_curv = np.mean(stacked, axis=0)
    score = float(np.mean(np.abs(mean_curv)))

    field = np.zeros((h, w), dtype=np.float32)
    x_sample = np.linspace(0, min_len - 1, w)
    curv_full = np.interp(x_sample, np.arange(min_len), mean_curv).astype(np.float32)
    for y in range(h):
        t = y / max(1, h - 1)
        taper = np.sin(np.pi * t) ** 0.5
        field[y, :] = -curv_full * taper

    max_disp = _MAX_DISPLACEMENT_FRACTION * h
    field = np.clip(field, -max_disp, max_disp)
    ksize = max(3, (w // 40) | 1)
    field = cv2.GaussianBlur(field, (ksize, ksize), 0)

    return field, score, True


def _apply_field(arr: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Apply a vertical displacement field via remap."""
    h, w = arr.shape[:2]
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.arange(h, dtype=np.float32).reshape(h, 1)
    map_y = ys + field
    return cv2.remap(
        arr, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def dewarp(
    arr: np.ndarray,
    strength: DewarpStrength = DewarpStrength.AUTO,
    max_side: int = 2600,
) -> DewarpResult:
    """Dewarp a (perspective-corrected) page image.

    Returns a DewarpResult. If dewarping cannot be reliably estimated,
    applied=False and the original array is returned unchanged.
    """
    scale = 1.0
    work = arr
    h, w = arr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        work = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    field, score, ok = estimate_curvature_field(gray)

    if strength == DewarpStrength.OFF:
        return DewarpResult(image=arr, applied=False, max_displacement_px=0.0,
                            curvature_score=score, method="off")

    scale_factor = _STRENGTH_SCALE.get(strength, 1.0)

    if not ok or score < 0.6:
        return DewarpResult(image=arr, applied=False, max_displacement_px=0.0,
                            curvature_score=score, method="fallback")

    field = field * scale_factor

    if scale != 1.0:
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / scale)

    max_disp = float(np.max(np.abs(field))) if field.size else 0.0
    if max_disp < 0.5:
        return DewarpResult(image=arr, applied=False, max_displacement_px=0.0,
                            curvature_score=score, method="fallback")

    dewarped = _apply_field(arr, field)
    return DewarpResult(image=dewarped, applied=True, max_displacement_px=max_disp,
                        curvature_score=score, method="baseline")


def auto_dewarp(arr: np.ndarray) -> DewarpResult:
    """Convenience: dewarp with AUTO strength."""
    return dewarp(arr, DewarpStrength.AUTO)
