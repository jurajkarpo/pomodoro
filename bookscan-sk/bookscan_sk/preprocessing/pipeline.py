"""Preprocessing pipeline orchestration.

Applies the configured sequence of operations to a detected page:
orientation -> crop/perspective -> dewarp -> illumination ->
enhancement -> border cleanup.

The original image is never modified; a new processed array is
returned and the caller persists both original and processed copies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..core.errors import DewarpingError
from ..core.types import (
    DetectedPage,
    DewarpStrength,
    PreprocessPreset,
    ProcessedPage,
    ProcessingSettings,
    SourceImage,
)
from . import dewarping, enhancement, illumination, orientation, perspective


@dataclass
class PreprocessResult:
    image: np.ndarray
    applied: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    transform_matrix: Optional[list[list[float]]] = None
    dewarp_applied: bool = False
    dewarp_max_displacement: float = 0.0


def _apply_preset_knobs(
    arr: np.ndarray,
    settings: ProcessingSettings,
) -> tuple[np.ndarray, dict[str, bool]]:
    """Apply enhancement knobs from settings/preset."""
    applied: dict[str, bool] = {}
    out = arr

    if settings.shadow_reduction:
        out = illumination.reduce_shadows(out, strength=0.8)
        applied["shadow_reduction"] = True

    if settings.illumination_norm:
        out = illumination.normalize_illumination(out, strength=0.7)
        applied["illumination_norm"] = True

    if settings.grayscale:
        out = enhancement.to_grayscale(out)
        applied["grayscale"] = True

    if abs(settings.contrast - 1.0) > 1e-6:
        out = enhancement.adjust_contrast(out, settings.contrast)
        applied["contrast"] = True

    if settings.denoise > 0:
        out = enhancement.denoise(out, settings.denoise)
        applied["denoise"] = True

    if settings.sharpen > 0:
        out = enhancement.sharpen(out, settings.sharpen)
        applied["sharpen"] = True

    if settings.adaptive_threshold or settings.black_white:
        bw = enhancement.adaptive_threshold(out, block=35, c=8)
        out = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB) if out.ndim == 3 else bw
        applied["adaptive_threshold"] = True

    if settings.black_white and out.ndim == 3:
        gray = enhancement.to_grayscale(out)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        applied["black_white"] = True

    out = enhancement.cleanup_borders(out, margin_frac=0.008)
    applied["border_cleanup"] = True

    return out, applied


def preprocess_page(
    image: np.ndarray,
    detected: DetectedPage,
    settings: ProcessingSettings,
    output_size: Optional[tuple[int, int]] = None,
) -> PreprocessResult:
    """Run the full preprocessing pipeline on one detected page."""
    notes: list[str] = []
    applied: dict[str, bool] = {}

    arr = image.copy()

    if abs(settings.rotate_degrees) > 1e-6:
        arr = orientation.rotate(arr, settings.rotate_degrees)
        applied["rotate"] = True

    matrix: Optional[list[list[float]]] = None
    if detected.quad is not None:
        size = output_size or perspective.estimate_output_size(detected.quad)
        arr, m = perspective.correct_perspective(arr, detected.quad)
        matrix = m.tolist()
        applied["perspective"] = True
        if perspective.needs_perspective_correction(detected.quad):
            notes.append("perspective corrected")

    arr, angle = orientation.deskew(arr, max_angle=8.0)
    if abs(angle) > 0.05:
        applied["deskew"] = True
        notes.append(f"deskewed {angle:.2f} deg")

    dewarp_applied = False
    dewarp_max_disp = 0.0
    if settings.dewarp != DewarpStrength.OFF:
        try:
            dw = dewarping.dewarp(arr, settings.dewarp)
            if dw.applied:
                arr = dw.image
                dewarp_applied = True
                dewarp_max_disp = dw.max_displacement_px
                applied["dewarp"] = True
                notes.append(f"dewarped (max {dw.max_displacement_px:.1f}px)")
            else:
                notes.append(f"dewarp skipped ({dw.method})")
        except (DewarpingError, Exception) as e:  # noqa: BLE001
            notes.append(f"dewarp error: {e}")

    arr, knob_applied = _apply_preset_knobs(arr, settings)
    applied.update(knob_applied)

    return PreprocessResult(
        image=arr,
        applied=applied,
        notes=notes,
        transform_matrix=matrix,
        dewarp_applied=dewarp_applied,
        dewarp_max_displacement=dewarp_max_disp,
    )


def build_processed_page(
    source: SourceImage,
    detected: DetectedPage,
    image: np.ndarray,
    result: PreprocessResult,
    settings: ProcessingSettings,
    processed_path: str,
    original_path: str,
    dpi: int = 300,
) -> ProcessedPage:
    """Assemble a ProcessedPage record from pipeline outputs."""
    h, w = image.shape[:2]
    return ProcessedPage(
        source=source,
        image_path=processed_path,
        original_path=original_path,
        width=w,
        height=h,
        dpi=dpi,
        quad=detected.quad,
        transform_matrix=result.transform_matrix,
        preset=settings.preset,
        dewarp_strength=settings.dewarp,
        dewarp_applied=result.dewarp_applied,
        status=ProcessedPage.status,
        warnings=result.notes,
    )
