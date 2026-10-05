"""Perspective correction for photographed pages."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Quad
from ..utils.geometry import (
    destination_quad,
    perspective_strength,
    perspective_transform,
    quad_height,
    quad_width,
)


def warp_perspective(arr: np.ndarray, quad: Quad, output_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image so the quad becomes a rectangle.

    Returns (warped_image, transform_matrix). The transform maps source
    pixel coordinates to output pixel coordinates.
    """
    w = quad_width(quad)
    h = quad_height(quad)
    if output_size is not None:
        out_w, out_h = output_size
    else:
        out_w = int(round(w))
        out_h = int(round(h))
    out_w = max(1, out_w)
    out_h = max(1, out_h)
    dst = destination_quad(out_w, out_h)
    matrix = perspective_transform(quad, dst)
    warped = cv2.warpPerspective(
        arr, matrix, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped, matrix


def correct_perspective(arr: np.ndarray, quad: Quad) -> tuple[np.ndarray, np.ndarray]:
    """Alias for warp_perspective with automatic output sizing."""
    return warp_perspective(arr, quad)


def estimate_output_size(quad: Quad, max_side: int = 4000) -> tuple[int, int]:
    """Estimate a sensible output size for a page quad, capped at max_side."""
    w = int(round(quad_width(quad)))
    h = int(round(quad_height(quad)))
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        w = int(round(w * scale))
        h = int(round(h * scale))
    return max(1, w), max(1, h)


def needs_perspective_correction(quad: Quad, threshold: float = 0.02) -> bool:
    """Return True if the quad shows meaningful perspective distortion."""
    return perspective_strength(quad) > threshold
