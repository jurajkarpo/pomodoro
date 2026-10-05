"""Illumination normalization and shadow reduction.

Photographed pages often have uneven lighting, a dark shadow near the
spine, or a bright flash hotspot. These operations normalize the
background illumination while preserving the text and page appearance.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_background(gray: np.ndarray, block: int = 61) -> np.ndarray:
    """Estimate the slowly-varying background illumination."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    k = block | 1
    return cv2.medianBlur(gray, k)


def normalize_illumination(
    arr: np.ndarray,
    strength: float = 1.0,
    block: int = 61,
    target: int = 235,
) -> np.ndarray:
    """Flat-field illumination normalization."""
    if strength <= 0:
        return arr
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    bg = estimate_background(gray, block).astype(np.float32)
    bg = np.clip(bg, 20.0, 255.0)
    ratio = (target / bg)
    ratio = 1.0 + (ratio - 1.0) * strength
    if arr.ndim == 2:
        out = gray.astype(np.float32) * ratio
        return np.clip(out, 0, 255).astype(np.uint8)
    out = arr.astype(np.float32)
    out = out * ratio[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def reduce_shadows(
    arr: np.ndarray,
    strength: float = 1.0,
    block: int = 81,
) -> np.ndarray:
    """Reduce dark shadows (e.g. near the spine) via morphological
    background estimation and blending."""
    if strength <= 0:
        return arr
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    k = block | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = np.clip(bg.astype(np.float32), 20.0, 255.0)
    gain = 200.0 / bg
    gain = 1.0 + (gain - 1.0) * strength
    gain = np.clip(gain, 0.8, 2.5)
    if arr.ndim == 2:
        out = gray.astype(np.float32) * gain
        return np.clip(out, 0, 255).astype(np.uint8)
    out = arr.astype(np.float32) * gain[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_enhance(gray: np.ndarray, clip: float = 2.0, grid: int = 16) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)
