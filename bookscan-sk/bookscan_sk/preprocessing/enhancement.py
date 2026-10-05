"""Image enhancement: contrast, sharpening, denoising, thresholding."""
from __future__ import annotations

import cv2
import numpy as np


def adjust_contrast(arr: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust contrast. factor=1.0 is a no-op."""
    if abs(factor - 1.0) < 1e-6:
        return arr
    if arr.ndim == 2:
        mean = float(arr.mean())
        out = (arr.astype(np.float32) - mean) * factor + mean
        return np.clip(out, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float32)
    mean = float(l.mean())
    l = (l - mean) * factor + mean
    lab[:, :, 0] = np.clip(l, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def sharpen(arr: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Unsharp mask sharpening. amount=0 is a no-op."""
    if amount <= 0:
        return arr
    blur = cv2.GaussianBlur(arr, (0, 0), 1.2)
    out = arr.astype(np.float32) + amount * (arr.astype(np.float32) - blur.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise(arr: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """Denoise. strength=0 is a no-op. Uses a bilateral filter."""
    if strength <= 0:
        return arr
    d = int(round(3 + strength * 6))
    d = max(3, d | 1)
    sigma = 20.0 + strength * 40.0
    return cv2.bilateralFilter(arr, d, sigma, sigma)


def adaptive_threshold(arr: np.ndarray, block: int = 31, c: int = 10) -> np.ndarray:
    """Adaptive binarization (returns single-channel uint8)."""
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    k = block | 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=k, C=c,
    )


def otsu_threshold(arr: np.ndarray) -> np.ndarray:
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def to_black_white(arr: np.ndarray) -> np.ndarray:
    """High-contrast black & white (single channel)."""
    return adaptive_threshold(arr, block=35, c=8)


def cleanup_borders(arr: np.ndarray, margin_frac: float = 0.01) -> np.ndarray:
    """Whiten a thin border around the page to remove scan artifacts."""
    h, w = arr.shape[:2]
    m = int(round(min(h, w) * margin_frac))
    if m <= 0:
        return arr
    out = arr.copy()
    out[:m, :] = 255
    out[-m:, :] = 255
    out[:, :m] = 255
    out[:, -m:] = 255
    return out
