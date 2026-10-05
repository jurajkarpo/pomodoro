"""Orientation correction: EXIF transpose, rotation, and skew deskewing."""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def apply_exif_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation value (1-8) to an image array."""
    if orientation == 1:
        return arr
    if orientation == 2:
        return cv2.flip(arr, 1)
    if orientation == 3:
        return cv2.flip(arr, -1)
    if orientation == 4:
        return cv2.flip(arr, 0)
    if orientation == 5:
        return cv2.flip(cv2.transpose(arr), 1)
    if orientation == 6:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(arr), 0)
    if orientation == 8:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return arr


def rotate(arr: np.ndarray, angle_degrees: float, border_value: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Rotate an image by an arbitrary angle, expanding the canvas."""
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    matrix[0, 2] += (new_w - w) / 2.0
    matrix[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(arr, matrix, (new_w, new_h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle (degrees) of a text page using Hough lines."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = 1200.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    if small.dtype != np.uint8:
        small = np.clip(small, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(
        bw, 1, np.pi / 180, threshold=80, minLineLength=int(small.shape[1] * 0.4), maxLineGap=8
    )
    if lines is None or len(lines) == 0:
        return 0.0
    angles: list[tuple[float, float]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        length = math.hypot(dx, dy)
        angles.append((angle, length))
    if not angles:
        return 0.0
    angles.sort(key=lambda a: a[0])
    total = sum(l for _, l in angles)
    cum = 0.0
    for angle, length in angles:
        cum += length
        if cum >= total / 2.0:
            return float(angle)
    return float(angles[len(angles) // 2][0])


def deskew(arr: np.ndarray, max_angle: float = 15.0) -> Tuple[np.ndarray, float]:
    """Automatically deskew a photographed page.

    Returns (deskewed_image, applied_angle_degrees).
    """
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    angle = estimate_skew_angle(gray)
    if abs(angle) > max_angle:
        angle = 0.0
    if abs(angle) < 0.05:
        return arr, 0.0
    return rotate(arr, -angle), -angle
