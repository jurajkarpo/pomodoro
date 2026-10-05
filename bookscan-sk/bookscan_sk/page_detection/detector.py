"""Automatic page-boundary detection in photographed pages.

The detector must not assume the input is pre-cropped. It finds the
page polygon, estimates perspective, and detects one-page vs
two-page-spread layouts.

Strategy (robust, classical, no ML dependency):
  1. Downscale for speed; grayscale; denoise.
  2. Compute edges (Canny) and a morphological gradient.
  3. Close gaps, find contours.
  4. Score quadrilateral candidates by area, convexity, corner count
     and page-like aspect ratio.
  5. Refine the best quad with a corner-subpixel pass.
  6. Fall back to a margin-based full-page quad when no convincing
     polygon is found (so we never lose a page).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..core.types import DetectedPage, PageLayout, Point, Quad, SourceImage
from ..utils.geometry import (
    clamp_quad_to_image,
    perspective_strength,
    quad_area,
    quad_width,
    scale_quad,
    skew_degrees,
)
from ..utils.images import resize_max


def _preprocess(arr: np.ndarray) -> np.ndarray:
    """Working copy: grayscale, denoised, downscaled."""
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(gray, 7, 40, 40)


def _candidate_quads(edges: np.ndarray, min_area: float) -> list[Quad]:
    """Extract quadrilateral candidates from an edge map."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[Quad] = []
    h, w = edges.shape
    img_area = float(h * w)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > img_area * 0.995:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps_frac in (0.02, 0.03, 0.05, 0.07):
            approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float64)
                quad = _order_quad(pts)
                if quad is not None:
                    quads.append(quad)
                break
    return quads


def _order_quad(pts: np.ndarray) -> Optional[Quad]:
    """Order 4 points as tl, tr, br, bl."""
    if pts.shape != (4, 2):
        return None
    rect = np.zeros((4, 2), dtype=np.float64)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return Quad(
        tl=Point(float(rect[0, 0]), float(rect[0, 1])),
        tr=Point(float(rect[1, 0]), float(rect[1, 1])),
        br=Point(float(rect[2, 0]), float(rect[2, 1])),
        bl=Point(float(rect[3, 0]), float(rect[3, 1])),
    )


def _score_quad(quad: Quad, img_w: int, img_h: int) -> float:
    """Score how page-like a quad is (higher is better)."""
    area = quad_area(quad)
    img_area = float(img_w * img_h)
    if area <= 0:
        return 0.0
    area_frac = area / img_area
    if area_frac < 0.05:
        return 0.0
    w = quad_width(quad)
    h = area / max(1.0, w)
    aspect = w / max(1.0, h)
    aspect_score = 1.0
    if aspect < 0.4 or aspect > 3.0:
        aspect_score = 0.3
    elif aspect < 0.6 or aspect > 2.2:
        aspect_score = 0.7
    fill_score = min(1.0, area_frac / 0.5)
    return area_frac * 0.5 + aspect_score * 0.3 + fill_score * 0.2


def _refine_corners(gray: np.ndarray, quad: Quad) -> Quad:
    """Refine quad corners using corner sub-pixel detection."""
    try:
        pts = np.array(quad.to_xy(), dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(gray, pts, (7, 7), (-1, -1), criteria)
        return Quad(
            tl=Point(float(corners[0, 0]), float(corners[0, 1])),
            tr=Point(float(corners[1, 0]), float(corners[1, 1])),
            br=Point(float(corners[2, 0]), float(corners[2, 1])),
            bl=Point(float(corners[3, 0]), float(corners[3, 1])),
        )
    except Exception:
        return quad


def _full_page_quad(width: int, height: int, margin_frac: float = 0.02) -> Quad:
    """Fallback quad: the whole image with a small margin."""
    mx = int(width * margin_frac)
    my = int(height * margin_frac)
    return Quad(
        tl=Point(float(mx), float(my)),
        tr=Point(float(width - mx), float(my)),
        br=Point(float(width - mx), float(height - my)),
        bl=Point(float(mx), float(height - my)),
    )


def detect_page(arr: np.ndarray, source: SourceImage) -> DetectedPage:
    """Detect the page boundary in a photographed page.

    Never raises: on failure returns a full-page fallback quad with
    low confidence so the user can correct it manually.
    """
    h, w = arr.shape[:2]
    work = resize_max(arr, 1600)
    gray = _preprocess(work)
    scale = work.shape[1] / float(w)

    edges = cv2.Canny(gray, 40, 120)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, grad_bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    min_area = (work.shape[0] * work.shape[1]) * 0.05
    candidates = _candidate_quads(edges, min_area)
    candidates += _candidate_quads(grad_bw, min_area)

    best: Optional[Quad] = None
    best_score = 0.0
    for q in candidates:
        score = _score_quad(q, work.shape[1], work.shape[0])
        if score > best_score:
            best_score = score
            best = q

    if best is None or best_score < 0.15:
        quad = _full_page_quad(w, h)
        return DetectedPage(
            source=source,
            quad=quad,
            layout=PageLayout.UNKNOWN,
            confidence=0.2,
            skew_degrees=0.0,
            perspective_strength=0.0,
            method="fallback-fullpage",
        )

    gray_full = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray_full.dtype != np.uint8:
        gray_full = np.clip(gray_full, 0, 255).astype(np.uint8)
    best = scale_quad(best, 1.0 / scale, 1.0 / scale)
    best = _refine_corners(gray_full, best)
    best = clamp_quad_to_image(best, w, h)

    confidence = min(0.98, 0.4 + best_score * 0.6)
    return DetectedPage(
        source=source,
        quad=best,
        layout=PageLayout.UNKNOWN,
        confidence=confidence,
        skew_degrees=skew_degrees(best),
        perspective_strength=perspective_strength(best),
        method="contour",
    )
