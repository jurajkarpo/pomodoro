"""Two-page spread detection and central gutter localization.

A photographed open book shows two pages side by side with a
shadowed gutter between them. This module:

  1. Decides whether a detected page is a single page or a spread.
  2. Locates the central gutter (the spine shadow).
  3. Splits a spread into left and right page quads.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.types import PageLayout, Point, Quad
from ..utils.geometry import quad_height, quad_width


_SPREAD_ASPECT_THRESHOLD = 1.55


def estimate_layout(quad: Quad) -> PageLayout:
    """Estimate whether a quad is a single page or a spread."""
    w = quad_width(quad)
    h = quad_height(quad)
    if h <= 0:
        return PageLayout.UNKNOWN
    aspect = w / h
    if aspect >= _SPREAD_ASPECT_THRESHOLD:
        return PageLayout.SPREAD
    return PageLayout.SINGLE


def find_gutter_x(gray: np.ndarray, quad: Quad) -> Optional[float]:
    """Find the x-coordinate of the central gutter (spine shadow)."""
    h, w = gray.shape
    if w < 40:
        return None
    y0 = int(h * 0.15)
    y1 = int(h * 0.85)
    if y1 <= y0:
        return None
    band = gray[y0:y1, :]
    if band.dtype != np.uint8:
        band = np.clip(band, 0, 255).astype(np.uint8)
    col_mean = band.mean(axis=0).astype(np.float32)
    k = max(3, (w // 60) | 1)
    col_mean = cv2.GaussianBlur(col_mean, (k, 1), 0)
    cx0 = int(w * 0.15)
    cx1 = int(w * 0.85)
    if cx1 <= cx0:
        return None
    search = col_mean[cx0:cx1]
    if len(search) == 0:
        return None
    global_min = int(np.argmin(search)) + cx0
    min_val = float(col_mean[global_min])
    median_val = float(np.median(col_mean))
    if median_val - min_val < 12.0:
        return None
    center = w / 2.0
    if abs(global_min - center) > w * 0.25:
        return None
    return float(global_min)


def split_spread(quad: Quad, gutter_x: Optional[float], img_w: int) -> Tuple[Quad, Quad]:
    """Split a spread quad into left and right page quads."""
    if gutter_x is None:
        top_mid = Point((quad.tl.x + quad.tr.x) / 2.0, (quad.tl.y + quad.tr.y) / 2.0)
        bot_mid = Point((quad.bl.x + quad.br.x) / 2.0, (quad.bl.y + quad.br.y) / 2.0)
    else:
        top_mid = _interp_x_on_edge(quad.tl, quad.tr, gutter_x, img_w)
        bot_mid = _interp_x_on_edge(quad.bl, quad.br, gutter_x, img_w)

    left = Quad(tl=quad.tl, tr=top_mid, br=bot_mid, bl=quad.bl)
    right = Quad(tl=top_mid, tr=quad.tr, br=quad.br, bl=bot_mid)
    return left, right


def _interp_x_on_edge(p0: Point, p1: Point, target_x: float, img_w: int) -> Point:
    """Find the point on edge (p0,p1) whose x is closest to target_x."""
    dx = p1.x - p0.x
    if abs(dx) < 1e-6:
        return Point((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0)
    t = (target_x - p0.x) / dx
    t = min(1.0, max(0.0, t))
    return Point(p0.x + t * dx, p0.y + t * (p1.y - p0.y))
