"""Geometry helpers for page polygons, perspective and coordinate mapping."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from ..core.types import Point, Quad, Rect


def quad_width(quad: Quad) -> float:
    """Average of top and bottom edge lengths."""
    top = math.dist((quad.tl.x, quad.tl.y), (quad.tr.x, quad.tr.y))
    bottom = math.dist((quad.bl.x, quad.bl.y), (quad.br.x, quad.br.y))
    return (top + bottom) / 2.0


def quad_height(quad: Quad) -> float:
    """Average of left and right edge lengths."""
    left = math.dist((quad.tl.x, quad.tl.y), (quad.bl.x, quad.bl.y))
    right = math.dist((quad.tr.x, quad.tr.y), (quad.br.x, quad.br.y))
    return (left + right) / 2.0


def quad_area(quad: Quad) -> float:
    """Shoelace area of the quad."""
    pts = quad.to_xy()
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def quad_centroid(quad: Quad) -> Point:
    pts = quad.to_xy()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return Point(sum(xs) / 4.0, sum(ys) / 4.0)


def perspective_strength(quad: Quad) -> float:
    """Estimate how non-rectangular the quad is (0..1)."""
    top = math.dist((quad.tl.x, quad.tl.y), (quad.tr.x, quad.tr.y))
    bottom = math.dist((quad.bl.x, quad.bl.y), (quad.br.x, quad.br.y))
    left = math.dist((quad.tl.x, quad.tl.y), (quad.bl.x, quad.bl.y))
    right = math.dist((quad.tr.x, quad.tr.y), (quad.br.x, quad.br.y))
    if top <= 0 or bottom <= 0 or left <= 0 or right <= 0:
        return 0.0
    w_diff = abs(top - bottom) / max(top, bottom)
    h_diff = abs(left - right) / max(left, right)
    return min(1.0, (w_diff + h_diff) / 2.0)


def skew_degrees(quad: Quad) -> float:
    """Rotation of the top edge relative to horizontal, in degrees."""
    dx = quad.tr.x - quad.tl.x
    dy = quad.tr.y - quad.tl.y
    if dx == 0:
        return 90.0
    return math.degrees(math.atan2(dy, dx))


def destination_quad(width: float, height: float) -> Quad:
    """Axis-aligned destination quad for a page of given size."""
    return Quad(
        tl=Point(0, 0),
        tr=Point(width, 0),
        br=Point(width, height),
        bl=Point(0, height),
    )


def perspective_transform(src: Quad, dst: Quad) -> np.ndarray:
    """Compute the 3x3 perspective transform from src quad to dst quad."""
    import cv2

    src_pts = np.array(src.to_xy(), dtype="float32")
    dst_pts = np.array(dst.to_xy(), dtype="float32")
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def map_point(matrix: np.ndarray, point: Point) -> Point:
    """Apply a 3x3 perspective matrix to a point."""
    v = np.array([point.x, point.y, 1.0], dtype="float64")
    out = matrix @ v
    if abs(out[2]) < 1e-12:
        return Point(point.x, point.y)
    return Point(float(out[0] / out[2]), float(out[1] / out[2]))


def map_rect(matrix: np.ndarray, rect: Rect) -> Rect:
    """Map a rectangle through a perspective matrix (returns bounding box)."""
    corners = [
        Point(rect.x0, rect.y0),
        Point(rect.x1, rect.y0),
        Point(rect.x1, rect.y1),
        Point(rect.x0, rect.y1),
    ]
    mapped = [map_point(matrix, c) for c in corners]
    xs = [p.x for p in mapped]
    ys = [p.y for p in mapped]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def map_quad(matrix: np.ndarray, quad: Quad) -> Quad:
    pts = [map_point(matrix, p) for p in quad.points()]
    return Quad(tl=pts[0], tr=pts[1], br=pts[2], bl=pts[3])


def scale_quad(quad: Quad, sx: float, sy: float) -> Quad:
    """Scale a quad by factors (e.g. when image is resized)."""
    return Quad(
        tl=Point(quad.tl.x * sx, quad.tl.y * sy),
        tr=Point(quad.tr.x * sx, quad.tr.y * sy),
        br=Point(quad.br.x * sx, quad.br.y * sy),
        bl=Point(quad.bl.x * sx, quad.bl.y * sy),
    )


def clamp_quad_to_image(quad: Quad, width: int, height: int) -> Quad:
    """Clamp quad corners to image bounds."""
    def cl(p: Point) -> Point:
        return Point(min(max(p.x, 0.0), float(width)), min(max(p.y, 0.0), float(height)))

    return Quad(tl=cl(quad.tl), tr=cl(quad.tr), br=cl(quad.br), bl=cl(quad.bl))


def polygon_to_min_area_rect(polygon: Sequence[Tuple[float, float]]) -> Rect:
    """Minimum-area axis-aligned bounding rect of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """Intersection of lines (p1,p2) and (p3,p4), or None if parallel."""
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return Point(px, py)
