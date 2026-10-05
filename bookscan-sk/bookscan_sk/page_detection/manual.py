"""Manual crop override data.

The UI provides a four-corner draggable crop editor. This module
holds the data model and validation so the UI and pipeline share
one representation.
"""
from __future__ import annotations

from ..core.types import DetectedPage, PageLayout, Point, Quad, SourceImage


def manual_quad_from_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> Quad:
    """Build a Quad from four user-provided points (tl, tr, br, bl),
    clamped to the image bounds."""
    if len(points) != 4:
        raise ValueError("Exactly four corner points are required")

    def cl(x: float, y: float) -> Point:
        return Point(
            min(max(x, 0.0), float(width)),
            min(max(y, 0.0), float(height)),
        )

    pts = [cl(x, y) for x, y in points]
    return Quad(tl=pts[0], tr=pts[1], br=pts[2], bl=pts[3])


def apply_manual_crop(
    detected: DetectedPage,
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> DetectedPage:
    """Override automatic detection with a manual quad."""
    quad = manual_quad_from_points(points, width, height)
    return DetectedPage(
        source=detected.source,
        quad=quad,
        layout=detected.layout,
        gutter_x=detected.gutter_x,
        confidence=1.0,
        skew_degrees=detected.skew_degrees,
        perspective_strength=detected.perspective_strength,
        method="manual",
        manual=True,
    )
