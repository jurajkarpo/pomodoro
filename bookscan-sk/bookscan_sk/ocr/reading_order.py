"""Reading order determination.

Determines the correct reading order for:
  - single-column books
  - two-column books
  - headings, footnotes, page numbers, captions

The default for a Slovak book is top-to-bottom, left-to-right
within each logical region. Unrelated columns are never merged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.types import OCRBlock, OCRLine, OCRPage, Rect


@dataclass
class ReadingOrderResult:
    page: OCRPage
    column_count: int = 1
    column_boundaries: Optional[list[float]] = None

    def __post_init__(self) -> None:
        if self.column_boundaries is None:
            self.column_boundaries = []


def detect_columns(lines: list[OCRLine], page_width: int) -> tuple[int, list[float]]:
    """Detect the number of text columns and their boundaries."""
    if not lines:
        return 1, []
    occupancy = np.zeros(page_width, dtype=np.int32)
    for line in lines:
        box = line.box
        if box is None:
            continue
        x0 = max(0, int(box.x0))
        x1 = min(page_width, int(box.x1))
        occupancy[x0:x1] += 1

    lo = int(page_width * 0.2)
    hi = int(page_width * 0.8)
    if hi <= lo:
        return 1, []
    min_gap = max(20, int(page_width * 0.04))
    best_gap_start = -1
    best_gap_len = 0
    run_start = -1
    run_len = 0
    for x in range(lo, hi):
        if occupancy[x] == 0:
            if run_start == -1:
                run_start = x
            run_len += 1
        else:
            if run_len > best_gap_len:
                best_gap_len = run_len
                best_gap_start = run_start
            run_start = -1
            run_len = 0
    if run_len > best_gap_len:
        best_gap_len = run_len
        best_gap_start = run_start

    if best_gap_len >= min_gap and best_gap_start >= 0:
        boundary = float(best_gap_start + best_gap_len / 2.0)
        return 2, [boundary]
    return 1, []


def assign_column(line: OCRLine, boundaries: list[float]) -> int:
    """Assign a line to a column index based on its center x."""
    if not boundaries or line.box is None:
        return 0
    cx = line.box.x0 + line.box.width / 2.0
    col = 0
    for b in boundaries:
        if cx >= b:
            col += 1
    return col


def classify_block(block: OCRBlock, page_height: int) -> str:
    """Heuristically classify a block as heading, footnote,
    page_number, or text."""
    if not block.lines:
        return "text"
    box = block.box
    if box is None:
        return "text"
    # Page number: small block near the bottom centre.
    if box.height < page_height * 0.04 and box.y0 > page_height * 0.9:
        text = block.text.strip()
        if len(text) <= 6 and text.replace(".", "").isdigit():
            return "page_number"
    # Footnote: block in the bottom 15% with small text.
    if box.y0 > page_height * 0.85:
        line_heights = [l.box.height for l in block.lines if l.box]
        if line_heights:
            avg_h = float(np.mean(line_heights))
            if avg_h < page_height * 0.035:
                return "footnote"
    # Heading: block with notably larger text than the page average.
    line_heights = [l.box.height for l in block.lines if l.box]
    if line_heights:
        avg = float(np.mean(line_heights))
        if avg > page_height * 0.05 and len(block.lines) <= 3:
            return "heading"
    return "text"


def apply_reading_order(page: OCRPage) -> ReadingOrderResult:
    """Reorder the blocks/lines of a page into reading order."""
    all_lines: list[OCRLine] = []
    for block in page.blocks:
        for line in block.lines:
            all_lines.append(line)

    column_count, boundaries = detect_columns(all_lines, page.width)

    for block in page.blocks:
        block.block_type = classify_block(block, page.height)

    body_blocks = []
    end_blocks = []
    for block in page.blocks:
        if block.block_type in ("page_number", "footnote"):
            end_blocks.append(block)
        else:
            body_blocks.append(block)

    if column_count == 1:
        body_blocks.sort(key=lambda b: (b.box.y0 if b.box else 0, b.box.x0 if b.box else 0))
    else:
        def sort_key(b: OCRBlock):
            box = b.box
            if box is None:
                return (0, 0, 0)
            cx = box.x0 + box.width / 2.0
            col = 0
            for boundary in boundaries:
                if cx >= boundary:
                    col += 1
            return (col, box.y0, box.x0)

        body_blocks.sort(key=sort_key)

    ordered = body_blocks + end_blocks
    for i, block in enumerate(ordered):
        block.reading_order = i

    page.blocks = ordered
    page.reading_order_applied = True
    return ReadingOrderResult(
        page=page,
        column_count=column_count,
        column_boundaries=boundaries,
    )


def reading_order_preview(page: OCRPage) -> list[str]:
    """Return a human-readable preview of the reading order."""
    out: list[str] = []
    for block in page.blocks:
        label = block.block_type
        box = block.box
        pos = f"({box.x0:.0f},{box.y0:.0f})" if box else "(?)"
        preview = block.text.strip().replace("\n", " ")[:60]
        out.append(f"[{block.reading_order}] {label} {pos}: {preview}")
    return out
