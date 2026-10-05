"""OCR engine abstraction and output normalization.

Every OCR engine adapter produces a common intermediate
representation (RawOCRLine) which is then normalized into the
typed OCRPage / OCRBlock / OCRLine / OCRWord structures.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.types import OCREngineName, OCRBlock, OCRLine, OCRPage, OCRWord, Rect


@dataclass
class RawWord:
    text: str
    box: Rect
    confidence: float = 0.0


@dataclass
class RawLine:
    """A raw detected text line from any engine."""

    text: str
    box: Rect
    confidence: float = 0.0
    words: list[RawWord] = field(default_factory=list)
    block_id: int = 0
    line_id: int = 0


class OCREngine(ABC):
    """Interface every OCR engine adapter implements."""

    name: OCREngineName

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the engine can run in this environment."""

    @abstractmethod
    def recognize(self, image: np.ndarray, language: str = "sk") -> OCRPage:
        """Run OCR on an RGB image and return a structured OCRPage."""

    @abstractmethod
    def version(self) -> str:
        """Human-readable engine/version string."""


def _words_from_line(line: RawLine) -> list[OCRWord]:
    """Convert a raw line into OCRWord objects."""
    if line.words:
        return [OCRWord(text=w.text, box=w.box, confidence=w.confidence) for w in line.words]
    text = line.text
    if not text.strip():
        return []
    parts = text.split(" ")
    total_chars = sum(len(p) for p in parts) or 1
    words: list[OCRWord] = []
    x = line.box.x0
    width = line.box.width
    for part in parts:
        w = max(1.0, width * (len(part) / total_chars))
        words.append(OCRWord(text=part, box=Rect(x, line.box.y0, w, line.box.height), confidence=line.confidence))
        x += w + (width * (1.0 / total_chars))
    return words


def build_ocr_page(
    lines: list[RawLine],
    width: int,
    height: int,
    engine: str,
    language: str = "sk",
) -> OCRPage:
    """Normalize raw engine output into a structured OCRPage."""
    blocks_map: dict[int, list[RawLine]] = {}
    use_blocks = any(l.block_id for l in lines)
    if use_blocks:
        for line in lines:
            blocks_map.setdefault(line.block_id, []).append(line)
    else:
        blocks_map[0] = list(lines)

    blocks: list[OCRBlock] = []
    for block_id in sorted(blocks_map.keys()):
        raw_lines = sorted(blocks_map[block_id], key=lambda l: (l.box.y0, l.box.x0))
        ocr_lines: list[OCRLine] = []
        for rl in raw_lines:
            words = _words_from_line(rl)
            ocr_lines.append(OCRLine(words=words, box=rl.box, confidence=rl.confidence))
        box = _union_boxes([l.box for l in ocr_lines if l.box])
        block_conf = (sum(l.confidence for l in ocr_lines) / len(ocr_lines)) if ocr_lines else 0.0
        blocks.append(OCRBlock(lines=ocr_lines, box=box, block_type="text", confidence=block_conf))

    avg = (sum(b.confidence for b in blocks) / len(blocks)) if blocks else 0.0
    full_text = "\n\n".join(b.text for b in blocks)
    return OCRPage(
        blocks=blocks,
        width=width,
        height=height,
        engine=engine,
        language=language,
        average_confidence=avg,
        raw_text=full_text,
    )


def _union_boxes(boxes: list[Rect]) -> Optional[Rect]:
    if not boxes:
        return None
    x0 = min(b.x0 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    return Rect(x0, y0, x1 - x0, y1 - y0)
