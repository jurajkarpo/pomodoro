"""RapidOCR engine adapter.

RapidOCR runs the PaddleOCR PP-OCR models (detection +
recognition) converted to ONNX format via onnxruntime. It is a
lightweight way to use the PaddleOCR models without the full
paddlepaddle framework, and is Apache-2.0 licensed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import OCRError
from ..core.types import OCREngineName, OCRPage, Rect
from .base import OCREngine, RawLine, build_ocr_page


class RapidOCREngine(OCREngine):
    name = OCREngineName.RAPID

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._ocr = None
        self._version = "rapidocr-onnxruntime (PP-OCR ONNX models)"

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # noqa: F401

                self._available = True
            except Exception:
                self._available = False
        return self._available

    def version(self) -> str:
        try:
            import rapidocr_onnxruntime

            v = getattr(rapidocr_onnxruntime, "__version__", "")
            return f"rapidocr-onnxruntime {v} (PP-OCR ONNX)"
        except Exception:
            return self._version

    def _get_ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        return self._ocr

    def recognize(self, image: np.ndarray, language: str = "sk") -> OCRPage:
        if not self.is_available():
            raise OCRError("RapidOCR is not installed", engine="rapid")
        ocr = self._get_ocr()
        result, _ = ocr(image)
        h, w = image.shape[:2]
        raw_lines: list[RawLine] = []
        if result:
            for item in result:
                box, text, conf = item[0], item[1], item[2]
                if not text or not text.strip():
                    continue
                raw_lines.append(
                    RawLine(text=text.strip(), box=_box_to_rect(box), confidence=float(conf))
                )
        return build_ocr_page(raw_lines, w, h, engine="rapid", language=language)


def _box_to_rect(box) -> Rect:
    """Convert a 4-point box to an axis-aligned Rect."""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return Rect(x0, y0, x1 - x0, y1 - y0)
