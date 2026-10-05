"""PaddleOCR engine adapter (primary engine).

Uses the official ``paddleocr`` package (PaddleOCR 3.x / PP-OCRv5)
with the Slovak ``sk`` language. PaddleOCR 3.x builds a PaddleX
OCR pipeline; models are downloaded on first use (see the model
manager for offline installation).

The adapter is defensive: if paddlepaddle or the models are not
available it reports unavailable so the manager can fall back to
RapidOCR or Tesseract rather than crashing the pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import OCRError
from ..core.types import OCREngineName, OCRPage
from .base import OCREngine, RawLine, build_ocr_page
from .rapid_engine import _box_to_rect


class PaddleOCREngine(OCREngine):
    name = OCREngineName.PADDLE

    def __init__(self, lang: str = "sk") -> None:
        self.lang = lang
        self._available: Optional[bool] = None
        self._ocr = None
        self._version = ""

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._detect()
        return self._available

    def _detect(self) -> bool:
        try:
            import paddle  # noqa: F401
            from paddleocr import PaddleOCR  # noqa: F401

            return True
        except Exception:
            return False

    def version(self) -> str:
        try:
            import paddle
            from paddleocr import __version__ as pv

            return f"PaddleOCR {pv} (paddle {paddle.__version__})"
        except Exception:
            return "PaddleOCR (unavailable)"

    def _get_ocr(self, language: str):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                lang=language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr

    def recognize(self, image: np.ndarray, language: str = "sk") -> OCRPage:
        if not self.is_available():
            raise OCRError("PaddleOCR is not available", engine="paddle")
        ocr = self._get_ocr(language)
        # PaddleOCR 3.x: ocr() accepts the image plus optional
        # pipeline kwargs (no 'cls' argument in v3).
        res = ocr.ocr(image)
        h, w = image.shape[:2]
        raw_lines: list[RawLine] = []
        if res:
            for page in res:
                if not page:
                    continue
                for item in page:
                    box = item[0]
                    text = item[1][0]
                    conf = float(item[1][1])
                    if not text or not text.strip():
                        continue
                    raw_lines.append(
                        RawLine(text=text.strip(), box=_box_to_rect(box), confidence=conf)
                    )
        return build_ocr_page(raw_lines, w, h, engine="paddle", language=language)
