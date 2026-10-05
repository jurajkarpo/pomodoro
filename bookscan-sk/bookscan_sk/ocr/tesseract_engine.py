"""Tesseract OCR engine adapter (secondary engine / fallback).

Uses pytesseract with the Slovak "slk" traineddata. Produces
word-level output with block/paragraph/line structure via the
TSV interface.
"""
from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np

from ..core.errors import OCRError
from ..core.types import OCREngineName, OCRPage, Rect
from .base import OCREngine, RawLine, RawWord, build_ocr_page


class TesseractEngine(OCREngine):
    name = OCREngineName.TESSERACT

    def __init__(self, cmd: str = "tesseract") -> None:
        self.cmd = cmd
        self._available: Optional[bool] = None
        self._version = ""

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._detect()
        return self._available

    def _detect(self) -> bool:
        try:
            out = subprocess.run(
                [self.cmd, "--version"],
                capture_output=True, text=True, timeout=20,
            )
            if out.returncode == 0:
                self._version = out.stdout.splitlines()[0] if out.stdout else ""
                return True
        except Exception:
            pass
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def version(self) -> str:
        if not self._version:
            try:
                import pytesseract

                self._version = f"tesseract {pytesseract.get_tesseract_version()}"
            except Exception:
                self._version = "tesseract (unknown)"
        return self._version

    def has_language(self, lang: str) -> bool:
        """Check whether a traineddata language is installed."""
        try:
            out = subprocess.run(
                [self.cmd, "--list-langs"],
                capture_output=True, text=True, timeout=20,
            )
            if out.returncode == 0:
                return lang in out.stdout.split()
        except Exception:
            pass
        return False

    def recognize(self, image: np.ndarray, language: str = "sk") -> OCRPage:
        if not self.is_available():
            raise OCRError("Tesseract is not installed", engine="tesseract")
        import pytesseract
        from PIL import Image

        tess_lang = _to_tesseract_lang(language)
        if not self.has_language(tess_lang):
            tess_lang = "eng"

        pil_img = Image.fromarray(image)
        data = pytesseract.image_to_data(
            pil_img, lang=tess_lang, output_type=pytesseract.Output.DICT
        )
        h, w = image.shape[:2]

        lines_map: dict[tuple[int, int, int], dict] = {}
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            conf = float(data["conf"][i] or 0.0)
            if conf < 0:
                conf = 0.0
            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])
            block = int(data["block_num"][i])
            par = int(data["par_num"][i])
            line = int(data["line_num"][i])
            key = (block, par, line)
            if key not in lines_map:
                lines_map[key] = {"block": block, "words": []}
            lines_map[key]["words"].append(
                RawWord(text=text, box=Rect(left, top, width, height), confidence=conf / 100.0)
            )

        raw_lines: list[RawLine] = []
        for key in sorted(lines_map.keys()):
            entry = lines_map[key]
            words = entry["words"]
            if not words:
                continue
            text = " ".join(w.text for w in words)
            box = _union([w.box for w in words])
            conf = sum(w.confidence for w in words) / len(words)
            raw_lines.append(
                RawLine(text=text, box=box, confidence=conf, words=words,
                        block_id=entry["block"], line_id=key[2])
            )

        return build_ocr_page(raw_lines, w, h, engine="tesseract", language=language)


def _union(boxes: list[Rect]) -> Rect:
    x0 = min(b.x0 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def _to_tesseract_lang(lang: str) -> str:
    """Map internal language codes to Tesseract codes."""
    mapping = {
        "sk": "slk", "slk": "slk", "slovak": "slk",
        "en": "eng", "eng": "eng", "cs": "ces",
    }
    return mapping.get(lang.lower(), lang.lower())
