"""OCR manager: engine selection, modes and dual validation.

Modes:
  PADDLE    - PaddleOCR (falls back to RapidOCR, then Tesseract)
  TESSERACT - Tesseract 5 + slk
  AUTO      - calibrates on the first pages, then uses the engine
              that best preserves Slovak diacritics and confidence
  DUAL      - runs PaddleOCR and Tesseract, compares, keeps both
              raw outputs and reports agreement

The manager never invents text: it only selects or combines
engine outputs and always exposes confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.errors import OCRError
from ..core.types import OCREngineChoice, OCREngineName, OCRPage
from .base import OCREngine
from .diacritics import diacritic_preservation_score
from .paddle_engine import PaddleOCREngine
from .rapid_engine import RapidOCREngine
from .tesseract_engine import TesseractEngine


@dataclass
class DualResult:
    """Outcome of dual-engine validation for one page."""

    primary: OCRPage
    secondary: OCRPage
    chosen: OCRPage
    agreement: float = 0.0
    primary_name: str = ""
    secondary_name: str = ""
    disagreements: list[str] = field(default_factory=list)


class OCRManager:
    """Owns the OCR engines and applies the chosen mode."""

    def __init__(
        self,
        tesseract_cmd: str = "tesseract",
        paddle_enabled: bool = True,
        rapid_enabled: bool = True,
        tesseract_enabled: bool = True,
    ) -> None:
        self.tesseract = TesseractEngine(cmd=tesseract_cmd)
        self.paddle = PaddleOCREngine()
        self.rapid = RapidOCREngine()
        self._paddle_enabled = paddle_enabled
        self._rapid_enabled = rapid_enabled
        self._tesseract_enabled = tesseract_enabled
        self._auto_choice: Optional[OCREngineName] = None
        self._calibration_samples = 0
        self._calibration_limit = 3

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def available_engines(self) -> list[OCREngineName]:
        out: list[OCREngineName] = []
        if self._paddle_enabled and self.paddle.is_available():
            out.append(OCREngineName.PADDLE)
        if self._rapid_enabled and self.rapid.is_available():
            out.append(OCREngineName.RAPID)
        if self._tesseract_enabled and self.tesseract.is_available():
            out.append(OCREngineName.TESSERACT)
        return out

    def primary_engine(self) -> OCREngine:
        """The preferred PaddleOCR-family engine."""
        if self._paddle_enabled and self.paddle.is_available():
            return self.paddle
        if self._rapid_enabled and self.rapid.is_available():
            return self.rapid
        if self._tesseract_enabled and self.tesseract.is_available():
            return self.tesseract
        raise OCRError("No OCR engine is available. Install PaddleOCR or Tesseract.")

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self, image: np.ndarray, language: str = "sk", mode: OCREngineChoice = OCREngineChoice.AUTO
    ) -> OCRPage:
        if mode == OCREngineChoice.TESSERACT:
            return self._run_tesseract(image, language)
        if mode == OCREngineChoice.PADDLE:
            return self._run_paddle(image, language)
        if mode == OCREngineChoice.DUAL:
            return self.recognize_dual(image, language).chosen
        return self._run_auto(image, language)

    def _run_paddle(self, image: np.ndarray, language: str) -> OCRPage:
        try:
            if self._paddle_enabled and self.paddle.is_available():
                return self.paddle.recognize(image, language)
        except OCRError:
            pass
        if self._rapid_enabled and self.rapid.is_available():
            return self.rapid.recognize(image, language)
        if self._tesseract_enabled and self.tesseract.is_available():
            return self.tesseract.recognize(image, language)
        raise OCRError("No OCR engine available")

    def _run_tesseract(self, image: np.ndarray, language: str) -> OCRPage:
        if self._tesseract_enabled and self.tesseract.is_available():
            return self.tesseract.recognize(image, language)
        return self._run_paddle(image, language)

    def _run_auto(self, image: np.ndarray, language: str) -> OCRPage:
        """Automatic engine selection with per-book calibration."""
        engines = self.available_engines()
        if not engines:
            raise OCRError("No OCR engine available")
        if len(engines) == 1:
            return self._engine_by_name(engines[0]).recognize(image, language)

        if self._auto_choice is None:
            self._calibration_samples += 1
            best_name = None
            best_score = -1.0
            results: dict[OCREngineName, OCRPage] = {}
            for name in engines:
                try:
                    page = self._engine_by_name(name).recognize(image, language)
                except OCRError:
                    continue
                results[name] = page
                score = self._quality_score(page)
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_name is None:
                raise OCRError("All OCR engines failed")
            if self._calibration_samples >= self._calibration_limit:
                self._auto_choice = best_name
            return results[best_name]

        return self._engine_by_name(self._auto_choice).recognize(image, language)

    def _quality_score(self, page: OCRPage) -> float:
        """Score an OCRPage: confidence + diacritic preservation."""
        conf = page.average_confidence
        diac = diacritic_preservation_score(page.raw_text)
        return 0.6 * conf + 0.4 * diac

    def _engine_by_name(self, name: OCREngineName) -> OCREngine:
        return {
            OCREngineName.PADDLE: self.paddle,
            OCREngineName.RAPID: self.rapid,
            OCREngineName.TESSERACT: self.tesseract,
        }[name]

    # ------------------------------------------------------------------
    # Dual validation
    # ------------------------------------------------------------------

    def recognize_dual(self, image: np.ndarray, language: str = "sk") -> DualResult:
        """Run PaddleOCR-family and Tesseract, compare, choose best."""
        primary = self._run_paddle(image, language)
        secondary = self._run_tesseract(image, language)

        agreement = self._agreement(primary, secondary)
        disagreements = self._find_disagreements(primary, secondary)

        primary_score = self._quality_score(primary)
        secondary_score = self._quality_score(secondary)
        chosen = primary if primary_score >= secondary_score else secondary

        chosen.secondary_text = secondary.raw_text
        chosen.agreement_score = agreement

        return DualResult(
            primary=primary,
            secondary=secondary,
            chosen=chosen,
            agreement=agreement,
            primary_name=primary.engine,
            secondary_name=secondary.engine,
            disagreements=disagreements,
        )

    def _agreement(self, a: OCRPage, b: OCRPage) -> float:
        """Word-level agreement between two OCR pages (0..1)."""
        words_a = {w.text.lower() for w in a.all_words() if w.text.strip()}
        words_b = {w.text.lower() for w in b.all_words() if w.text.strip()}
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _find_disagreements(self, a: OCRPage, b: OCRPage) -> list[str]:
        """Return words where the two engines disagree."""
        words_a = {w.text.lower() for w in a.all_words() if w.text.strip()}
        words_b = {w.text.lower() for w in b.all_words() if w.text.strip()}
        only_a = words_a - words_b
        only_b = words_b - words_a
        return sorted(only_a | only_b)

    def reset_calibration(self) -> None:
        self._auto_choice = None
        self._calibration_samples = 0
