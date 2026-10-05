"""Model manager: installation, status and checksum verification.

The manager never silently downloads models. Every download
is an explicit action with a known URL, expected checksum
and license. Status is reported clearly so the user knows
exactly what is installed.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from ..core.config import MODELS_DIR
from .registry import ALL_MODELS, ModelEntry, get_model


@dataclass
class ModelStatus:
    entry: ModelEntry
    installed: bool = False
    location: str = ""
    checksum_verified: bool = False
    installed_at: str = ""
    error: str = ""


class ModelManager:
    """Manages OCR models, traineddata and fonts."""

    def __init__(self, models_dir: Optional[str] = None) -> None:
        self.models_dir = models_dir or str(MODELS_DIR)
        os.makedirs(self.models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, model_id: str) -> ModelStatus:
        entry = get_model(model_id)
        if entry is None:
            raise ValueError(f"Unknown model: {model_id}")
        status = ModelStatus(entry=entry)
        if entry.category == "ocr" and entry.id.startswith("tesseract"):
            status.installed, status.location = self._tesseract_lang_status(
                entry.id.split(".")[-1]
            )
        elif entry.category == "ocr" and entry.id.startswith("paddle"):
            status.installed, status.location = self._paddle_status()
        elif entry.category == "font":
            status.installed, status.location = self._font_status()
        status.checksum_verified = self._verify_checksum(entry, status.location)
        return status

    def all_status(self) -> list[ModelStatus]:
        return [self.status(m.id) for m in ALL_MODELS]

    def _tesseract_lang_status(self, lang: str) -> tuple[bool, str]:
        """Check whether a tesseract language is installed."""
        try:
            out = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True, text=True, timeout=20,
            )
            if out.returncode == 0:
                langs = out.stdout.split()
                if lang in langs:
                    tessdir = self._find_tessdata()
                    return True, os.path.join(tessdir, f"{lang}.traineddata")
        except Exception:
            pass
        return False, ""

    def _paddle_status(self) -> tuple[bool, str]:
        """Check whether PaddleOCR + paddlepaddle are importable."""
        try:
            import paddle  # noqa: F401
            import paddleocr  # noqa: F401

            cache = os.path.expanduser("~/.paddlex")
            return True, cache
        except Exception:
            return False, ""

    def _font_status(self) -> tuple[bool, str]:
        """Check whether a Slovak-capable font is available."""
        from ..pdf.writer import _find_font

        try:
            path = _find_font()
            return True, path
        except Exception:
            return False, ""

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def install(self, model_id: str, progress=None) -> ModelStatus:
        """Explicitly install a model. Never silent."""
        entry = get_model(model_id)
        if entry is None:
            raise ValueError(f"Unknown model: {model_id}")

        if entry.category == "ocr" and entry.id.startswith("tesseract"):
            return self._install_tesseract_lang(entry, progress)
        if entry.category == "ocr" and entry.id.startswith("paddle"):
            return self._install_paddle(entry, progress)
        if entry.category == "font":
            return self._install_font(entry, progress)

        return ModelStatus(entry=entry, error="Unsupported model category")

    def _install_tesseract_lang(
        self, entry: ModelEntry, progress=None
    ) -> ModelStatus:
        lang = entry.id.split(".")[-1]
        status = ModelStatus(entry=entry)
        tessdir = self._find_tessdata()
        dest = os.path.join(tessdir, f"{lang}.traineddata")
        if os.path.isfile(dest):
            status.installed = True
            status.location = dest
            return status
        try:
            self._download(entry.model_url, dest, progress)
            status.installed = os.path.isfile(dest)
            status.location = dest
        except Exception as e:
            status.error = str(e)
        return status

    def _install_paddle(self, entry: ModelEntry, progress=None) -> ModelStatus:
        status = ModelStatus(entry=entry)
        try:
            from paddleocr import PaddleOCR

            if progress:
                progress("Downloading PaddleOCR models (first use)...")
            try:
                PaddleOCR(lang="sk", show_log=False)
            except TypeError:
                PaddleOCR(lang="sk")
            status.installed = True
            status.location = os.path.expanduser("~/.paddlex")
        except Exception as e:
            status.error = str(e)
        return status

    def _install_font(self, entry: ModelEntry, progress=None) -> ModelStatus:
        status = ModelStatus(entry=entry)
        installed, location = self._font_status()
        status.installed = installed
        status.location = location
        if not installed:
            status.error = (
                "No Slovak-capable font found. Install the "
                "'fonts-dejavu' package or place a TTF font."
            )
        return status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_tessdata(self) -> str:
        """Locate the tessdata directory."""
        candidates = [
            "/usr/share/tesseract-ocr/5/tessdata",
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/share/tesseract-ocr/tessdata",
            "/usr/share/tessdata",
            os.path.expanduser("~/.local/share/tessdata"),
            os.path.join(self.models_dir, "tessdata"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        td = os.path.join(self.models_dir, "tessdata")
        os.makedirs(td, exist_ok=True)
        return td

    def _verify_checksum(self, entry: ModelEntry, location: str) -> bool:
        """Verify SHA-256 checksum if we have a known value."""
        if not entry.checksum or not location or not os.path.isfile(location):
            return False
        try:
            h = hashlib.sha256()
            with open(location, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest() == entry.checksum
        except Exception:
            return False

    def _download(self, url: str, dest: str, progress=None) -> None:
        """Download a file with progress."""
        import urllib.request

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        if progress:
            progress(f"Downloading {os.path.basename(dest)}...")

        def _reporthook(count, block_size, total_size):
            if progress and total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                progress(f"  {pct}%")

        urllib.request.urlretrieve(url, tmp, _reporthook)
        shutil.move(tmp, dest)
