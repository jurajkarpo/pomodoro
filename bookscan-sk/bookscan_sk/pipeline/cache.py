"""Page cache.

Avoids recomputing a page when its source image and settings
have not changed. Cache keys are derived from the source
SHA-256 and a settings fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from ..core.config import CACHE_DIR
from ..core.types import ProcessingSettings


def settings_fingerprint(settings: ProcessingSettings) -> str:
    """A stable hash of the processing settings."""
    payload = json.dumps(
        {
            "preset": settings.preset.value,
            "dewarp": settings.dewarp.value,
            "ocr_engine": settings.ocr_engine.value,
            "language": settings.language,
            "dpi": settings.dpi,
            "jpeg_quality": settings.jpeg_quality,
            "rotate": settings.rotate_degrees,
            "contrast": settings.contrast,
            "sharpen": settings.sharpen,
            "denoise": settings.denoise,
            "adaptive_threshold": settings.adaptive_threshold,
            "grayscale": settings.grayscale,
            "black_white": settings.black_white,
            "shadow": settings.shadow_reduction,
            "illumination": settings.illumination_norm,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def cache_key(source_sha256: str, settings: ProcessingSettings) -> str:
    return f"{source_sha256}-{settings_fingerprint(settings)}"


class PageCache:
    """Filesystem cache for processed page images and OCR data."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root or str(CACHE_DIR)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str, suffix: str) -> str:
        return os.path.join(self.root, f"{key}{suffix}")

    def has_processed_image(self, key: str) -> bool:
        return os.path.isfile(self._path(key, ".jpg"))

    def get_processed_image(self, key: str) -> Optional[str]:
        p = self._path(key, ".jpg")
        return p if os.path.isfile(p) else None

    def store_processed_image(self, key: str, image_path: str) -> None:
        import shutil

        dest = self._path(key, ".jpg")
        if os.path.abspath(image_path) != os.path.abspath(dest):
            shutil.copyfile(image_path, dest)

    def has_ocr(self, key: str) -> bool:
        return os.path.isfile(self._path(key, ".ocr.json"))

    def get_ocr(self, key: str) -> Optional[str]:
        p = self._path(key, ".ocr.json")
        return p if os.path.isfile(p) else None

    def store_ocr(self, key: str, ocr_json_path: str) -> None:
        import shutil

        dest = self._path(key, ".ocr.json")
        if os.path.abspath(ocr_json_path) != os.path.abspath(dest):
            shutil.copyfile(ocr_json_path, dest)

    def invalidate(self, key: str) -> None:
        for suffix in (".jpg", ".ocr.json"):
            p = self._path(key, suffix)
            if os.path.isfile(p):
                os.remove(p)
