"""Error hierarchy for BookScan SK.

The pipeline never fails silently: every failure is represented by a
typed exception (or a PageResult.error string) so callers can react.
"""
from __future__ import annotations


class BookScanError(Exception):
    """Base class for all BookScan SK errors."""


class ConfigurationError(BookScanError):
    """Invalid user or application configuration."""


class ImageError(BookScanError):
    """An image could not be read or decoded."""


class PageDetectionError(BookScanError):
    """Automatic page boundary detection failed."""

    def __init__(self, message: str, page_index: int = -1) -> None:
        super().__init__(message)
        self.page_index = page_index


class DewarpingError(BookScanError):
    """Dewarping could not be estimated; caller should fall back."""


class OCRError(BookScanError):
    """An OCR engine failed."""

    def __init__(self, message: str, engine: str = "") -> None:
        super().__init__(message)
        self.engine = engine


class ModelNotInstalledError(BookScanError):
    """A required model is not installed."""

    def __init__(self, model: str, hint: str = "") -> None:
        super().__init__(f"Model not installed: {model}. {hint}")
        self.model = model
        self.hint = hint


class PDFError(BookScanError):
    """PDF generation or import failed."""


class ProjectError(BookScanError):
    """Project persistence failed."""


class ExportError(BookScanError):
    """Export (TXT/ALTO/PAGE XML) failed."""
