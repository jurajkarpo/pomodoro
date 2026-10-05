"""Searchable PDF generation.

Each PDF page contains:
  1. the processed page image (visually faithful)
  2. an invisible OCR text layer (render mode 3) positioned
     accurately over the corresponding text

The text layer is selectable, searchable and extractable with
standard PDF tools, while the page still looks like the
photograph. Unicode (Slovak diacritics) is preserved via an
embedded TrueType font.

Generation uses reportlab (BSD-licensed) for precise image and
invisible-text placement with embedded fonts; pikepdf
(Apache-2.0) is then used for metadata and optimization.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
from PIL import Image

from ..core.errors import PDFError
from ..core.types import DocumentMetadata, ExportOptions, OCRPage, OCRWord, Rect


def _find_font() -> str:
    """Locate a TrueType font with full Slovak diacritic support."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for pattern in (
        "/usr/share/fonts/**/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
    ):
        candidates.extend(glob.glob(pattern, recursive=True))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise PDFError(
        "No TrueType font with Slovak support found. "
        "Install DejaVu Sans or place a TTF font and configure it."
    )


class SearchablePDFWriter:
    """Incrementally builds a searchable PDF from page images + OCR."""

    def __init__(self, dpi: int = 300, jpeg_quality: int = 92) -> None:
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality
        self._font_path = _find_font()
        self._font_name = "BookScanSans"
        self._registered = False
        self._canvas = None
        self._page_count = 0
        self._metadata = DocumentMetadata()

    def _ensure_font(self) -> None:
        if self._registered:
            return
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        pdfmetrics.registerFont(TTFont(self._font_name, self._font_path))
        self._registered = True

    def set_metadata(self, metadata: DocumentMetadata) -> None:
        self._metadata = metadata

    def start(self, path: str) -> None:
        """Start writing to a path."""
        from reportlab.pdfgen import canvas as rl_canvas

        self._canvas = rl_canvas.Canvas(path)
        self._page_count = 0

    def add_page(
        self,
        image: np.ndarray,
        ocr: Optional[OCRPage] = None,
        page_number: Optional[int] = None,
    ) -> None:
        """Add one page: the image plus its invisible OCR text layer."""
        from reportlab.lib.utils import ImageReader

        self._ensure_font()
        canvas = self._canvas
        if canvas is None:
            raise PDFError("Writer not started; call start() first.")

        h, w = image.shape[:2]
        scale = 72.0 / self.dpi
        page_w = w * scale
        page_h = h * scale
        canvas.setPageSize((page_w, page_h))

        img = Image.fromarray(image)
        img_reader = ImageReader(img)
        canvas.drawImage(img_reader, 0, 0, width=page_w, height=page_h)

        if ocr is not None:
            self._draw_invisible_text(canvas, ocr, scale, page_h)

        canvas.showPage()
        self._page_count += 1

    def _draw_invisible_text(
        self, canvas, ocr: OCRPage, scale: float, page_h: float
    ) -> None:
        """Draw the invisible text layer with accurate coordinates."""
        for block in ocr.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.ignored or not word.text.strip():
                        continue
                    text = word.display_text
                    box = word.box
                    if box is None:
                        continue
                    x = box.x0 * scale
                    y = page_h - (box.y0 + box.height) * scale
                    font_size = max(1.0, box.height * scale)
                    font_size = min(200.0, max(1.0, font_size))
                    t = canvas.beginText(x, y)
                    t.setFont(self._font_name, font_size)
                    t.setTextRenderMode(3)  # invisible
                    t.textOut(text)
                    canvas.drawText(t)

    def save(self, path: str, options: Optional[ExportOptions] = None) -> str:
        """Finalize and save the PDF, then apply metadata/optimization."""
        if self._canvas is None:
            raise PDFError("Nothing to save; add pages first.")
        md = self._metadata
        self._canvas.setTitle(md.title or "BookScan SK")
        self._canvas.setAuthor(md.author)
        self._canvas.setSubject(md.subject)
        self._canvas.setCreator(md.creator)
        self._canvas.setKeywords(md.keywords)
        self._canvas.save()
        self._canvas = None
        return self._postprocess(path, options)

    def _postprocess(self, path: str, options: Optional[ExportOptions]) -> str:
        """Apply metadata and linearization via pikepdf."""
        import pikepdf

        md = self._metadata
        try:
            pdf = pikepdf.open(path)
            with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
                if md.title:
                    meta["dc:title"] = md.title
                if md.author:
                    meta["dc:creator"] = [md.author]
                if md.subject:
                    meta["dc:description"] = md.subject
                if md.creation_date:
                    meta["dc:date"] = md.creation_date
                meta["pdf:Producer"] = "BookScan SK"
            pdf.save(path, linearize=True, compress_streams=True)
            pdf.close()
        except Exception:
            pass
        return path

    @property
    def page_count(self) -> int:
        return self._page_count


def build_searchable_pdf(
    pages: list[tuple[np.ndarray, Optional[OCRPage]]],
    output_path: str,
    metadata: Optional[DocumentMetadata] = None,
    dpi: int = 300,
    jpeg_quality: int = 92,
    options: Optional[ExportOptions] = None,
) -> str:
    """Convenience: build a complete searchable PDF from a list of
    (image, ocr_page) tuples."""
    writer = SearchablePDFWriter(dpi=dpi, jpeg_quality=jpeg_quality)
    if metadata is not None:
        writer.set_metadata(metadata)
    writer.start(output_path)
    for image, ocr in pages:
        writer.add_page(image, ocr)
    return writer.save(output_path, options)
