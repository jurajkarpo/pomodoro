"""PDF import: render an existing PDF's pages to images.

Uses pypdfium2 (PDFium, BSD-licensed) to render each page to a
high-quality RGB image so a scanned/photographed PDF can be
reprocessed through the same pipeline.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..core.errors import PDFError
from ..core.types import ImageFormat, SourceImage


def pdf_page_count(path: str) -> int:
    """Return the number of pages in a PDF."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        n = len(pdf)
        pdf.close()
        return n
    except Exception as e:
        raise PDFError(f"Cannot read PDF {path}: {e}")


def render_pdf_page(
    path: str,
    page_index: int,
    dpi: int = 300,
) -> np.ndarray:
    """Render one PDF page (0-based) to an RGB numpy array."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        page = pdf[page_index]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        pdf.close()
        return np.ascontiguousarray(arr)
    except Exception as e:
        raise PDFError(f"Cannot render page {page_index} of {path}: {e}")


def import_pdf_to_images(
    path: str,
    output_dir: str,
    dpi: int = 300,
    prefix: str = "page",
) -> list[str]:
    """Render every page of a PDF to image files in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    n = pdf_page_count(path)
    paths: list[str] = []
    from PIL import Image

    for i in range(n):
        arr = render_pdf_page(path, i, dpi=dpi)
        out = os.path.join(output_dir, f"{prefix}_{i + 1:04d}.png")
        Image.fromarray(arr).save(out)
        paths.append(out)
    return paths


def pdf_to_source_images(
    path: str,
    output_dir: str,
    dpi: int = 300,
) -> list[SourceImage]:
    """Import a PDF as a list of SourceImage references."""
    from ..utils.images import image_size, sha256_file
    from ..utils.sorting import sort_image_paths

    image_paths = import_pdf_to_images(path, output_dir, dpi=dpi)
    sources: list[SourceImage] = []
    for idx, p in enumerate(sort_image_paths(image_paths)):
        w, h = image_size(p)
        sources.append(
            SourceImage(
                path=p,
                index=idx,
                filename=os.path.basename(p),
                format=ImageFormat.PNG,
                width=w,
                height=h,
                sha256=sha256_file(p),
            )
        )
    return sources
