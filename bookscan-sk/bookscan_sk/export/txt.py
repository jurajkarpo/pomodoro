"""TXT export."""
from __future__ import annotations

from ..core.types import OCRPage, PageResult


def export_txt(
    pages: list[PageResult],
    output_path: str,
    first_page_number: int = 1,
    page_separator: bool = True,
) -> str:
    """Export OCR results to a plain text file.

    Pages are separated by form feeds so the text can be
    re-associated with page numbers.
    """
    lines: list[str] = []
    for i, page in enumerate(pages):
        if page.ocr is None:
            continue
        page_no = first_page_number + i
        if page_separator:
            lines.append(f"\f\n=== Page {page_no} ===\n")
        lines.append(page.ocr.full_text.strip())
        lines.append("")
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def ocr_page_to_txt(page: OCRPage) -> str:
    """Convert a single OCRPage to text."""
    return page.full_text
