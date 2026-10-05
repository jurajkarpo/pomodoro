"""PDF metadata handling."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..core.types import DocumentMetadata


def current_iso_date() -> str:
    """Current UTC date in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_metadata(
    title: str = "",
    author: str = "",
    subject: str = "",
    creator: str = "BookScan SK",
    ocr_language: str = "sk",
    keywords: str = "",
    first_page_number: int = 1,
    page_offset: int = 0,
) -> DocumentMetadata:
    """Build a DocumentMetadata with a creation date."""
    return DocumentMetadata(
        title=title,
        author=author,
        subject=subject,
        creator=creator,
        ocr_language=ocr_language,
        creation_date=current_iso_date(),
        keywords=keywords,
        first_page_number=first_page_number,
        page_offset=page_offset,
    )


def read_pdf_metadata(path: str) -> dict[str, str]:
    """Read metadata from an existing PDF using pikepdf."""
    import pikepdf

    out: dict[str, str] = {}
    try:
        pdf = pikepdf.open(path)
        info = pdf.docinfo
        mapping = {
            "/Title": "title",
            "/Author": "author",
            "/Subject": "subject",
            "/Creator": "creator",
            "/Producer": "producer",
            "/Keywords": "keywords",
        }
        for key, name in mapping.items():
            if key in info:
                out[name] = str(info[key])
        pdf.close()
    except Exception:
        pass
    return out
