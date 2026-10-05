"""Slovak OCR quality diagnostics.

Detects common OCR error patterns in Slovak text:
  - diacritic loss (ľ -> l, č -> c, ...)
  - mistaken punctuation
  - missing accents
  - broken hyphenation
  - incorrect line joins
  - incorrect word segmentation

The module only *detects and scores*; it never rewrites
text automatically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.types import OCRPage
from ..ocr import diacritics as dia


@dataclass
class QualityIssue:
    kind: str  # diacritic_loss, punctuation, hyphenation, line_join, segmentation
    description: str
    location: str = ""
    severity: str = "medium"  # low, medium, high
    suggestion: str = ""


@dataclass
class PageQualityReport:
    page_index: int
    average_confidence: float = 0.0
    diacritic_preservation: float = 0.0
    issue_count: int = 0
    issues: list[QualityIssue] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)


_PUNCT_ISSUES = [
    (re.compile(r"\s,"), "space before comma"),
    (re.compile(r"\s\."), "space before period"),
    (re.compile(r",,"), "double comma"),
    (re.compile(r"\.\."), "double period"),
    (re.compile(r"\s{2,}"), "multiple spaces"),
    (re.compile(r"[a-z]\.[A-Z]"), "missing space after period"),
]

_GARBAGE_RE = re.compile(
    r"[^a-zA-Z0-9\s.,;:!?\"'()\-–—/„“”‚‘’áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]"
)


def analyze_page(page: OCRPage, page_index: int = 0) -> PageQualityReport:
    """Run all Slovak quality checks on one OCRPage."""
    report = PageQualityReport(
        page_index=page_index,
        average_confidence=page.average_confidence,
    )
    text = page.raw_text

    dia_report = dia.analyze_text(text)
    report.diacritic_preservation = dia_report.preservation_score
    for sw in dia_report.suspicious_words:
        report.issues.append(QualityIssue(
            kind="diacritic_loss",
            description=f"possible lost diacritic in '{sw.word}'",
            location=sw.word,
            severity="medium",
            suggestion=sw.suggestion,
        ))

    for pattern, desc in _PUNCT_ISSUES:
        for m in pattern.finditer(text):
            report.issues.append(QualityIssue(
                kind="punctuation",
                description=desc,
                location=text[max(0, m.start() - 10):m.end() + 10].replace("\n", " "),
                severity="low",
            ))

    for m in _GARBAGE_RE.finditer(text):
        report.issues.append(QualityIssue(
            kind="garbage",
            description=f"unusual character '{m.group()}'",
            location=text[max(0, m.start() - 10):m.end() + 10].replace("\n", " "),
            severity="low",
        ))

    low_conf = page.low_confidence_words(threshold=0.5)
    for w in low_conf[:20]:
        report.issues.append(QualityIssue(
            kind="low_confidence",
            description=f"low confidence ({w.confidence:.2f}) for '{w.text}'",
            location=w.text,
            severity="medium",
        ))

    for word in page.all_words():
        if len(word.text) > 30 and " " not in word.text:
            report.issues.append(QualityIssue(
                kind="segmentation",
                description=f"suspiciously long token '{word.text[:30]}...'",
                location=word.text[:30],
                severity="low",
            ))

    report.issue_count = len(report.issues)

    reasons: list[str] = []
    if page.average_confidence < 0.6:
        reasons.append(f"low average confidence ({page.average_confidence:.2f})")
    if report.diacritic_preservation < 0.3:
        reasons.append("possible diacritic loss")
    if len(low_conf) > 5:
        reasons.append(f"{len(low_conf)} low-confidence words")
    if dia_report.suspicious_words:
        reasons.append(f"{len(dia_report.suspicious_words)} suspicious words")
    if report.issue_count > 15:
        reasons.append(f"{report.issue_count} quality issues")
    report.review_reasons = reasons
    report.needs_review = len(reasons) > 0

    return report


def build_review_queue(
    pages: list[OCRPage],
    confidence_threshold: float = 0.6,
) -> list[int]:
    """Return indices of pages that need review, prioritized
    by severity (worst first)."""
    scored: list[tuple[float, int]] = []
    for i, page in enumerate(pages):
        if page is None:
            continue
        report = analyze_page(page, i)
        priority = (
            (1.0 - page.average_confidence) * 2.0
            + (1.0 - report.diacritic_preservation)
            + min(report.issue_count / 20.0, 1.0)
        )
        if report.needs_review:
            scored.append((priority, i))
    scored.sort(reverse=True)
    return [i for _, i in scored]
