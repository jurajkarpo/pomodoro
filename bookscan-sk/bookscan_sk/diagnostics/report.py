"""Final processing report."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..core.types import PageResult, ProcessingResult


@dataclass
class ProcessingReport:
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    skipped_pages: int = 0
    average_confidence: float = 0.0
    total_processing_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    page_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_pages": self.total_pages,
            "successful_pages": self.successful_pages,
            "failed_pages": self.failed_pages,
            "skipped_pages": self.skipped_pages,
            "average_confidence": round(self.average_confidence, 4),
            "total_processing_ms": round(self.total_processing_ms, 1),
            "errors": self.errors,
            "warnings": self.warnings,
            "page_details": self.page_details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def summary(self) -> str:
        lines = [
            "BookScan SK - Processing Report",
            "=" * 40,
            f"Total pages:      {self.total_pages}",
            f"Successful:       {self.successful_pages}",
            f"Failed:           {self.failed_pages}",
            f"Skipped:          {self.skipped_pages}",
            f"Avg confidence:   {self.average_confidence:.3f}",
            f"Processing time:  {self.total_processing_ms / 1000.0:.1f}s",
        ]
        if self.errors:
            lines.append(f"Errors:           {len(self.errors)}")
        if self.warnings:
            lines.append(f"Warnings:         {len(self.warnings)}")
        return "\n".join(lines)


def build_report(result: ProcessingResult) -> ProcessingReport:
    """Build a ProcessingReport from a ProcessingResult."""
    report = ProcessingReport()
    report.total_pages = len(result.pages)
    report.successful_pages = result.successful_pages
    report.failed_pages = result.failed_pages
    report.skipped_pages = sum(1 for p in result.pages if p.status.value == "skipped")
    report.average_confidence = result.average_confidence
    report.total_processing_ms = sum(p.processing_ms for p in result.pages)
    report.errors = list(result.errors)
    report.warnings = list(result.warnings)

    for p in result.pages:
        report.page_details.append({
            "page": p.page_index,
            "status": p.status.value,
            "confidence": (round(p.ocr.average_confidence, 4) if p.ocr else None),
            "engine": p.ocr.engine if p.ocr else None,
            "processing_ms": round(p.processing_ms, 1),
            "error": p.error,
            "warnings": p.warnings,
        })

    return report


def save_report(report: ProcessingReport, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    return path
