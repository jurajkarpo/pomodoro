"""Command-line interface for BookScan SK.

Usage:
    bookscan-sk input_folder output.pdf --lang sk
    bookscan-sk ./photos output.pdf --ocr dual
    bookscan-sk ./photos output.pdf --dewarp auto --preset ocr

The CLI exposes the full processing pipeline so it can be
automated without the GUI.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from .core.config import APP_NAME, APP_VERSION, ensure_dirs
from .core.types import (
    DewarpStrength,
    ExportOptions,
    HardwareMode,
    OCREngineChoice,
    PreprocessPreset,
    ProcessingSettings,
    SourceImage,
)
from .diagnostics.report import build_report, save_report
from .export.alto import export_alto
from .export.page_xml import export_page_xml
from .export.txt import export_txt
from .models.manager import ModelManager
from .pdf.importer import pdf_to_source_images
from .pdf.metadata import build_metadata
from .pdf.writer import SearchablePDFWriter
from .pipeline.runner import RunnerConfig, PipelineRunner
from .storage.project import create_project
from .utils.images import image_size, sha256_file
from .utils.sorting import list_image_files, sort_image_paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bookscan-sk",
        description=(
            "BookScan SK - convert photos of Slovak book pages "
            "into a searchable PDF with accurate Slovak OCR."
        ),
    )
    p.add_argument("input", help="Input folder of images, image file, or PDF")
    p.add_argument("output", help="Output PDF path")
    p.add_argument(
        "--lang", default="sk",
        help="OCR language code (default: sk)",
    )
    p.add_argument(
        "--ocr",
        choices=["paddle", "tesseract", "auto", "dual"],
        default="auto",
        help="OCR engine mode (default: auto)",
    )
    p.add_argument(
        "--dewarp",
        choices=["off", "low", "medium", "high", "auto"],
        default="auto",
        help="Dewarping strength (default: auto)",
    )
    p.add_argument(
        "--preset",
        choices=["original", "natural", "document", "ocr", "black_white"],
        default="natural",
        help="Preprocessing preset (default: natural)",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers (0 = auto)",
    )
    p.add_argument(
        "--parallel", action="store_true",
        help="Enable multiprocessing worker pool",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300)",
    )
    p.add_argument(
        "--jpeg-quality", type=int, default=92,
        help="JPEG quality for embedded images (default: 92)",
    )
    p.add_argument(
        "--first-page", type=int, default=1,
        help="First PDF page number (default: 1)",
    )
    p.add_argument(
        "--page-offset", type=int, default=0,
        help="Page number offset (default: 0)",
    )
    p.add_argument("--title", default="", help="PDF title metadata")
    p.add_argument("--author", default="", help="PDF author metadata")
    p.add_argument("--subject", default="", help="PDF subject metadata")
    p.add_argument(
        "--export-txt", action="store_true",
        help="Also export a plain-text (.txt) file",
    )
    p.add_argument(
        "--export-alto", action="store_true",
        help="Also export ALTO XML",
    )
    p.add_argument(
        "--export-page-xml", action="store_true",
        help="Also export PAGE XML",
    )
    p.add_argument(
        "--import-pdf", action="store_true",
        help="Treat the input as a PDF and render its pages",
    )
    p.add_argument(
        "--hardware",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Hardware mode (default: auto)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted job from the last completed page",
    )
    p.add_argument(
        "--no-dewarp", action="store_true",
        help="Disable dewarping (shorthand for --dewarp off)",
    )
    p.add_argument(
        "--project-dir", default="",
        help="Directory for processed images and job state",
    )
    p.add_argument(
        "--report", default="",
        help="Write a JSON processing report to this path",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Verbose progress output",
    )
    p.add_argument(
        "--version", action="version",
        version=f"{APP_NAME} {APP_VERSION}",
    )
    p.add_argument(
        "--models", action="store_true",
        help="Show model installation status and exit",
    )
    p.add_argument(
        "--install-model", default="",
        help="Install a model by id (e.g. tesseract.slk) and exit",
    )
    return p


def _preset_from_name(name: str) -> PreprocessPreset:
    return PreprocessPreset(name)


def _dewarp_from_name(name: str) -> DewarpStrength:
    return DewarpStrength(name)


def _ocr_from_name(name: str) -> OCREngineChoice:
    return OCREngineChoice(name)


def _collect_sources(
    input_path: str,
    import_pdf: bool,
    project_dir: str,
) -> list[SourceImage]:
    """Collect source images from a folder, file, or PDF."""
    if import_pdf or input_path.lower().endswith(".pdf"):
        render_dir = os.path.join(project_dir, "imported")
        return pdf_to_source_images(input_path, render_dir, dpi=300)

    if os.path.isfile(input_path):
        paths = [input_path]
    elif os.path.isdir(input_path):
        paths = list_image_files(input_path)
    else:
        raise SystemExit(f"Input not found: {input_path}")

    if not paths:
        raise SystemExit(f"No supported images found in: {input_path}")

    sources: list[SourceImage] = []
    for idx, p in enumerate(sort_image_paths(paths)):
        w, h = image_size(p)
        sources.append(
            SourceImage(
                path=p,
                index=idx,
                filename=os.path.basename(p),
                width=w,
                height=h,
                sha256=sha256_file(p),
            )
        )
    return sources


def show_model_status() -> int:
    mm = ModelManager()
    print(f"{APP_NAME} - Model Status")
    print("=" * 50)
    for status in mm.all_status():
        e = status.entry
        state = "INSTALLED" if status.installed else "NOT INSTALLED"
        print(f"\n[{state}] {e.name}")
        print(f"  id:        {e.id}")
        print(f"  version:   {e.version}")
        print(f"  license:   {e.license}")
        print(f"  source:    {e.source_url}")
        if status.location:
            print(f"  location:  {status.location}")
        if not status.installed:
            print(f"  hint:      {e.install_hint}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)

    # Model management commands.
    if args.models:
        return show_model_status()
    if args.install_model:
        mm = ModelManager()
        print(f"Installing model: {args.install_model}")
        status = mm.install(args.install_model)
        if status.installed:
            print(f"Installed: {status.location}")
            return 0
        print(f"Failed: {status.error}")
        return 1

    # Resolve dewarp (no-dewarp shorthand).
    dewarp_name = "off" if args.no_dewarp else args.dewarp

    settings = ProcessingSettings(
        preset=_preset_from_name(args.preset),
        dewarp=_dewarp_from_name(dewarp_name),
        ocr_engine=_ocr_from_name(args.ocr),
        language=args.lang,
        hardware=HardwareMode(args.hardware),
        workers=args.workers,
        dpi=args.dpi,
        jpeg_quality=args.jpeg_quality,
        first_page_number=args.first_page,
        page_offset=args.page_offset,
    )

    # Project / working directory.
    if args.project_dir:
        project_dir = args.project_dir
    else:
        base = os.path.dirname(os.path.abspath(args.output))
        project_dir = os.path.join(base, ".bookscan_work")
    os.makedirs(project_dir, exist_ok=True)

    # Collect sources.
    if args.verbose:
        print(f"Collecting sources from: {args.input}")
    sources = _collect_sources(args.input, args.import_pdf, project_dir)
    if args.verbose:
        print(f"Found {len(sources)} pages")

    # Run the pipeline.
    processed_dir = os.path.join(project_dir, "processed")
    state_path = os.path.join(project_dir, ".jobstate.json")
    config = RunnerConfig(
        settings=settings,
        output_dir=processed_dir,
        workers=args.workers,
        use_multiprocessing=args.parallel,
        state_path=state_path,
    )
    runner = PipelineRunner(config)

    events: list[str] = []

    def on_event(event):
        if args.verbose or event.kind in ("error", "warning", "done"):
            print(f"[{event.kind}] {event.message}")

    t0 = time.time()
    result = runner.run(sources, on_event=on_event)
    elapsed = time.time() - t0

    # Build the searchable PDF.
    if args.verbose:
        print("Generating searchable PDF...")
    metadata = build_metadata(
        title=args.title,
        author=args.author,
        subject=args.subject,
        ocr_language=args.lang,
        first_page_number=args.first_page,
        page_offset=args.page_offset,
    )
    writer = SearchablePDFWriter(dpi=args.dpi, jpeg_quality=args.jpeg_quality)
    writer.set_metadata(metadata)
    writer.start(args.output)
    for page in result.pages:
        if page.status.value != "done":
            continue
        image_path = page.processed.image_path
        if not image_path or not os.path.isfile(image_path):
            continue
        from .utils.images import read_image

        image = read_image(image_path)
        writer.add_page(image, page.ocr, page_number=page.page_index)
    writer.save(args.output)

    # Exports.
    base = os.path.splitext(args.output)[0]
    if args.export_txt:
        txt_path = base + ".txt"
        export_txt(result.pages, txt_path, first_page_number=args.first_page)
        if args.verbose:
            print(f"Exported TXT: {txt_path}")
    if args.export_alto:
        alto_path = base + ".alto.xml"
        export_alto(result.pages, alto_path, first_page_number=args.first_page)
        if args.verbose:
            print(f"Exported ALTO: {alto_path}")
    if args.export_page_xml:
        page_path = base + ".page.xml"
        export_page_xml(result.pages, page_path, first_page_number=args.first_page)
        if args.verbose:
            print(f"Exported PAGE XML: {page_path}")

    # Report.
    report = build_report(result)
    print(report.summary())
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output: {args.output}")
    if args.report:
        save_report(report, args.report)
        print(f"Report: {args.report}")

    return 0 if report.failed_pages == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
