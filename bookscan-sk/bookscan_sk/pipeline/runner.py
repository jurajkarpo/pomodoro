"""Pipeline runner: batched, parallel, resumable processing.

Design goals:
  - Never load an entire book into RAM (stream page by page).
  - Bounded worker pool for CPU-bound preprocessing + OCR.
  - Pause / resume / cancel without losing completed pages.
  - Persist job state after every page so an interrupted job
    can resume from the last completed page.
  - Cache reuse: skip pages whose source + settings are unchanged.

Two execution modes:
  serial   - one OCR manager reused in the main process
             (most reliable; default)
  parallel - a persistent pool of workers, each with its own
             OCR manager (faster on many-core machines)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.types import (
    DetectedPage,
    JobStatus,
    OCRPage,
    PageResult,
    PageStatus,
    ProcessingJob,
    ProcessingResult,
    ProcessingSettings,
    ProcessedPage,
    ProgressEvent,
    SourceImage,
)
from ..ocr.hyphenation import apply_hyphenation
from ..ocr.manager import OCRManager
from ..ocr.reading_order import apply_reading_order
from ..page_detection.detector import detect_page
from ..preprocessing.orientation import apply_exif_orientation
from ..preprocessing.pipeline import preprocess_page
from ..utils.images import exif_orientation, read_image, save_image, sha256_file
from .cache import PageCache


@dataclass
class RunnerConfig:
    settings: ProcessingSettings
    output_dir: str
    workers: int = 0  # 0 = auto
    use_multiprocessing: bool = False  # serial by default (most reliable)
    tesseract_cmd: str = "tesseract"
    cache: Optional[PageCache] = None
    state_path: Optional[str] = None


@dataclass
class JobState:
    """Persisted job state for resume."""

    completed: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    current: int = 0
    status: str = JobStatus.RUNNING.value

    def to_json(self) -> str:
        return json.dumps({
            "completed": self.completed,
            "failed": self.failed,
            "current": self.current,
            "status": self.status,
        })

    @classmethod
    def from_json(cls, data: str) -> "JobState":
        obj = json.loads(data)
        return cls(
            completed=obj.get("completed", {}),
            failed=obj.get("failed", {}),
            current=obj.get("current", 0),
            status=obj.get("status", JobStatus.RUNNING.value),
        )


class ControlState:
    """Shared pause/cancel control."""

    def __init__(self) -> None:
        self._paused = False
        self._cancelled = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def wait_if_paused(self) -> None:
        while self._paused and not self._cancelled:
            time.sleep(0.2)


# ---------------------------------------------------------------------------
# Settings (de)serialization
# ---------------------------------------------------------------------------

def _settings_to_dict(s: ProcessingSettings) -> dict:
    return {
        "preset": s.preset.value,
        "dewarp": s.dewarp.value,
        "ocr_engine": s.ocr_engine.value,
        "language": s.language,
        "hardware": s.hardware.value,
        "workers": s.workers,
        "dpi": s.dpi,
        "jpeg_quality": s.jpeg_quality,
        "first_page_number": s.first_page_number,
        "page_offset": s.page_offset,
        "rotate_degrees": s.rotate_degrees,
        "contrast": s.contrast,
        "sharpen": s.sharpen,
        "denoise": s.denoise,
        "adaptive_threshold": s.adaptive_threshold,
        "grayscale": s.grayscale,
        "black_white": s.black_white,
        "shadow_reduction": s.shadow_reduction,
        "illumination_norm": s.illumination_norm,
    }


def _settings_from_dict(d: dict) -> ProcessingSettings:
    from ..core.types import DewarpStrength, HardwareMode, OCREngineChoice, PreprocessPreset

    return ProcessingSettings(
        preset=PreprocessPreset(d["preset"]),
        dewarp=DewarpStrength(d["dewarp"]),
        ocr_engine=OCREngineChoice(d["ocr_engine"]),
        language=d["language"],
        hardware=HardwareMode(d.get("hardware", "auto")),
        workers=d.get("workers", 0),
        dpi=d.get("dpi", 300),
        jpeg_quality=d.get("jpeg_quality", 92),
        first_page_number=d.get("first_page_number", 1),
        page_offset=d.get("page_offset", 0),
        rotate_degrees=d.get("rotate_degrees", 0.0),
        contrast=d.get("contrast", 1.0),
        sharpen=d.get("sharpen", 0.0),
        denoise=d.get("denoise", 0.0),
        adaptive_threshold=d.get("adaptive_threshold", False),
        grayscale=d.get("grayscale", False),
        black_white=d.get("black_white", False),
        shadow_reduction=d.get("shadow_reduction", True),
        illumination_norm=d.get("illumination_norm", True),
    )


# ---------------------------------------------------------------------------
# (De)serialization of detection and OCR results
# ---------------------------------------------------------------------------

def _detected_to_dict(d: DetectedPage) -> dict:
    return {
        "layout": d.layout.value,
        "confidence": d.confidence,
        "skew_degrees": d.skew_degrees,
        "perspective_strength": d.perspective_strength,
        "method": d.method,
        "manual": d.manual,
        "quad": [
            [d.quad.tl.x, d.quad.tl.y],
            [d.quad.tr.x, d.quad.tr.y],
            [d.quad.br.x, d.quad.br.y],
            [d.quad.bl.x, d.quad.bl.y],
        ],
    }


def _ocr_to_dict(page: OCRPage) -> dict:
    blocks = []
    for b in page.blocks:
        lines = []
        for l in b.lines:
            words = [
                {
                    "text": w.text,
                    "box": [w.box.x, w.box.y, w.box.width, w.box.height],
                    "confidence": w.confidence,
                    "verified": w.verified,
                    "ignored": w.ignored,
                    "corrected_text": w.corrected_text,
                }
                for w in l.words
            ]
            box = [l.box.x, l.box.y, l.box.width, l.box.height] if l.box else None
            lines.append({"words": words, "box": box, "confidence": l.confidence})
        box = [b.box.x, b.box.y, b.box.width, b.box.height] if b.box else None
        blocks.append({
            "lines": lines,
            "box": box,
            "block_type": b.block_type,
            "reading_order": b.reading_order,
            "confidence": b.confidence,
        })
    return {
        "blocks": blocks,
        "width": page.width,
        "height": page.height,
        "engine": page.engine,
        "language": page.language,
        "average_confidence": page.average_confidence,
        "raw_text": page.raw_text,
        "secondary_text": page.secondary_text,
        "agreement_score": page.agreement_score,
    }


def _ocr_from_dict(d: dict) -> OCRPage:
    from ..core.types import OCRBlock, OCRLine, OCRWord, Rect

    blocks = []
    for bd in d.get("blocks", []):
        lines = []
        for ld in bd.get("lines", []):
            words = []
            for wd in ld.get("words", []):
                bx = wd.get("box") or [0, 0, 0, 0]
                words.append(
                    OCRWord(
                        text=wd["text"],
                        box=Rect(bx[0], bx[1], bx[2], bx[3]),
                        confidence=wd.get("confidence", 0.0),
                        verified=wd.get("verified", False),
                        ignored=wd.get("ignored", False),
                        corrected_text=wd.get("corrected_text"),
                    )
                )
            lb = ld.get("box")
            lines.append(
                OCRLine(
                    words=words,
                    box=Rect(lb[0], lb[1], lb[2], lb[3]) if lb else None,
                    confidence=ld.get("confidence", 0.0),
                )
            )
        bb = bd.get("box")
        blocks.append(
            OCRBlock(
                lines=lines,
                box=Rect(bb[0], bb[1], bb[2], bb[3]) if bb else None,
                block_type=bd.get("block_type", "text"),
                reading_order=bd.get("reading_order", 0),
                confidence=bd.get("confidence", 0.0),
            )
        )
    return OCRPage(
        blocks=blocks,
        width=d.get("width", 0),
        height=d.get("height", 0),
        engine=d.get("engine", ""),
        language=d.get("language", "sk"),
        average_confidence=d.get("average_confidence", 0.0),
        raw_text=d.get("raw_text", ""),
        secondary_text=d.get("secondary_text", ""),
        agreement_score=d.get("agreement_score", 0.0),
    )


def _safe_name(name: str) -> str:
    keep = "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in name)[:80]


# ---------------------------------------------------------------------------
# Single-page processing
# ---------------------------------------------------------------------------

def process_page(
    source_path: str,
    index: int,
    settings: ProcessingSettings,
    output_dir: str,
    manager: Optional[OCRManager] = None,
    tesseract_cmd: str = "tesseract",
) -> dict:
    """Process a single page. Returns a JSON-serializable dict."""
    import traceback

    result = {
        "index": index,
        "source_path": source_path,
        "status": PageStatus.FAILED.value,
        "error": None,
        "warnings": [],
        "processed_image": "",
        "original_copy": "",
        "ocr_json": "",
        "detected": None,
        "processing_ms": 0.0,
    }
    t0 = time.time()
    try:
        image = read_image(source_path)
        orientation = exif_orientation(source_path)
        if orientation != 1:
            image = apply_exif_orientation(image, orientation)

        source = SourceImage(
            path=source_path,
            index=index,
            filename=os.path.basename(source_path),
            width=image.shape[1],
            height=image.shape[0],
            exif_orientation=orientation,
            sha256=sha256_file(source_path),
        )

        detected = detect_page(image, source)
        result["detected"] = _detected_to_dict(detected)

        pre = preprocess_page(image, detected, settings)

        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(source_path))[0]
        safe = _safe_name(f"{index:04d}_{base}")
        original_copy = os.path.join(output_dir, f"{safe}_original.jpg")
        processed_path = os.path.join(output_dir, f"{safe}_processed.jpg")
        save_image(image, original_copy, quality=95)
        save_image(pre.image, processed_path, quality=settings.jpeg_quality)
        result["processed_image"] = processed_path
        result["original_copy"] = original_copy

        if manager is None:
            manager = OCRManager(tesseract_cmd=tesseract_cmd)
        ocr_page = manager.recognize(pre.image, settings.language, settings.ocr_engine)
        apply_reading_order(ocr_page)
        apply_hyphenation(ocr_page)

        ocr_json = os.path.join(output_dir, f"{safe}_ocr.json")
        with open(ocr_json, "w", encoding="utf-8") as f:
            json.dump(_ocr_to_dict(ocr_page), f, ensure_ascii=False)
        result["ocr_json"] = ocr_json
        result["status"] = PageStatus.DONE.value
        result["warnings"] = pre.notes
    except Exception as e:  # noqa: BLE001 - one page must not kill the job
        result["status"] = PageStatus.FAILED.value
        result["error"] = f"{type(e).__name__}: {e}"
        result["warnings"].append(traceback.format_exc(limit=3))
    result["processing_ms"] = (time.time() - t0) * 1000.0
    return result


# Worker entry point for multiprocessing.
_WORKER_TESSERACT_CMD = "tesseract"


def _worker_init(tesseract_cmd: str) -> None:
    global _WORKER_TESSERACT_CMD
    _WORKER_TESSERACT_CMD = tesseract_cmd


def _worker_process(task: dict) -> dict:
    """Pool worker: builds its own OCR manager and processes a page."""
    settings = _settings_from_dict(task["settings"])
    return process_page(
        task["source_path"],
        task["index"],
        settings,
        task["output_dir"],
        manager=None,
        tesseract_cmd=task.get("tesseract_cmd", _WORKER_TESSERACT_CMD),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Runs a processing job over a list of source images."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.cache = config.cache or PageCache()
        self.control = ControlState()
        self._state = JobState()
        self._state_path = config.state_path or os.path.join(
            config.output_dir, ".jobstate.json"
        )
        self._manager: Optional[OCRManager] = None
        self._pool: Optional[mp.Pool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        sources: list[SourceImage],
        on_event: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> ProcessingResult:
        os.makedirs(self.config.output_dir, exist_ok=True)
        job = ProcessingJob(
            name="book",
            settings=self.config.settings,
            source_paths=[s.path for s in sources],
            total_pages=len(sources),
        )
        result = ProcessingResult(job=job)
        result.started_at = time.time()

        self._load_state()
        tasks = self._build_tasks(sources)
        total = len(tasks)
        self._emit(on_event, ProgressEvent("start", f"Processing {total} pages", 0, total))

        completed = 0
        try:
            if self.config.use_multiprocessing and self.config.workers != 1:
                iterator = self._parallel_iter(tasks)
            else:
                iterator = self._serial_iter(tasks)

            for raw in iterator:
                if self.control.is_cancelled:
                    self._state.status = JobStatus.CANCELLED.value
                    self._save_state()
                    result.job.status = JobStatus.CANCELLED
                    self._emit(on_event, ProgressEvent("cancelled", "Cancelled"))
                    break
                self.control.wait_if_paused()

                idx = raw["index"]
                if str(idx) in self._state.completed:
                    completed += 1
                    self._emit_progress(on_event, completed, total)
                    continue

                page_result = self._raw_to_page_result(raw)
                self._record_result(idx, page_result)
                result.pages.append(page_result)
                completed += 1
                self._emit_progress(on_event, completed, total, page_result)
        finally:
            self._cleanup()

        result.finished_at = time.time()
        if result.job.status != JobStatus.CANCELLED:
            result.job.status = JobStatus.COMPLETED
        self._emit(on_event, ProgressEvent("done", "Processing complete", total, total))
        return result

    def pause(self) -> None:
        self.control.pause()

    def resume(self) -> None:
        self.control.resume()

    def cancel(self) -> None:
        self.control.cancel()

    # ------------------------------------------------------------------
    # Iterators
    # ------------------------------------------------------------------

    def _serial_iter(self, tasks: list[dict]):
        """Serial processing reusing one OCR manager."""
        if self._manager is None:
            self._manager = OCRManager(tesseract_cmd=self.config.tesseract_cmd)
        for task in tasks:
            settings = _settings_from_dict(task["settings"])
            yield process_page(
                task["source_path"],
                task["index"],
                settings,
                task["output_dir"],
                manager=self._manager,
                tesseract_cmd=self.config.tesseract_cmd,
            )

    def _parallel_iter(self, tasks: list[dict]):
        """Parallel processing with a bounded persistent pool."""
        workers = self.config.workers or max(1, (os.cpu_count() or 2) - 1)
        workers = max(1, min(workers, 8))
        self._pool = mp.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(self.config.tesseract_cmd,),
        )
        yield from self._pool.imap(_worker_process, tasks)

    def _cleanup(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tasks(self, sources: list[SourceImage]) -> list[dict]:
        settings_dict = _settings_to_dict(self.config.settings)
        return [
            {
                "source_path": s.path,
                "index": s.index,
                "settings": settings_dict,
                "output_dir": self.config.output_dir,
                "tesseract_cmd": self.config.tesseract_cmd,
            }
            for s in sources
        ]

    def _raw_to_page_result(self, raw: dict) -> PageResult:
        source = SourceImage(
            path=raw["source_path"],
            index=raw["index"],
            filename=os.path.basename(raw["source_path"]),
        )
        ocr = None
        if raw.get("ocr_json") and os.path.isfile(raw["ocr_json"]):
            try:
                with open(raw["ocr_json"], encoding="utf-8") as f:
                    ocr = _ocr_from_dict(json.load(f))
            except Exception:
                ocr = None
        status = PageStatus(raw.get("status", PageStatus.FAILED.value))
        return PageResult(
            page_index=raw["index"],
            source=source,
            processed=ProcessedPage(
                source=source,
                image_path=raw.get("processed_image", ""),
                original_path=raw.get("original_copy", ""),
                status=status,
            ),
            ocr=ocr,
            status=status,
            error=raw.get("error"),
            warnings=raw.get("warnings", []),
            processing_ms=raw.get("processing_ms", 0.0),
        )

    def _record_result(self, idx: int, page_result: PageResult) -> None:
        ocr_path = ""
        if page_result.processed and page_result.processed.image_path:
            base = os.path.splitext(page_result.processed.image_path)[0]
            ocr_path = base + "_ocr.json"
        record = {
            "status": page_result.status.value,
            "error": page_result.error,
            "processed_image": page_result.processed.image_path if page_result.processed else "",
            "ocr_json": ocr_path,
        }
        if page_result.status == PageStatus.DONE:
            self._state.completed[str(idx)] = json.dumps(record)
        else:
            self._state.failed[str(idx)] = json.dumps(record)
        self._state.current = idx + 1
        self._save_state()

    def _load_state(self) -> None:
        if os.path.isfile(self._state_path):
            try:
                with open(self._state_path, encoding="utf-8") as f:
                    self._state = JobState.from_json(f.read())
            except Exception:
                self._state = JobState()

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as f:
                f.write(self._state.to_json())
        except Exception:
            pass

    def _emit(
        self,
        on_event: Optional[Callable[[ProgressEvent], None]],
        event: ProgressEvent,
    ) -> None:
        if on_event:
            on_event(event)

    def _emit_progress(
        self,
        on_event: Optional[Callable[[ProgressEvent], None]],
        completed: int,
        total: int,
        page_result: Optional[PageResult] = None,
    ) -> None:
        conf = 0.0
        if page_result and page_result.ocr:
            conf = page_result.ocr.average_confidence
        self._emit(
            on_event,
            ProgressEvent(
                "progress",
                f"Processing {completed} / {total}",
                completed,
                total,
                page_index=page_result.page_index if page_result else -1,
                average_confidence=conf,
            ),
        )
