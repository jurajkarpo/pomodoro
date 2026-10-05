"""Core typed data structures for BookScan SK.

Every stage of the pipeline exchanges explicit, immutable-ish dataclasses so
that UI, OCR and image-processing code never share ad-hoc dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ImageFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    BMP = "bmp"
    UNKNOWN = "unknown"


class DewarpStrength(Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class PreprocessPreset(Enum):
    ORIGINAL = "original"
    NATURAL = "natural"
    DOCUMENT = "document"
    OCR = "ocr"
    BLACK_WHITE = "black_white"


class OCREngineChoice(Enum):
    PADDLE = "paddle"
    TESSERACT = "tesseract"
    AUTO = "auto"
    DUAL = "dual"


class OCREngineName(Enum):
    PADDLE = "paddle"
    RAPID = "rapid"
    TESSERACT = "tesseract"


class HardwareMode(Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class PageLayout(Enum):
    SINGLE = "single"
    SPREAD = "spread"
    UNKNOWN = "unknown"


class PageStatus(Enum):
    PENDING = "pending"
    DETECTING = "detecting"
    PREPROCESSING = "preprocessing"
    OCR = "ocr"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"


class JobStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def x0(self) -> float:
        return self.x

    @property
    def y0(self) -> float:
        return self.y

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def y1(self) -> float:
        return self.y + self.height

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Quad:
    """A four-corner polygon (page boundary) in clockwise order:
    top-left, top-right, bottom-right, bottom-left."""

    tl: Point
    tr: Point
    br: Point
    bl: Point

    def points(self) -> list[Point]:
        return [self.tl, self.tr, self.br, self.bl]

    def to_xy(self) -> list[tuple[float, float]]:
        return [(p.x, p.y) for p in self.points()]

    def to_numpy(self):
        import numpy as np

        return np.array(self.to_xy(), dtype="float32")

    @classmethod
    def from_points(cls, pts: list[tuple[float, float]]) -> "Quad":
        if len(pts) != 4:
            raise ValueError("Quad requires exactly 4 points")
        return cls(
            tl=Point(*pts[0]),
            tr=Point(*pts[1]),
            br=Point(*pts[2]),
            bl=Point(*pts[3]),
        )

    @classmethod
    def from_rect(cls, rect: Rect) -> "Quad":
        return cls(
            tl=Point(rect.x0, rect.y0),
            tr=Point(rect.x1, rect.y0),
            br=Point(rect.x1, rect.y1),
            bl=Point(rect.x0, rect.y1),
        )


# ---------------------------------------------------------------------------
# Source image
# ---------------------------------------------------------------------------

@dataclass
class SourceImage:
    """Reference to one input photograph."""

    path: str
    index: int
    filename: str
    format: ImageFormat = ImageFormat.UNKNOWN
    width: int = 0
    height: int = 0
    exif_orientation: int = 1
    sha256: str = ""
    added_at: float = 0.0

    @property
    def display_name(self) -> str:
        return self.filename


# ---------------------------------------------------------------------------
# Detected / processed page
# ---------------------------------------------------------------------------

@dataclass
class DetectedPage:
    """Result of page-boundary detection on a source image."""

    source: SourceImage
    quad: Quad
    layout: PageLayout = PageLayout.UNKNOWN
    gutter_x: Optional[float] = None  # x of central gutter for spreads
    confidence: float = 0.0
    skew_degrees: float = 0.0
    perspective_strength: float = 0.0  # 0..1
    method: str = "auto"
    manual: bool = False


@dataclass
class ProcessedPage:
    """A page after preprocessing: ready for OCR and PDF embedding."""

    source: SourceImage
    image_path: str = ""          # path to processed image on disk
    original_path: str = ""       # path to untouched copy
    width: int = 0
    height: int = 0
    dpi: int = 300
    quad: Optional[Quad] = None
    transform_matrix: Optional[list[list[float]]] = None  # source->processed
    preset: PreprocessPreset = PreprocessPreset.NATURAL
    dewarp_strength: DewarpStrength = DewarpStrength.AUTO
    dewarp_applied: bool = False
    status: PageStatus = PageStatus.PENDING
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OCR structures
# ---------------------------------------------------------------------------

@dataclass
class OCRWord:
    text: str
    box: Rect
    confidence: float = 0.0
    verified: bool = False
    ignored: bool = False
    corrected_text: Optional[str] = None
    engine: str = ""

    @property
    def display_text(self) -> str:
        return self.corrected_text if self.corrected_text is not None else self.text


@dataclass
class OCRLine:
    words: list[OCRWord] = field(default_factory=list)
    box: Optional[Rect] = None
    confidence: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.display_text for w in self.words)

    @property
    def raw_text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class OCRBlock:
    lines: list[OCRLine] = field(default_factory=list)
    box: Optional[Rect] = None
    block_type: str = "text"  # text, heading, footnote, page_number
    reading_order: int = 0
    confidence: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class OCRPage:
    blocks: list[OCRBlock] = field(default_factory=list)
    width: int = 0
    height: int = 0
    engine: str = ""
    language: str = "sk"
    average_confidence: float = 0.0
    reading_order_applied: bool = False
    hyphenation_applied: bool = False
    raw_text: str = ""
    # dual-validation diagnostics
    secondary_text: str = ""
    agreement_score: float = 0.0

    @property
    def full_text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)

    def all_words(self) -> list[OCRWord]:
        out: list[OCRWord] = []
        for block in self.blocks:
            for line in block.lines:
                out.extend(line.words)
        return out

    def low_confidence_words(self, threshold: float = 0.6) -> list[OCRWord]:
        return [w for w in self.all_words() if w.confidence < threshold and not w.ignored]


# ---------------------------------------------------------------------------
# Document / job / result
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    title: str = ""
    author: str = ""
    subject: str = ""
    creator: str = "BookScan SK"
    ocr_language: str = "sk"
    creation_date: str = ""
    keywords: str = ""
    first_page_number: int = 1
    page_offset: int = 0


@dataclass
class ExportOptions:
    output_pdf: str = ""
    output_txt: str = ""
    output_alto: str = ""
    output_page_xml: str = ""
    export_txt: bool = False
    export_alto: bool = False
    export_page_xml: bool = False
    jpeg_quality: int = 92
    dpi: int = 300
    pdf_a: bool = False
    include_original: bool = False


@dataclass
class ProcessingSettings:
    preset: PreprocessPreset = PreprocessPreset.NATURAL
    dewarp: DewarpStrength = DewarpStrength.AUTO
    ocr_engine: OCREngineChoice = OCREngineChoice.AUTO
    language: str = "sk"
    hardware: HardwareMode = HardwareMode.AUTO
    workers: int = 0  # 0 = auto
    dpi: int = 300
    jpeg_quality: int = 92
    first_page_number: int = 1
    page_offset: int = 0
    # advanced preprocessing knobs
    rotate_degrees: float = 0.0
    contrast: float = 1.0
    sharpen: float = 0.0
    denoise: float = 0.0
    adaptive_threshold: bool = False
    grayscale: bool = False
    black_white: bool = False
    shadow_reduction: bool = True
    illumination_norm: bool = True


@dataclass
class PageResult:
    """Outcome of processing a single page."""

    page_index: int
    source: SourceImage
    detected: Optional[DetectedPage] = None
    processed: Optional[ProcessedPage] = None
    ocr: Optional[OCRPage] = None
    status: PageStatus = PageStatus.PENDING
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    processing_ms: float = 0.0


@dataclass
class ProcessingJob:
    id: str = ""
    name: str = ""
    settings: ProcessingSettings = field(default_factory=ProcessingSettings)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    export_options: ExportOptions = field(default_factory=ExportOptions)
    status: JobStatus = JobStatus.CREATED
    created_at: float = 0.0
    updated_at: float = 0.0
    source_paths: list[str] = field(default_factory=list)
    completed_pages: int = 0
    total_pages: int = 0


@dataclass
class ProcessingResult:
    job: ProcessingJob
    pages: list[PageResult] = field(default_factory=list)
    output_pdf: str = ""
    output_txt: str = ""
    output_alto: str = ""
    output_page_xml: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def successful_pages(self) -> int:
        return sum(1 for p in self.pages if p.status == PageStatus.DONE)

    @property
    def failed_pages(self) -> int:
        return sum(1 for p in self.pages if p.status == PageStatus.FAILED)

    @property
    def average_confidence(self) -> float:
        confs = [p.ocr.average_confidence for p in self.pages if p.ocr is not None]
        return sum(confs) / len(confs) if confs else 0.0


# ---------------------------------------------------------------------------
# Progress events
# ---------------------------------------------------------------------------

class ProgressEvent:
    """Emitted by the pipeline to UI/CLI."""

    def __init__(
        self,
        kind: str,
        message: str = "",
        current: int = 0,
        total: int = 0,
        page_index: int = -1,
        stage: str = "",
        average_confidence: float = 0.0,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        self.kind = kind  # stage, page_done, progress, error, warning, done
        self.message = message
        self.current = current
        self.total = total
        self.page_index = page_index
        self.stage = stage
        self.average_confidence = average_confidence
        self.extra = extra or {}

    def __repr__(self) -> str:
        return (
            f"ProgressEvent(kind={self.kind!r}, message={self.message!r}, "
            f"current={self.current}, total={self.total})"
        )
