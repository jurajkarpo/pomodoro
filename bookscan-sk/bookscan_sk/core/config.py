"""Application-wide configuration and presets."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .types import (
    DewarpStrength,
    HardwareMode,
    OCREngineChoice,
    PreprocessPreset,
    ProcessingSettings,
)

APP_NAME = "BookScan SK"
APP_VERSION = "1.0.0"
APP_SLUG = "bookscan-sk"

# Slovak diacritic alphabet (lowercase + uppercase) used across the app.
SLOVAK_DIACRITICS_LOWER = "áäčďéíľĺňóôŕšťúýž"
SLOVAK_DIACRITICS_UPPER = "ÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ"
SLOVAK_DIACRITICS = SLOVAK_DIACRITICS_LOWER + SLOVAK_DIACRITICS_UPPER


def sys_platform() -> str:
    return sys.platform


def _default_data_dir() -> Path:
    base = os.environ.get("BOOKSCAN_SK_DATA")
    if base:
        return Path(base)
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", str(Path.home() / ".bookscan-sk"))) / "BookScanSK"
    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Application Support" / "BookScanSK"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "bookscan-sk"


DATA_DIR = _default_data_dir()
MODELS_DIR = DATA_DIR / "models"
PROJECTS_DIR = DATA_DIR / "projects"
CACHE_DIR = DATA_DIR / "cache"
TEMP_DIR = DATA_DIR / "temp"


@dataclass
class AppConfig:
    """Persistent application configuration."""

    default_preset: PreprocessPreset = PreprocessPreset.NATURAL
    default_dewarp: DewarpStrength = DewarpStrength.AUTO
    default_ocr_engine: OCREngineChoice = OCREngineChoice.AUTO
    default_language: str = "sk"
    default_hardware: HardwareMode = HardwareMode.AUTO
    default_workers: int = 0
    default_dpi: int = 300
    default_jpeg_quality: int = 92
    tesseract_cmd: str = "tesseract"
    paddle_enabled: bool = True
    rapid_enabled: bool = True
    tesseract_enabled: bool = True
    telemetry: bool = False  # always off by default; privacy first
    recent_projects: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "AppConfig":
        return cls()

    def save(self) -> None:
        pass


@dataclass(frozen=True)
class PresetDefinition:
    name: PreprocessPreset
    label: str
    description: str
    grayscale: bool = False
    black_white: bool = False
    contrast: float = 1.0
    sharpen: float = 0.0
    denoise: float = 0.0
    adaptive_threshold: bool = False
    shadow_reduction: bool = True
    illumination_norm: bool = True
    dewarp: DewarpStrength = DewarpStrength.AUTO


PRESETS: dict[PreprocessPreset, PresetDefinition] = {
    PreprocessPreset.ORIGINAL: PresetDefinition(
        name=PreprocessPreset.ORIGINAL, label="Original",
        description="Keep the photograph as-is; only crop and deskew.",
        grayscale=False, black_white=False, contrast=1.0, sharpen=0.0,
        denoise=0.0, adaptive_threshold=False, shadow_reduction=False,
        illumination_norm=False, dewarp=DewarpStrength.OFF,
    ),
    PreprocessPreset.NATURAL: PresetDefinition(
        name=PreprocessPreset.NATURAL, label="Natural",
        description="Gentle cleanup that preserves color and appearance.",
        grayscale=False, black_white=False, contrast=1.05, sharpen=0.2,
        denoise=0.3, adaptive_threshold=False, shadow_reduction=True,
        illumination_norm=True, dewarp=DewarpStrength.AUTO,
    ),
    PreprocessPreset.DOCUMENT: PresetDefinition(
        name=PreprocessPreset.DOCUMENT, label="Document",
        description="Balanced cleanup for clearer text while keeping the look.",
        grayscale=True, black_white=False, contrast=1.15, sharpen=0.4,
        denoise=0.4, adaptive_threshold=False, shadow_reduction=True,
        illumination_norm=True, dewarp=DewarpStrength.AUTO,
    ),
    PreprocessPreset.OCR: PresetDefinition(
        name=PreprocessPreset.OCR, label="OCR optimized",
        description="Maximize OCR accuracy; still faithful to the page.",
        grayscale=True, black_white=False, contrast=1.25, sharpen=0.6,
        denoise=0.5, adaptive_threshold=True, shadow_reduction=True,
        illumination_norm=True, dewarp=DewarpStrength.AUTO,
    ),
    PreprocessPreset.BLACK_WHITE: PresetDefinition(
        name=PreprocessPreset.BLACK_WHITE, label="Black & white",
        description="High-contrast binarized output for archival scans.",
        grayscale=True, black_white=True, contrast=1.3, sharpen=0.5,
        denoise=0.4, adaptive_threshold=True, shadow_reduction=True,
        illumination_norm=True, dewarp=DewarpStrength.OFF,
    ),
}


def settings_from_preset(preset: PreprocessPreset) -> ProcessingSettings:
    """Build a ProcessingSettings from a preset, keeping OCR/engine defaults."""
    defn = PRESETS[preset]
    return ProcessingSettings(
        preset=preset,
        dewarp=defn.dewarp,
        grayscale=defn.grayscale,
        black_white=defn.black_white,
        contrast=defn.contrast,
        sharpen=defn.sharpen,
        denoise=defn.denoise,
        adaptive_threshold=defn.adaptive_threshold,
        shadow_reduction=defn.shadow_reduction,
        illumination_norm=defn.illumination_norm,
    )


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, PROJECTS_DIR, CACHE_DIR, TEMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
