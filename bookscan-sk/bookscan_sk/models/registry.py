"""Model registry.

Every model, trained-data file and bundled asset is registered
with its source URL, version, checksum, license and
redistribution status. This is the single source of truth
for the license audit (THIRD_PARTY_LICENSES.md) and for
the model manager's setup diagnostics.

Licenses verified against upstream sources:
  - PaddleOCR / PP-OCRv5 models: Apache-2.0
    (https://github.com/PaddlePaddle/PaddleOCR)
  - Tesseract traineddata (tessdata_fast / tessdata_best):
    Apache-2.0
    (https://github.com/tesseract-ocr/tessdata_fast)
  - DejaVu Sans font: Bitstream Vera License / free for
    redistribution
    (https://dejavu-fonts.github.io/)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelEntry:
    id: str
    name: str
    version: str
    source_url: str
    model_url: str
    checksum: str = ""  # SHA-256 (empty = not yet verified)
    license: str = ""
    license_url: str = ""
    redistribution: bool = True
    category: str = "ocr"  # ocr, font, dictionary
    install_hint: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# PaddleOCR PP-OCRv5 models (Apache-2.0)
# ---------------------------------------------------------------------------

PADDLE_OCR_MODELS: list[ModelEntry] = [
    ModelEntry(
        id="paddle.ppocrv5_det",
        name="PP-OCRv5 text detection",
        version="PP-OCRv5",
        source_url="https://github.com/PaddlePaddle/PaddleOCR",
        model_url=(
            "https://paddle-model-ecology.bj.bcebos.com/"
            "paddlex/official_inference_model/paddle3.0b1/"
            "PP-OCRv5_server_det_infer.tar"
        ),
        license="Apache-2.0",
        license_url="https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE",
        redistribution=True,
        category="ocr",
        install_hint="Installed automatically by PaddleOCR on first use.",
        notes="Text detection model. Downloaded by paddlex to the PaddleX model cache.",
    ),
    ModelEntry(
        id="paddle.ppocrv5_rec_multilingual",
        name="PP-OCRv5 multilingual recognition",
        version="PP-OCRv5",
        source_url="https://github.com/PaddlePaddle/PaddleOCR",
        model_url=(
            "https://paddle-model-ecology.bj.bcebos.com/"
            "paddlex/official_inference_model/paddle3.0b1/"
            "PP-OCRv5_server_rec_infer.tar"
        ),
        license="Apache-2.0",
        license_url="https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE",
        redistribution=True,
        category="ocr",
        install_hint="Installed automatically by PaddleOCR on first use.",
        notes=(
            "Multilingual recognition model covering 106 languages "
            "including Slovak (sk). See PP-OCRv5_multi_languages.md."
        ),
    ),
    ModelEntry(
        id="paddle.ppocrv5_rec_latin",
        name="PP-OCRv5 Latin-script recognition",
        version="PP-OCRv5",
        source_url="https://github.com/PaddlePaddle/PaddleOCR",
        model_url=(
            "https://paddle-model-ecology.bj.bcebos.com/"
            "paddlex/official_inference_model/paddle3.0b1/"
            "latin_PP-OCRv5_mobile_rec_infer.tar"
        ),
        license="Apache-2.0",
        license_url="https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE",
        redistribution=True,
        category="ocr",
        install_hint="Optional: better Latin-script diacritic accuracy.",
        notes="Latin-script recognition model (Slovak is Latin-script).",
    ),
]

# ---------------------------------------------------------------------------
# Tesseract traineddata (Apache-2.0)
# ---------------------------------------------------------------------------

TESSERACT_MODELS: list[ModelEntry] = [
    ModelEntry(
        id="tesseract.slk",
        name="Tesseract Slovak (slk) traineddata",
        version="tessdata_fast",
        source_url="https://github.com/tesseract-ocr/tessdata_fast",
        model_url=(
            "https://github.com/tesseract-ocr/tessdata_fast/raw/main/slk.traineddata"
        ),
        license="Apache-2.0",
        license_url="https://github.com/tesseract-ocr/tessdata_fast/blob/main/LICENSE",
        redistribution=True,
        category="ocr",
        install_hint=(
            "Install via: apt install tesseract-ocr tesseract-ocr-slk  "
            "(Debian/Ubuntu) or download slk.traineddata to the tessdata dir."
        ),
        notes="Slovak language model for Tesseract 5.",
    ),
    ModelEntry(
        id="tesseract.eng",
        name="Tesseract English (eng) traineddata",
        version="tessdata_fast",
        source_url="https://github.com/tesseract-ocr/tessdata_fast",
        model_url=(
            "https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata"
        ),
        license="Apache-2.0",
        license_url="https://github.com/tesseract-ocr/tessdata_fast/blob/main/LICENSE",
        redistribution=True,
        category="ocr",
        install_hint="Usually bundled with Tesseract.",
        notes="English language model (fallback).",
    ),
]

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

FONT_ASSETS: list[ModelEntry] = [
    ModelEntry(
        id="font.dejavu_sans",
        name="DejaVu Sans",
        version="2.37",
        source_url="https://dejavu-fonts.github.io/",
        model_url="https://dejavu-fonts.github.io/Download.html",
        license="Bitstream Vera License (free for redistribution)",
        license_url="https://dejavu-fonts.github.io/License.html",
        redistribution=True,
        category="font",
        install_hint="Install the 'fonts-dejavu' system package.",
        notes=(
            "Used for the invisible PDF text layer. Covers the full "
            "Slovak alphabet including all diacritics."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_MODELS: list[ModelEntry] = (
    PADDLE_OCR_MODELS + TESSERACT_MODELS + FONT_ASSETS
)


def get_model(model_id: str) -> ModelEntry | None:
    for m in ALL_MODELS:
        if m.id == model_id:
            return m
    return None


def models_by_category(category: str) -> list[ModelEntry]:
    return [m for m in ALL_MODELS if m.category == category]
