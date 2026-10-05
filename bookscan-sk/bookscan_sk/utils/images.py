"""Image I/O: reading, EXIF orientation, format detection, saving."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from ..core.types import ImageFormat


def detect_format(path: str) -> ImageFormat:
    _, ext = os.path.splitext(path)
    ext = ext.lower().lstrip(".")
    mapping = {
        "jpg": ImageFormat.JPEG,
        "jpeg": ImageFormat.JPEG,
        "png": ImageFormat.PNG,
        "webp": ImageFormat.WEBP,
        "tif": ImageFormat.TIFF,
        "tiff": ImageFormat.TIFF,
        "bmp": ImageFormat.BMP,
    }
    return mapping.get(ext, ImageFormat.UNKNOWN)


def read_image(path: str) -> np.ndarray:
    """Read an image to an RGB numpy array (H, W, 3), applying EXIF orientation."""
    from ..core.errors import ImageError

    if not os.path.isfile(path):
        raise ImageError(f"File not found: {path}")
    try:
        img = Image.open(path)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return np.ascontiguousarray(arr)
    except ImageError:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise ImageError(f"Cannot read image {path}: {e}")


def read_image_pil(path: str) -> Image.Image:
    from ..core.errors import ImageError

    try:
        img = Image.open(path)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        return img
    except Exception as e:
        raise ImageError(f"Cannot read image {path}: {e}")


def exif_orientation(path: str) -> int:
    """Return the EXIF orientation tag value (default 1)."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        return int(exif.get(274, 1) or 1)
    except Exception:
        return 1


def image_size(path: str) -> Tuple[int, int]:
    """Return (width, height) without loading full pixel data."""
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


def save_image(arr: np.ndarray, path: str, quality: int = 92, format: Optional[str] = None) -> str:
    """Save an RGB numpy array to disk. Returns the path."""
    from ..core.errors import ImageError

    try:
        img = Image.fromarray(arr)
        ext = (format or os.path.splitext(path)[1].lstrip(".") or "jpg").lower()
        if ext in ("jpg", "jpeg"):
            img = img.convert("RGB")
            img.save(path, "JPEG", quality=quality, subsampling=0)
        elif ext == "webp":
            img.save(path, "WEBP", quality=quality)
        elif ext == "png":
            img.save(path, "PNG")
        elif ext in ("tif", "tiff"):
            img.save(path, "TIFF")
        else:
            img = img.convert("RGB")
            img.save(path, "JPEG", quality=quality)
        return path
    except Exception as e:
        raise ImageError(f"Cannot save image {path}: {e}")


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        return arr[:, :, :3]
    return arr


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert RGB array to single-channel grayscale (uint8)."""
    import cv2

    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def resize_max(arr: np.ndarray, max_side: int) -> np.ndarray:
    """Resize so the longest side is <= max_side, preserving aspect."""
    import cv2

    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
