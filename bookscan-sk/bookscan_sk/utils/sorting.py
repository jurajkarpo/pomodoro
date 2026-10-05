"""Natural (human-friendly) filename sorting.

Sorts "page2" before "page10" instead of lexicographic ordering.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(s: str) -> list:
    """Split a string into text and numeric chunks for natural sorting."""
    parts: list = []
    for part in _NUM_RE.split(s):
        if part.isdigit():
            parts.append((1, int(part), ""))
        elif part:
            parts.append((0, 0, part.lower()))
    return parts


def natural_sort_key(s: str):
    return _natural_key(s)


def natural_sorted(paths: Iterable[str]) -> List[str]:
    """Return a new list sorted with natural numeric ordering."""
    return sorted(paths, key=natural_sort_key)


def natural_compare(a: str, b: str) -> int:
    ka, kb = _natural_key(a), _natural_key(b)
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


def sort_image_paths(paths: Iterable[str]) -> List[str]:
    """Sort image paths naturally, preserving a stable order."""
    return natural_sorted(paths)


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp",
}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTENSIONS


def list_image_files(directory: str) -> List[str]:
    """List all supported image files in a directory, naturally sorted."""
    if not os.path.isdir(directory):
        return []
    entries = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if is_image_file(name) and os.path.isfile(os.path.join(directory, name))
    ]
    return sort_image_paths(entries)
