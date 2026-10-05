"""Hyphenation handling.

Detects words broken at the end of a line and rejoins them:

    "nej-\\naký"  ->  "nejaký"

Rules:
  - A hyphen at the END of a line (followed by a line break)
    where the next line begins with a lowercase letter is
    treated as a line-break hyphen and rejoined.
  - A hyphen in the MIDDLE of a line is a real hyphen and is
    kept (e.g. "čierno-biely").
  - Exception: if both sides of an end-of-line hyphen are
    complete common words, the hyphen is kept (a real compound
    that happens to wrap).

The original OCR representation is always preserved alongside
the rejoined text for debugging.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.types import OCRBlock, OCRLine, OCRPage, OCRWord, Rect

# Common Slovak words used to detect *real* compound hyphens
# that happen to fall at a line break.
_COMMON_WORDS = {
    "a", "aj", "ale", "ani", "bez", "cez", "do", "na", "ne", "no",
    "o", "od", "pod", "po", "pre", "pri", "s", "v", "z", "za", "zo",
    "co", "ci", "du", "e", "ho", "i", "im", "in", "is", "it", "je", "ju",
    "k", "ke", "ku", "la", "le", "ma", "me", "mi", "mu", "om", "on", "os",
    "ot", "ou", "ov", "pa", "pe", "pro", "rad", "se", "si", "so", "sub",
    "ta", "te", "ti", "to", "tu", "u", "uz", "va", "ve", "vi", "vo", "vs",
    "vy", "ze", "zi", "dva", "dve", "tri", "styri", "pat", "desat",
    "sto", "tisic", "milion", "den", "noc", "cas", "svet", "zivot",
    "clovek", "ludia", "voda", "zem", "dom", "mesto", "kniha", "strom",
    "slnko", "mesiac", "hviezda", "pes", "krava", "kon", "ovca",
    "vela", "malo", "velky", "maly", "novy", "stary", "dobry", "zly",
    "prvy", "posledny", "dalsi", "svoj", "tvoj", "jeho", "jej", "ich",
    "toto", "take", "takeho",
}


@dataclass
class HyphenationResult:
    text: str
    rejoined_count: int = 0
    changes: list[HyphenChange] = field(default_factory=list)


@dataclass
class HyphenChange:
    original: str
    rejoined: str
    line_index: int


def rejoin_text(text: str) -> HyphenationResult:
    """Rejoin line-break hyphens in a raw OCR text."""
    changes: list[HyphenChange] = []
    lines = text.split("\n")
    out_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()
            merged, change = _try_rejoin(line, next_line, i)
            if merged is not None:
                out_lines.append(merged)
                changes.append(change)
                consumed = change.original.split("\n")[1]
                if lines[i + 1].startswith(consumed):
                    lines[i + 1] = lines[i + 1][len(consumed):]
                if lines[i + 1].strip() == "":
                    i += 1
                i += 1
                continue
        out_lines.append(line)
        i += 1

    rejoined = "\n".join(out_lines)
    return HyphenationResult(
        text=rejoined,
        rejoined_count=len(changes),
        changes=changes,
    )


def _try_rejoin(line: str, next_line: str, line_index: int) -> tuple[str | None, HyphenChange | None]:
    """Try to rejoin a hyphen at the end of `line` with `next_line`."""
    stripped = line.rstrip()
    if not stripped.endswith("-"):
        return None, None
    if not next_line:
        return None, None
    if len(stripped) < 2 or not stripped[-2].isalpha():
        return None, None
    first = next_line[0]
    if not (first.isalpha() and first.islower()):
        return None, None

    before = stripped[:-1]
    parts = next_line.split(" ", 1)
    continuation = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    before_word = _last_word(before)
    if before_word.lower() in _COMMON_WORDS and continuation.lower() in _COMMON_WORDS:
        return None, None

    rejoined_word = before_word + continuation
    new_line = before[: len(before) - len(before_word)] + rejoined_word
    if rest:
        new_line = new_line + " " + rest

    original = stripped + "\n" + next_line
    change = HyphenChange(original=original, rejoined=new_line, line_index=line_index)
    return new_line, change


def _last_word(text: str) -> str:
    parts = text.split()
    return parts[-1] if parts else text


def apply_hyphenation(page: OCRPage) -> OCRPage:
    """Apply hyphenation rejoining to an OCRPage."""
    for block in page.blocks:
        block_text = "\n".join(line.text for line in block.lines)
        result = rejoin_text(block_text)
        if result.rejoined_count > 0:
            new_lines = result.text.split("\n")
            for idx, line in enumerate(block.lines):
                if idx < len(new_lines):
                    _update_line_words(line, new_lines[idx])
    page.hyphenation_applied = True
    return page


def _update_line_words(line: OCRLine, new_text: str) -> None:
    """Update a line's words to match rejoined text (best effort)."""
    new_words = new_text.split()
    if len(new_words) == len(line.words):
        for w, t in zip(line.words, new_words):
            w.text = t
            w.corrected_text = None
        return
    if line.words:
        box = line.words[0].box
        conf = sum(w.confidence for w in line.words) / len(line.words)
        line.words = [OCRWord(text=new_text, box=box, confidence=conf)]
    else:
        line.words = [OCRWord(text=new_text, box=line.box or Rect(0, 0, 0, 0))]
