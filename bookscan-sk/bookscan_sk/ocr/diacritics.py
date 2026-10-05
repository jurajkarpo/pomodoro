"""Slovak diacritic diagnostics.

Detects common OCR errors on Slovak text:
  - lost diacritics (ľ -> l, č -> c, ž -> z, ...)
  - missing accents
  - suspicious character sequences

The module never rewrites text automatically. It scores and flags
so the user (or the review queue) can decide.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from ..core.config import SLOVAK_DIACRITICS
from ..core.types import OCRPage, OCRWord

# Mapping of common OCR mis-recognitions: wrong -> correct Slovak char.
# These are *suggestions only*, never applied automatically.
DIACRITIC_CONFUSIONS: dict[str, str] = {
    "l": "ľ", "t": "ť", "d": "ď", "n": "ň", "c": "č", "s": "š",
    "z": "ž", "o": "ô", "a": "ä", "r": "ŕ", "u": "ú", "y": "ý",
    "e": "é", "i": "í",
    "L": "Ľ", "T": "Ť", "D": "Ď", "N": "Ň", "C": "Č", "S": "Š",
    "Z": "Ž", "O": "Ô", "A": "Ä", "R": "Ŕ", "U": "Ú", "Y": "Ý",
    "E": "É", "I": "Í",
}


@dataclass
class DiacriticReport:
    text: str
    diacritic_count: int = 0
    total_letters: int = 0
    preservation_score: float = 0.0
    suspicious_words: list[SuspiciousWord] = field(default_factory=list)


@dataclass
class SuspiciousWord:
    word: str
    reason: str
    suggestion: str = ""


def count_diacritics(text: str) -> int:
    return sum(1 for ch in text if ch in SLOVAK_DIACRITICS)


def count_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def diacritic_preservation_score(text: str) -> float:
    """Estimate how well diacritics are preserved (0..1).

    Slovak text typically has ~5-8% diacritic characters.
    """
    letters = count_letters(text)
    if letters == 0:
        return 1.0
    diac = count_diacritics(text)
    density = diac / letters
    expected = 0.06
    return min(1.0, density / expected)


def analyze_text(text: str) -> DiacriticReport:
    """Analyze a text string for diacritic quality."""
    words = text.split()
    suspicious: list[SuspiciousWord] = []
    for w in words:
        clean = _strip_punct(w)
        if not clean:
            continue
        sus = _analyze_word(clean)
        if sus:
            suspicious.append(sus)
    letters = count_letters(text)
    diac = count_diacritics(text)
    return DiacriticReport(
        text=text,
        diacritic_count=diac,
        total_letters=letters,
        preservation_score=diacritic_preservation_score(text),
        suspicious_words=suspicious,
    )


def _strip_punct(word: str) -> str:
    return "".join(ch for ch in word if ch.isalpha())


# A small allow-list of very common Slovak words that have no diacritics.
_PLAIN_SLOVAK_WORDS = {
    "a", "aj", "ale", "ani", "aspon", "atd", "bez", "cez", "co", "ci",
    "do", "du", "e", "este", "eu", "ho", "i", "ich", "ili", "im", "in",
    "is", "it", "je", "ju", "k", "ke", "ku", "la", "le", "ma", "me", "mi",
    "mu", "na", "ne", "ni", "no", "nu", "o", "od", "okolo", "om",
    "on", "os", "ot", "ou", "ov", "pa", "pe", "po", "pod", "pre", "pri",
    "pro", "proti", "rad", "s", "se", "si", "so", "sub", "svoje",
    "ta", "te", "ti", "to", "tu", "u", "uz", "v", "va", "ve", "vi", "vo", "vs",
    "vy", "z", "za", "ze", "zi", "zo", "zto", "nad", "medzi", "pomocou",
    "dva", "dve", "tri", "styri", "desat", "sto", "tisic", "milion",
    "den", "noc", "cas", "svet", "zivot", "voda", "zem", "dom", "mesto",
    "kniha", "strom", "slnko", "mesiac", "hviezda", "pes", "krava", "kon", "ovca",
    "vela", "malo", "velky", "maly", "novy", "stary", "dobry", "zly",
    "prvy", "posledny", "dalsi", "svoj", "tvoj", "jeho", "jej", "ich",
    "toto", "ten", "tie", "tí", "take", "dnes", "vtedy", "potom",
    "preto", "tak", "kto", "kedy", "kde", "ako", "preco",
}


def _analyze_word(word: str) -> SuspiciousWord | None:
    """Flag a word that likely lost diacritics."""
    if not word:
        return None
    has_target = any(ch in DIACRITIC_CONFUSIONS for ch in word)
    if not has_target:
        return None
    if word.lower() in _PLAIN_SLOVAK_WORDS:
        return None
    suggestion = _suggest_diacritics(word)
    return SuspiciousWord(word=word, reason="possible lost diacritic(s)", suggestion=suggestion)


def _suggest_diacritics(word: str) -> str:
    """Propose a diacritic variant using a small context-free
    substitution table of common Slovak words.

    This is a *suggestion* for the review UI, never auto-applied.
    """
    common = {
        "nejaky": "nejaký", "nejake": "nejaké",
        "pretoze": "pretože", "este": "ešte",
        "vela": "veľa", "zivot": "život", "cas": "čas",
        "den": "deň", "ludia": "ľudia", "clovek": "človek",
        "vsetko": "všetko", "vsetci": "všetci", "vsetky": "všetky",
        "pricom": "pričom", "takze": "takže", "preco": "prečo",
        "aj": "aj", "toto": "toto", "kto": "kto", "kedy": "kedy",
        "kde": "kde", "ako": "ako", "tak": "tak", "preto": "preto",
        "dva": "dva", "dve": "dve", "tri": "tri", "styri": "štyri",
        "svoj": "svoj", "svoje": "svoje", "svoja": "svoja", "svoju": "svoju",
        "dnes": "dnes", "vtedy": "vtedy", "potom": "potom",
        "zivoty": "životy", "prichod": "príchod", "odchod": "odchod",
        "urok": "úrok", "matka": "matka", "otec": "otec",
        "brat": "brat", "sestra": "sestra", "priatel": "priateľ",
        "chlap": "chlap", "zena": "žena", "dievca": "dievča",
        "mesto": "mesto", "dedina": "dedina", "krajina": "krajina",
        "svet": "svet", "kniha": "kniha", "list": "list",
        "strom": "strom", "kvietok": "kvietok", "trava": "trava",
        "les": "les", "hora": "hora", "rieka": "rieka", "jazero": "jazero",
        "more": "more", "plaz": "plaz", "ryba": "ryba", "vtak": "vtak",
        "pes": "pes", "macka": "mačka", "krava": "krava", "kon": "koň",
        "ovca": "ovca", "koza": "koza", "prasa": "prasa",
    }
    return common.get(word.lower(), "")


def page_diacritic_report(page: OCRPage) -> DiacriticReport:
    """Analyze a full OCRPage for diacritic quality."""
    return analyze_text(page.raw_text)


def low_diacritic_words(page: OCRPage) -> list[OCRWord]:
    """Return words that likely lost diacritics."""
    out: list[OCRWord] = []
    for w in page.all_words():
        clean = _strip_punct(w.text)
        if clean and _analyze_word(clean):
            out.append(w)
    return out


def normalize_nfkc(text: str) -> str:
    """Unicode NFC normalization (preserves Slovak diacritics)."""
    return unicodedata.normalize("NFC", text)
