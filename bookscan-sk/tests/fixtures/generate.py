#!/usr/bin/env python3
"""Generate synthetic Slovak test pages for testing and benchmarking.

Creates images containing Slovak sentences with full diacritics,
rendered with DejaVu Sans (which covers the Slovak alphabet).
"""
from __future__ import annotations

import os
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# The canonical Slovak test sentence from the spec.
CANONICAL = "Červená žaba ďalej šla po lúke, kým Ľuboš čítal krásnu knihu."

SLOVAK_SENTENCES = [
    "Červená žaba ďalej šla po lúke, kým Ľuboš čítal krásnu knihu.",
    "Slovenská republika má krásnú prírodu a historické mestá.",
    "Včera bolo nádherné počasie a my sme išli na výlet do hôr.",
    "Šedý štít maďarského ôzvača nechal ľudí ľakúť sa.",
    "Příliš žluťoučký kůň úpěl ďábelské ódy (testovacia veta).",
    "Mojí rodičia chodia každý deň na prechádzku do parku.",
    "Výsledky skúšky boli úžasné a všetci boli šťastní.",
    "Teraz je čas ísť domov a odpočívať si po náročnom dni.",
    "Kniha o slovenskej histórii je veľmi zaujímavá a poučná.",
    "Děti sa hrali na ihrisku a smiali sa celý popoludní.",
]

DIACRITIC_STRESS = [
    "ľ ĺ ň ŕ š ť ž á ä č ď é í ó ô ú ý",
    "Ľ Ĺ Ň Ŕ Š Ť Ž Á Ä Č Ď É Í Ó Ô Ú Ý",
    "nejaký nejaká nejaké pretože ešte veľa život čas deň",
    "ľudia človek všetko všetci všetky pričom takže prečo",
    "mačka koň ovca žena dievča príchod odchod úrok",
]

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_PATH
    return ImageFont.truetype(path, size)


def make_page(
    path: str,
    sentences: Optional[list[str]] = None,
    width: int = 1240,
    height: int = 1754,  # A4 at 150 DPI
    font_size: int = 46,
    margin: int = 120,
    title: Optional[str] = None,
    noise: bool = False,
) -> str:
    """Render a synthetic Slovak page to an image file."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    title_font = _font(font_size + 8, bold=True)

    y = margin
    if title:
        draw.text((margin, y), title, font=title_font, fill="black")
        y += int(font_size * 2.2)

    for s in (sentences or SLOVAK_SENTENCES):
        # Word-wrap.
        words = s.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            bbox = draw.textbbox((margin, y), test, font=font)
            if bbox[2] > width - margin:
                draw.text((margin, y), line, font=font, fill="black")
                y += int(font_size * 1.6)
                line = w
            else:
                line = test
        if line:
            draw.text((margin, y), line, font=font, fill="black")
            y += int(font_size * 1.9)
        y += int(font_size * 0.4)

    if noise:
        # Add mild Gaussian-like noise to simulate a photograph.
        import numpy as np

        arr = np.array(img).astype(np.int16)
        n = np.random.normal(0, 6, arr.shape)
        arr = np.clip(arr + n, 0, 255).astype("uint8")
        img = Image.fromarray(arr)

    img.save(path)
    return path


def make_fixture_dir(output_dir: str, count: int = 5) -> list[str]:
    """Generate a directory of synthetic Slovak pages."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    for i in range(count):
        title = f"Strana {i + 1}"
        p = os.path.join(output_dir, f"page{i + 1}.png")
        make_page(p, title=title, noise=(i % 2 == 0))
        paths.append(p)
    return paths


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sk_fixtures"
    paths = make_fixture_dir(out, 5)
    print(f"Generated {len(paths)} pages in {out}")
    for p in paths:
        print(" ", p)
