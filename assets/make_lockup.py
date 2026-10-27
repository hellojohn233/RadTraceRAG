#!/usr/bin/env python3
"""用 logo 图合成横版 lockup（图标 + 字标）。

    python make_lockup.py <图标png> <输出png> [深/浅]

深色版：深底 + 米白字；浅色版：米白底 + 墨色字。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = (22, 32, 46)
PAPER = (251, 250, 247)
ACCENT = (192, 71, 43)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> None:
    icon = Image.open(sys.argv[1]).convert("RGBA")
    out = Path(sys.argv[2])
    dark = len(sys.argv) < 4 or sys.argv[3] != "light"

    W, H = 1120, 250
    base = Image.new("RGBA", (W, H), (INK + (255,)) if dark else (PAPER + (255,)))
    d = ImageDraw.Draw(base)

    size = 196
    icon = icon.resize((size, size), Image.LANCZOS)
    base.alpha_composite(icon, (40, (H - size) // 2))

    x = 40 + size + 46
    title_font = load_font(72)
    tag_font = load_font(19)

    title = "RadTrace"
    d.text((x, 62), title, font=title_font, fill=PAPER if dark else INK)
    tw = d.textlength(title, font=title_font)
    d.text((x + tw, 62), "RAG", font=title_font, fill=ACCENT)

    d.text(
        (x + 3, 162),
        "TRACEABLE MULTIMODAL RAG",
        font=tag_font,
        fill=(138, 148, 162) if dark else (138, 148, 162),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
