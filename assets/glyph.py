#!/usr/bin/env python3
"""从 TTF 提取字形轮廓成 SVG path。

用真实衬线字体的字形做单体字标，比手画路径精细得多。

    python glyph.py R T --font "Georgia Bold.ttf" --size 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

FONTS = {
    "georgia-bold": "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "georgia": "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "georgia-italic": "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
}


def glyph_path(font_path: str, ch: str) -> tuple[str, float, float, float, float]:
    """返回 (path_d, xmin, ymin, xmax, ymax)，坐标已翻到 y 向下的 SVG 空间。"""
    font = TTFont(font_path)
    upm = font["head"].unitsPerEm
    scale = 1.0 / upm
    cmap = font.getBestCmap()
    name = cmap[ord(ch)]
    glyphset = font.getGlyphSet()
    pen = SVGPathPen(glyphset)
    # 翻 y 轴：字体坐标 y 向上，SVG y 向下
    tpen = TransformPen(pen, (scale, 0, 0, -scale, 0, 0))
    glyphset[name].draw(tpen)
    bpen = BoundsPen(glyphset)
    glyphset[name].draw(TransformPen(bpen, (scale, 0, 0, -scale, 0, 0)))
    x0, y0, x1, y1 = bpen.bounds
    d = pen.getCommands()
    return d, x0, y0, x1, y1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chars", nargs="+")
    ap.add_argument("--font", default="georgia-bold")
    ap.add_argument("--size", type=float, default=100.0)
    args = ap.parse_args()

    font_path = FONTS.get(args.font, args.font)
    for ch in args.chars:
        d, x0, y0, x1, y1 = glyph_path(font_path, ch)
        w, h = x1 - x0, y1 - y0
        print(f"--- {ch} ---")
        print(f"d={d}")
        print(f"bbox=({x0:.4f},{y0:.4f})-({x1:.4f},{y1:.4f}) w={w:.4f} h={h:.4f}")


if __name__ == "__main__":
    main()
