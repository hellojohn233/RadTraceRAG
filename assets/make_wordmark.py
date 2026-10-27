#!/usr/bin/env python3
"""排几版纯字标（wordmark）。

纯字标不是生图，是排版——用本机 Georgia 衬线排，字母拼写不会有错，
单色印刷也天然成立。

用法：python make_wordmark.py <输出目录>
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = (22, 32, 46)
PAPER = (251, 250, 247)
ACCENT = (192, 71, 43)
MUTE = (138, 148, 162)

GEORGIA_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
GEORGIA = "/System/Library/Fonts/Supplemental/Georgia.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def variant_1(path: Path) -> None:
    """RadTrace 墨色 + RAG 朱砂，紧字距，最基本的做法。"""
    W, H = 1400, 260
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    f = font(GEORGIA_BOLD, 118)
    x, y = 60, 168
    for part, col in [("RadTrace", INK), ("RAG", ACCENT)]:
        d.text((x, y), part, font=f, fill=col)
        x += d.textlength(part, font=f)
    im.save(path)


def variant_2(path: Path) -> None:
    """全大写 + 宽字距，像期刊刊头。"""
    W, H = 1500, 240
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    f = font(GEORGIA_BOLD, 86)
    x, y = 60, 150
    track = 9
    for part, col in [("RADTRACE", INK), ("RAG", ACCENT)]:
        for ch in part:
            d.text((x, y), ch, font=f, fill=col)
            x += d.textlength(ch, font=f) + track
    im.save(path)


def variant_3(path: Path) -> None:
    """Rad 用衬线、TraceRAG 用无衬线粗体，对比。"""
    helv = "/System/Library/Fonts/HelveticaNeue.ttc"
    W, H = 1500, 250
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    f1 = font(GEORGIA_BOLD, 122)
    f2 = ImageFont.truetype(helv, 112, index=1)
    x, y = 60, 168
    for part, col, ff in [
        ("Rad", INK, f1),
        ("Trace", INK, f2),
        ("RAG", ACCENT, f2),
    ]:
        d.text((x, y), part, font=ff, fill=col)
        x += d.textlength(part, font=ff)
    im.save(path)


def variant_4(path: Path) -> None:
    """衬线主标 + 下方小字说明，最接近论文页眉。"""
    W, H = 1500, 320
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    f = font(GEORGIA_BOLD, 104)
    d.text((60, 90), "RadTrace", font=f, fill=INK)
    x = 60 + d.textlength("RadTrace", font=f)
    d.text((x, 90), "RAG", font=f, fill=ACCENT)
    tag = font(GEORGIA, 22)
    d.text((64, 240), "TRACEABLE MULTIMODAL RAG", font=tag, fill=MUTE)
    im.save(path)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    for i, fn in enumerate([variant_1, variant_2, variant_3, variant_4], 1):
        p = out / f"wordmark-{i}.png"
        fn(p)
        print("saved", p)
