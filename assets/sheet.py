#!/usr/bin/env python3
"""候选 logo 联系表。

一屏看全每个候选的四种状态：
  1. 大图（造型、比例、留白）
  2. 32px 缩略（favicon 场景还认不认得出）
  3. 纯黑单色 32px（印刷/单色场景还能不能用）
  4. 与 Georgia 衬线字标并排（会不会打架）

用法：python sheet.py <输出png> <候选名=图片路径> ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = (22, 32, 46)
PAPER = (251, 250, 247)
RULE = (225, 221, 212)
GEORGIA = "/System/Library/Fonts/Supplemental/Georgia.ttf"
GEORGIA_B = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"


def crop_square(im: Image.Image, pad: float = 0.1) -> Image.Image:
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    h, w, _ = a.shape
    cor = np.concatenate(
        [
            a[:6, :6].reshape(-1, 3),
            a[:6, -6:].reshape(-1, 3),
            a[-6:, :6].reshape(-1, 3),
            a[-6:, -6:].reshape(-1, 3),
        ]
    )
    bg = np.median(cor, axis=0)
    mask = np.abs(a - bg).sum(axis=2) > 26
    if not mask.any():
        return im.convert("RGB")
    r = np.where(mask.any(axis=1))[0]
    c = np.where(mask.any(axis=0))[0]
    l, t, rr, bb = c[0], r[0], c[-1] + 1, r[-1] + 1
    s = int(max(rr - l, bb - t) * (1 + pad * 2))
    cx, cy = (l + rr) / 2, (t + bb) / 2
    x0 = max(0, min(int(cx - s / 2), w - s))
    y0 = max(0, min(int(cy - s / 2), h - s))
    s = min(s, w, h)
    return im.crop((x0, y0, x0 + s, y0 + s)).convert("RGB")


def to_mono(im: Image.Image) -> Image.Image:
    g = im.convert("L")
    return g.point(lambda v: 0 if v < 150 else 255, mode="L").convert("RGB")


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    out = Path(sys.argv[1])
    pairs = [a.split("=", 1) for a in sys.argv[2:]]

    CW, CH = 210, 300
    sheet = Image.new("RGB", (CW * len(pairs), CH + 40), PAPER)
    d = ImageDraw.Draw(sheet)
    f_lab = font("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    f_word = font(GEORGIA, 21)
    f_wordb = font(GEORGIA_B, 21)

    for i, (name, path) in enumerate(pairs):
        x = i * CW
        im = crop_square(Image.open(path))
        im.thumbnail((180, 180), Image.LANCZOS)
        sheet.paste(im, (x + (CW - im.width) // 2, 6))

        small = im.resize((32, 32), Image.LANCZOS)
        sheet.paste(small, (x + 26, 194))
        mono = to_mono(im).resize((32, 32), Image.LANCZOS)
        sheet.paste(mono, (x + 70, 194))

        # 与衬线字标并排
        mark = im.resize((30, 30), Image.LANCZOS)
        sheet.paste(mark, (x + 112, 195))
        d.text((x + 148, 200), "RadTrace", font=f_wordb, fill=INK)
        d.text((x + 148, 214), "RAG", font=f_word, fill=INK)

        d.line([(x + 8, 236), (x + CW - 8, 236)], fill=RULE, width=1)
        d.text((x + 8, 244), name[:22], fill=(90, 100, 112), font=f_lab)
        d.text((x + 8, 262), "大图 / 32px / 单色32px / 衬线并排", fill=(150, 156, 164), font=f_lab)

    sheet.save(out)
    print("saved", out, sheet.size)


if __name__ == "__main__":
    main()
