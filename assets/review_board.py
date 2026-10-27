#!/usr/bin/env python3
"""logo 候选评审板。

按三条硬判据排版，不靠"好不好看"：
  1. 大图 —— 造型和比例
  2. 32px 缩略 —— favicon 场景还认不认得出
  3. 纯黑单色版 —— 印刷/单色场景还能不能用

用法：python review_board.py <输出png> <候选名=图片路径> ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def crop_square(im: Image.Image, pad: float = 0.12) -> Image.Image:
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
    """转纯黑：按亮度做二值，模拟单色印刷。"""
    g = im.convert("L")
    return g.point(lambda v: 0 if v < 150 else 255, mode="L").convert("RGB")


def main() -> None:
    out = Path(sys.argv[1])
    pairs = [a.split("=", 1) for a in sys.argv[2:]]

    CW, CH = 200, 300
    cols = len(pairs)
    sheet = Image.new("RGB", (cols * CW, CH + 26), (235, 233, 227))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for i, (name, path) in enumerate(pairs):
        im = crop_square(Image.open(path))
        im.thumbnail((170, 170), Image.LANCZOS)
        x = i * CW
        sheet.paste(im, (x + (CW - im.width) // 2, 8))
        small = im.resize((32, 32), Image.LANCZOS)
        sheet.paste(small, (x + (CW - 32) // 2, 190))
        mono = to_mono(im).resize((32, 32), Image.LANCZOS)
        sheet.paste(mono, (x + (CW - 32) // 2, 232))
        d.text((x + 8, CH - 2), name[:16], fill=(60, 70, 82), font=font)

    sheet.save(out)
    print("saved", out, sheet.size)


if __name__ == "__main__":
    main()
