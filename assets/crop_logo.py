#!/usr/bin/env python3
"""把 verge 出的 3:4 图裁成正方形的 logo 素材。

AI 出图是 3:4 画幅，logo 居中且四周留白。这里自动找到非背景像素的边界，
裁成正方形（带一点余量），再导出各尺寸。

用法：python crop_logo.py <输入图> <输出目录> <前缀>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SIZES = [1024, 512, 256, 128, 64, 32]
FAVICON = [180, 32, 16]


def content_box(im: Image.Image) -> tuple[int, int, int, int]:
    """找出非背景内容的包围盒。背景取四角像素的中位色。"""
    rgb = im.convert("RGB")
    a = np.asarray(rgb).astype(np.int16)
    h, w, _ = a.shape
    corners = np.concatenate(
        [a[:6, :6].reshape(-1, 3), a[:6, -6:].reshape(-1, 3), a[-6:, :6].reshape(-1, 3), a[-6:, -6:].reshape(-1, 3)]
    )
    bg = np.median(corners, axis=0)
    diff = np.abs(a - bg).sum(axis=2)
    mask = diff > 26
    if not mask.any():
        return 0, 0, w, h
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def square_crop(im: Image.Image, pad_ratio: float = 0.10) -> Image.Image:
    left, top, right, bottom = content_box(im)
    size = max(right - left, bottom - top)
    size = int(size * (1 + pad_ratio * 2))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = max(0, min(x0, im.width - size))
    y0 = max(0, min(y0, im.height - size))
    if size > im.height:
        size = im.height
        y0 = 0
    return im.crop((x0, y0, x0 + size, y0 + size))


def main() -> None:
    src = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else "logo"
    outdir.mkdir(parents=True, exist_ok=True)

    im = Image.open(src)
    sq = square_crop(im)
    # 背景统一成米白纸色，和界面一致
    bg = Image.new("RGB", sq.size, (251, 250, 247))
    bg.paste(sq, (0, 0), sq if sq.mode == "RGBA" else None)
    sq = bg

    for s in SIZES:
        sq.resize((s, s), Image.LANCZOS).save(outdir / f"{prefix}-{s}.png")
    for s in FAVICON:
        sq.resize((s, s), Image.LANCZOS).save(outdir / f"{prefix}-favicon-{s}.png")
    print(f"{outdir}/{prefix}-*.png  ({len(SIZES) + len(FAVICON)} 张, 源 {im.size})")


if __name__ == "__main__":
    main()
