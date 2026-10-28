#!/usr/bin/env python3
"""把定稿 logo 接进现有资产位。

换掉这些文件（文件名不变，UI 和 README 不用改代码）：
  assets/png/logo-mark-{32,64,128,256,512,1024}.png
  assets/png/logo-mark-favicon-{16,32,180}.png
  assets/png/logo-lockup.png          深底版
  assets/png/logo-lockup-light.png    浅底版

用法：python install_logo.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from render_svg import render  # noqa: E402

MARK = ROOT / "assets" / "logo" / "svg" / "01-window-level.svg"
LOCKUP_SVG = ROOT / "assets" / "logo" / "final" / "radtracerag-lockup.svg"
PNG = ROOT / "assets" / "png"
PAPER = (251, 250, 247)
INK = (22, 32, 46)

# 标志内容在 100x100 画布里占 x 6..93 / y 22..84，取正方形裁切做图标
ICON_VIEWBOX = "6 9.5 87 87"


def write_icon_svg() -> Path:
    src = MARK.read_text(encoding="utf-8")
    src = re.sub(r'viewBox="0 0 100 100"', f'viewBox="{ICON_VIEWBOX}"', src, count=1)
    out = Path(tempfile.gettempdir()) / "_icon-window-level.svg"
    out.write_text(src, encoding="utf-8")
    return out


def render_lockup(dark: bool) -> None:
    svg = LOCKUP_SVG.read_text(encoding="utf-8")
    if dark:
        svg = svg.replace("#16202E", "#FBFAF7")
    tmp = Path(tempfile.gettempdir()) / f"_lock_{'dark' if dark else 'light'}.svg"
    tmp.write_text(svg, encoding="utf-8")

    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    w, h = float(m.group(1)), float(m.group(2))
    px_w = 2400
    px_h = round(px_w * h / w)

    html = Path(tempfile.gettempdir()) / f"_lock_{px_w}.html"
    html.write_text(
        f'<html><body style="margin:0;background:transparent">'
        f'<img src="file://{tmp.resolve()}" width="{px_w}" height="{px_h}"></body></html>',
        encoding="utf-8",
    )
    png = Path(tempfile.gettempdir()) / f"_lock_{'dark' if dark else 'light'}.png"
    subprocess.run(
        [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--screenshot={png}", f"--window-size={px_w},{px_h}",
            "--default-background-color=00000000", f"file://{html}",
        ],
        capture_output=True,
        check=False,
    )

    im = Image.open(png).convert("RGBA")
    flat = Image.new("RGB", im.size, INK if dark else PAPER)
    flat.paste(im, mask=im.split()[-1])
    name = "logo-lockup.png" if dark else "logo-lockup-light.png"
    flat.save(PNG / name, optimize=True)
    print("saved", PNG / name, flat.size)


def main() -> None:
    icon = write_icon_svg()
    for size in (32, 64, 128, 256, 512, 1024):
        render(icon, PNG / f"logo-mark-{size}.png", size, bg="transparent", color="16202E", mono=False)
        print("saved", PNG / f"logo-mark-{size}.png")
    for size in (16, 32, 180):
        render(icon, PNG / f"logo-mark-favicon-{size}.png", size, bg="transparent", color="16202E", mono=False)
        print("saved", PNG / f"logo-mark-favicon-{size}.png")

    render_lockup(dark=True)
    render_lockup(dark=False)


if __name__ == "__main__":
    main()
