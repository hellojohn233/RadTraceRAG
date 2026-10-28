#!/usr/bin/env python3
"""把定稿 lockup 导出成高分辨率 PNG / JPEG。

    python export_final_logo.py [--width 2400]

JPEG 铺纸白底 #FBFAF7（没有 alpha），PNG 保留透明。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
OUT = ROOT / "assets" / "logo" / "final"
SVG = OUT / "radtracerag-lockup.svg"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAPER = (251, 250, 247)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=2400)
    args = ap.parse_args()

    svg = SVG.read_text(encoding="utf-8")
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    w, h = float(m.group(1)), float(m.group(2))
    px_w = args.width
    px_h = round(px_w * h / w)

    html = Path(tempfile.gettempdir()) / "_lockup.html"
    html.write_text(
        f'<html><body style="margin:0;background:transparent">'
        f'<img src="file://{SVG.resolve()}" width="{px_w}" height="{px_h}"></body></html>',
        encoding="utf-8",
    )
    png = OUT / f"radtracerag-lockup-{px_w}.png"
    subprocess.run(
        [
            CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--screenshot={png}", f"--window-size={px_w},{px_h}",
            "--default-background-color=00000000",
            f"file://{html}",
        ],
        capture_output=True,
        check=False,
    )

    im = Image.open(png).convert("RGBA")
    for name, bg in (("paper", PAPER), ("white", (255, 255, 255))):
        flat = Image.new("RGB", im.size, bg)
        flat.paste(im, mask=im.split()[-1])
        jpg = OUT / f"radtracerag-lockup-{px_w}-{name}.jpg"
        flat.save(jpg, "JPEG", quality=95, subsampling=0, optimize=True)
        print("saved", jpg, flat.size)
    print("saved", png, im.size)


if __name__ == "__main__":
    main()
