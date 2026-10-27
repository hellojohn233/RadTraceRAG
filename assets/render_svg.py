#!/usr/bin/env python3
"""把 SVG 渲染成 PNG（Chrome headless），用于评审和导出。

    python render_svg.py <svg文件...> --out <目录> [--size 512]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def render(svg: Path, out: Path, size: int, bg: str = "FBFAF7", color: str = "16202E") -> None:
    bg_css = "transparent" if bg.lower() == "transparent" else f"#{bg}"
    html = Path(tempfile.gettempdir()) / f"_r_{svg.stem}.html"
    html.write_text(
        f"""<html><body style="margin:0;background:{bg_css};color:#{color}">
<div style="width:{size}px;height:{size}px;display:flex;align-items:center;justify-content:center">
<img src="file://{svg.resolve()}" width="{size}" height="{size}">
</div></body></html>""",
        encoding="utf-8",
    )
    args = [
        CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
        f"--screenshot={out}", f"--window-size={size},{size}",
    ]
    if bg.lower() == "transparent":
        args.append("--default-background-color=00000000")
    args.append(f"file://{html}")
    subprocess.run(
        args,
        capture_output=True,
        check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svgs", nargs="+")
    ap.add_argument("--out", default="assets/png")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--bg", default="FBFAF7")
    ap.add_argument("--color", default="16202E")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for s in args.svgs:
        svg = Path(s)
        name = f"{args.prefix}{svg.stem}-{args.size}.png"
        render(svg, outdir / name, args.size, args.bg, args.color)
        print("rendered", outdir / name)


if __name__ == "__main__":
    main()
