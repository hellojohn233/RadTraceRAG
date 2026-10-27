#!/usr/bin/env python3
"""导出候选 logo 的生产尺寸 PNG + 横版 lockup。

    python export_logo_sizes.py

输出：
  assets/logo/png/<id>-<size>.png   size in 1024/512/256/64/32/16
  assets/logo/lockup/<id>-lockup.png 横版（字标 + 图标）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from render_svg import render  # noqa: E402

ROOT = Path(__file__).parent.parent
LOGO = ROOT / "assets" / "logo"
SIZES = [1024, 256, 64, 32, 16]
LOCKUP_IDS = ["01-monogram-r", "05-trace-spiral"]


def main() -> None:
    png_dir = LOGO / "png"
    for svg in sorted((LOGO / "svg").glob("*.svg")):
        vid = svg.stem
        for size in SIZES:
            out = png_dir / f"{vid}-{size}.png"
            render(svg, out, size, bg="transparent", color="16202E")
        print("sizes ->", vid)

    lock_dir = LOGO / "lockup"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for vid in LOCKUP_IDS:
        src = png_dir / f"{vid}-512.png"
        out = lock_dir / f"{vid}-lockup.png"
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "assets" / "make_lockup.py"), str(src), str(out), "light"],
            check=True,
        )


if __name__ == "__main__":
    main()
