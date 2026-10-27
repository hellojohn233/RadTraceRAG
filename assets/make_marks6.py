#!/usr/bin/env python3
"""第五波：螺旋溯源（起始角调过，不再像数字 9）。

  q1b-spiral  一条线绕回中心，中心一个实点 —— 溯源

输出：assets/marks6/*.svg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from glyph import glyph_path  # noqa: E402

GEORGIA_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
OUT = Path(__file__).parent / "marks6"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n',
        encoding="utf-8",
    )
    print("wrote", OUT / name)


def main() -> None:
    d_r, rx0, ry0, rx1, ry1 = glyph_path(GEORGIA_BOLD, "R")
    cap = 74
    s = cap / (ry1 - ry0)
    tx = 8.7 - rx0 * s
    ty = 12 - ry0 * s

    # 螺旋：起始角换到左下，别读成 9
    cx, cy = 50, 50
    a0, sweep = 196, 5.45
    r0, r1 = 4.6, 32.0
    pts = []
    for i in range(141):
        t = i / 140
        th = math.radians(a0) + sweep * t
        r = r0 + (r1 - r0) * t
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)
    ex, ey = pts[0]
    write(
        "q1b-spiral.svg",
        f"""  <!-- 一条线绕回中心，中心是那个被追到的原点 -->
  <path d="{d}" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round"/>
  <circle cx="{ex:.2f}" cy="{ey:.2f}" r="4.6" fill="currentColor"/>""",
    )


if __name__ == "__main__":
    main()
