#!/usr/bin/env python3
"""第四波：符号化的三个新方向。

  q1-spiral   一条线绕回中心，中心一个实点 —— 溯源
  q2-taper    一笔由细到粗的锥形墨迹 —— 落笔成痕
  q3-leader   实点 + 折线引出 + 方块端点 —— 标注引线
  q4-halfring 两段错位的粗弧 —— 断层/两次观察

输出：assets/marks5/*.svg
"""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).parent / "marks5"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n',
        encoding="utf-8",
    )
    print("wrote", OUT / name)


def spiral_points(cx, cy, r0, r1, a0, sweep, n=140):
    pts = []
    for i in range(n + 1):
        t = i / n
        th = math.radians(a0) + sweep * t
        r = r0 + (r1 - r0) * t
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts


def poly_path(pts) -> str:
    d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    for x, y in pts[1:]:
        d += f" L {x:.2f} {y:.2f}"
    return d


def bezier(p0, p1, p2, p3, n=90):
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def tapered_stroke(pts, w_start, w_end, ease=1.0):
    """把一条中心线变成由细到粗的实心笔画。"""
    n = len(pts) - 1
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        t = i / n
        w = (w_start + (w_end - w_start) * (t**ease)) / 2
        if i == 0:
            dx, dy = pts[1][0] - x, pts[1][1] - y
        elif i == n:
            dx, dy = x - pts[-2][0], y - pts[-2][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        ln = math.hypot(dx, dy) or 1
        nx, ny = -dy / ln, dx / ln
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    outline = left + right[::-1]
    d = poly_path(outline) + " Z"
    return d


def main() -> None:
    # ---- q1 螺旋溯源 ----
    cx, cy = 50, 50
    a0, sweep = 118, 5.45
    r0, r1 = 4.6, 32.5
    pts = spiral_points(cx, cy, r0, r1, a0, sweep)
    ex, ey = pts[0]
    write(
        "q1-spiral.svg",
        f"""  <!-- 一条线绕回中心，中心是那个被追到的原点 -->
  <path d="{poly_path(pts)}" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round"/>
  <circle cx="{ex:.2f}" cy="{ey:.2f}" r="4.6" fill="currentColor"/>""",
    )

    # ---- q2 锥形墨迹 ----
    pts = bezier((14, 82), (26, 80), (34, 34), (80, 20))
    d = tapered_stroke(pts, 1.6, 13.0, ease=1.4)
    write(
        "q2-taper.svg",
        f'  <!-- 一笔由细到粗 -->\n  <path d="{d}" fill="currentColor"/>',
    )

    # ---- q3 引线标注 ----
    write(
        "q3-leader.svg",
        """  <!-- 实点 + 折线引出 + 方块端点 -->
  <circle cx="24" cy="74" r="7.5" fill="currentColor"/>
  <path d="M 29 69 L 52 48 L 74 48" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="72" y="40" width="16" height="16" rx="2.5" fill="currentColor"/>""",
    )

    # ---- q4 两段错位粗弧 ----
    write(
        "q4-halfring.svg",
        """  <!-- 两段错位的粗弧 -->
  <g fill="none" stroke="currentColor" stroke-linecap="round">
    <path d="M 22 74 A 30 30 0 1 1 70 34" stroke-width="8.5"/>
    <path d="M 76 78 A 20 20 0 0 1 34 76" stroke-width="4" opacity="0.45"/>
  </g>""",
    )


if __name__ == "__main__":
    main()
