#!/usr/bin/env python3
"""第三波：用 Georgia Bold 真字形做几个成品级候选。

  p1-semicolon        分号本体（引用列表）
  p2-semicolon-square 分号，但句点改成实心方块
  p3-r                R 单体字标
  p4-r-line           R 后面横穿一条细线（溯源线）
  p5-gapring          环缺 + 节点
  p6-comma            逗号本体
  p7-rt               RT 合字（T 横杠压在 R 的顶线上）

输出：assets/marks4/*.svg
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from glyph import glyph_path  # noqa: E402

GEORGIA_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
OUT = Path(__file__).parent / "marks4"


def raw(ch: str):
    d, x0, y0, x1, y1 = glyph_path(GEORGIA_BOLD, ch)
    return d, x0, y0, x1, y1


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n',
        encoding="utf-8",
    )
    print("wrote", OUT / name)


def group(d: str, tx: float, ty: float, s: float, fill: str = "currentColor") -> str:
    return (
        f'<g transform="translate({tx:.3f},{ty:.3f}) scale({s:.5f})">'
        f'<path d="{d}" fill="{fill}"/></g>'
    )


def main() -> None:
    # ---------- p1 分号 ----------
    d_sc, sx0, sy0, sx1, sy1 = raw(";")
    h = sy1 - sy0
    s = 62 / h
    # 居中：水平居中，整体顶到 y=20
    tx = 50 - (sx0 + sx1) / 2 * s
    ty = 20 - sy0 * s
    write(
        "p1-semicolon.svg",
        f"  <!-- 分号：引文列表的记号 -->\n  {group(d_sc, tx, ty, s)}",
    )

    # ---------- p2 分号 + 方点 ----------
    d_dt, dx0, dy0, dx1, dy1 = raw(".")
    dot_cx = (dx0 + dx1) / 2
    dot_cy = (dy0 + dy1) / 2
    dot_w = (dx1 - dx0) * s
    dot_h = (dy1 - dy0) * s
    # 方块：与句点同中心，边长取句点宽度
    sq = dot_w * 1.02
    cx = dot_cx * s + tx
    cy = dot_cy * s + ty
    write(
        "p2-semicolon-square.svg",
        f"""  <!-- 分号，句点换成实心方块 -->
  <defs>
    <mask id="nod">
      <rect width="100" height="100" fill="#fff"/>
      <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{dot_w / 2 * 1.15:.2f}" fill="#000"/>
    </mask>
  </defs>
  <g mask="url(#nod)">
    {group(d_sc, tx, ty, s)}
  </g>
  <rect x="{cx - sq / 2:.2f}" y="{cy - sq / 2:.2f}" width="{sq:.2f}" height="{sq:.2f}"
        rx="{sq * 0.14:.2f}" fill="currentColor"/>""",
    )

    # ---------- p3 R ----------
    d_r, rx0, ry0, rx1, ry1 = raw("R")
    cap = 66
    s = cap / (ry1 - ry0)
    tx = 50 - (rx0 + rx1) / 2 * s
    ty = 16 - ry0 * s
    write(
        "p3-r.svg",
        f"  <!-- R 单体字标 -->\n  {group(d_r, tx, ty, s)}",
    )

    # ---------- p4 R + 背后的溯源线 ----------
    cap = 62
    s = cap / (ry1 - ry0)
    tx = 8 - rx0 * s
    ty = 22 - ry0 * s
    # 细线从左下穿到右上，左端收一个小实点
    write(
        "p4-r-line.svg",
        f"""  <!-- 溯源线从字标背后穿过 -->
  <line x1="8" y1="84" x2="94" y2="26" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/>
  <circle cx="8" cy="84" r="4" fill="currentColor"/>
  {group(d_r, tx, ty, s)}""",
    )

    # ---------- p5 环缺 + 节点 ----------
    r = 30
    cx, cy = 48, 52
    circ = 2 * math.pi * r
    gap_deg = 46
    gap_len = circ * gap_deg / 360
    dash = circ - gap_len
    ang = math.radians(-40)
    nx = cx + (r + 10) * math.cos(ang)
    ny = cy + (r + 10) * math.sin(ang)
    write(
        "p5-gapring.svg",
        f"""  <!-- 环开一道缺口，节点从缺口探出 -->
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="currentColor" stroke-width="8.5"
          stroke-dasharray="{dash:.1f} {gap_len:.1f}" transform="rotate({-40 + (360 - gap_deg) / 2:.1f} {cx} {cy})"/>
  <circle cx="{nx:.1f}" cy="{ny:.1f}" r="6.5" fill="currentColor"/>""",
    )

    # ---------- p6 逗号 ----------
    d_cm, cx0, cy0, cx1, cy1 = raw(",")
    s = 64 / (cy1 - cy0)
    tx = 50 - (cx0 + cx1) / 2 * s
    ty = 20 - cy0 * s
    write(
        "p6-comma.svg",
        f"  <!-- 引用逗号 -->\n  {group(d_cm, tx, ty, s)}",
    )

    # ---------- p7 RT 合字 ----------
    cap = 58
    s = cap / (ry1 - ry0)
    tx = 22 - rx0 * s
    ty = 30 - ry0 * s
    write(
        "p7-rt.svg",
        f"""  <!-- RT 合字：T 横杠压在 R 的顶线上 -->
  <rect x="4" y="21" width="92" height="6.5" rx="1.2" fill="currentColor"/>
  {group(d_r, tx, ty, s)}""",
    )


if __name__ == "__main__":
    main()
