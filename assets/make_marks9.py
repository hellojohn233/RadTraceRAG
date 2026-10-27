#!/usr/bin/env python3
"""影像派 v2：四个站得住的方向。

  m2-window     窗宽窗位：像素值直方图 + 朱砂窗位框
  m7-lesion     病灶：取景角标圈住的那一点
  m8-trace      检索轨迹：一路走到命中的朱砂点
  m9-voxel      影像矩阵里被标出来的那一格

输出：assets/marks9/*.svg
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "marks9"
ACCENT = "#C0472B"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n',
        encoding="utf-8",
    )
    print("wrote", OUT / name)


def main() -> None:
    write(
        "m2-window.svg",
        f"""  <!-- 像素值直方图 + 朱砂窗位框：窗住哪一段，就是显示哪一段 -->
  <g fill="currentColor">
    <rect x="6" y="58" width="9" height="22" rx="1"/>
    <rect x="19" y="46" width="9" height="34" rx="1"/>
    <rect x="32" y="36" width="9" height="44" rx="1"/>
    <rect x="45" y="28" width="9" height="52" rx="1"/>
    <rect x="58" y="34" width="9" height="46" rx="1"/>
    <rect x="71" y="48" width="9" height="32" rx="1"/>
    <rect x="84" y="62" width="9" height="18" rx="1"/>
  </g>
  <rect x="28" y="22" width="42" height="62" rx="4" fill="none" stroke="{ACCENT}" stroke-width="2.6"/>""",
    )

    write(
        "m7-lesion.svg",
        f"""  <!-- 阅片时圈出来的那一点 -->
  <g fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round">
    <path d="M 20 42 L 20 22 L 40 22"/>
    <path d="M 20 62 L 20 82 L 40 82"/>
    <path d="M 64 82 L 82 82 L 82 64"/>
    <path d="M 82 40 L 82 22 L 66 22"/>
  </g>
  <circle cx="48" cy="51" r="6" fill="{ACCENT}"/>""",
    )

    write(
        "m8-trace.svg",
        f"""  <!-- 一路走到命中：两个墨点，一个朱砂点 -->
  <path d="M 16 80 L 16 50 L 44 50 L 44 26 L 66 26" fill="none" stroke="currentColor"
        stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="16" cy="80" r="3.4" fill="currentColor"/>
  <circle cx="44" cy="50" r="4.2" fill="currentColor"/>
  <circle cx="66" cy="26" r="5.8" fill="{ACCENT}"/>""",
    )

    cells = []
    for r in range(3):
        for c in range(3):
            cells.append(
                f'<rect x="{16 + c * 20}" y="{16 + r * 20}" width="15" height="15" rx="2.5"/>'
            )
    write(
        "m9-voxel.svg",
        f"""  <!-- 影像矩阵里被标出来的那一格 -->
  <g fill="currentColor">
{chr(10).join("    " + c for c in cells)}
  </g>
  <circle cx="46.5" cy="46.5" r="5.5" fill="{ACCENT}"/>
  <line x1="52" y1="52" x2="76" y2="76" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/>
  <rect x="76" y="76" width="14" height="14" rx="3" fill="currentColor"/>""",
    )


if __name__ == "__main__":
    main()
