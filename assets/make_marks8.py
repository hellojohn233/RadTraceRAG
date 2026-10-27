#!/usr/bin/env python3
"""影像派 v2：只保留站得住的四个方向，并按界面规范用朱砂点。

  m2-window     窗宽窗位：像素值直方图 + 朱砂窗位带
  m7-lesion     病灶标记：圈 + 朱砂发现点 + 指向证据的引线
  m8-trace      检索轨迹：两个墨点 + 一个朱砂命中点，中间是折线
  m9-voxel      影像里的那一格：3x3 像素矩阵 + 朱砂发现 + 指向证据

输出：assets/marks8/*.svg
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "marks8"
ACCENT = "#C0472B"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n',
        encoding="utf-8",
    )
    print("wrote", OUT / name)


def main() -> None:
    # ---- 窗宽窗位 ----
    write(
        "m2-window.svg",
        f"""  <!-- 像素值直方图 + 窗位带 -->
  <g fill="currentColor">
    <rect x="6" y="58" width="9" height="22" rx="1"/>
    <rect x="19" y="46" width="9" height="34" rx="1"/>
    <rect x="32" y="36" width="9" height="44" rx="1"/>
    <rect x="45" y="28" width="9" height="52" rx="1"/>
    <rect x="58" y="34" width="9" height="46" rx="1"/>
    <rect x="71" y="48" width="9" height="32" rx="1"/>
    <rect x="84" y="62" width="9" height="18" rx="1"/>
  </g>
  <rect x="29" y="14" width="40" height="6" rx="3" fill="{ACCENT}"/>
  <line x1="33" y1="20" x2="33" y2="33" stroke="{ACCENT}" stroke-width="2.4" stroke-linecap="round"/>
  <line x1="65" y1="20" x2="65" y2="31" stroke="{ACCENT}" stroke-width="2.4" stroke-linecap="round"/>""",
    )

    # ---- 病灶标记 ----
    write(
        "m7-lesion.svg",
        f"""  <!-- 圈出来的病灶，和指向证据的引线 -->
  <circle cx="42" cy="56" r="23" fill="none" stroke="currentColor" stroke-width="3.4"/>
  <circle cx="42" cy="56" r="5.5" fill="{ACCENT}"/>
  <line x1="61" y1="43" x2="77" y2="29" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/>
  <rect x="76" y="14" width="15" height="15" rx="3" fill="currentColor"/>""",
    )

    # ---- 检索轨迹 ----
    write(
        "m8-trace.svg",
        f"""  <!-- 一路走到命中：两个墨点，一个朱砂点 -->
  <path d="M 16 80 L 16 50 L 44 50 L 44 26 L 66 26" fill="none" stroke="currentColor"
        stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="16" cy="80" r="3.4" fill="currentColor"/>
  <circle cx="44" cy="50" r="4.2" fill="currentColor"/>
  <circle cx="66" cy="26" r="5.8" fill="{ACCENT}"/>""",
    )

    # ---- 影像里的那一格 ----
    cells = []
    for r in range(3):
        for c in range(3):
            x = 16 + c * 20
            y = 16 + r * 20
            cells.append(f'<rect x="{x}" y="{y}" width="15" height="15" rx="2.5"/>')
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
