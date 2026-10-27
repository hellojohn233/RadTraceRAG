#!/usr/bin/env python3
"""影像派：从项目自己的元素里长出标志。

RadTraceRAG 的独有元素（不是通用医学图标）：
  腹部轴位断层 / 影像灯箱 / 窗宽窗位 / 上标引用编号 / 检索轨迹 / 证据编号

  m1-slice-ray    影像切片 + 一条指出去的线
  m2-window       窗宽窗位：像素值直方图 + 窗位带
  m3-plate-cite   影像格 + 朱砂引用角标
  m4-roi-trail    取景框（病灶区）+ 追进来的轨迹
  m5-register     对位标记
  m6-series       从序列里抽出来的那一片
  m7-lesion       病灶标记环 + 指向它的引线

输出：assets/marks7/*.svg
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent / "marks7"
ACCENT = "#C0472B"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n',
        encoding="utf-8",
    )
    print("wrote", OUT / name)


def main() -> None:
    # ---- m1 影像切片 + 指出去的线 ----
    write(
        "m1-slice-ray.svg",
        """  <!-- 一片影像，和一条从它指出去的线 -->
  <rect x="12" y="26" width="56" height="60" rx="9" fill="currentColor"/>
  <line x1="70" y1="42" x2="83" y2="29" stroke="currentColor" stroke-width="3.2" stroke-linecap="round"/>
  <rect x="78" y="12" width="16" height="16" rx="3" fill="currentColor"/>""",
    )

    # ---- m2 窗宽窗位 ----
    write(
        "m2-window.svg",
        """  <!-- 像素值直方图 + 窗位带 -->
  <g fill="currentColor">
    <rect x="6" y="58" width="9" height="22" rx="1"/>
    <rect x="19" y="46" width="9" height="34" rx="1"/>
    <rect x="32" y="36" width="9" height="44" rx="1"/>
    <rect x="45" y="28" width="9" height="52" rx="1"/>
    <rect x="58" y="34" width="9" height="46" rx="1"/>
    <rect x="71" y="48" width="9" height="32" rx="1"/>
    <rect x="84" y="62" width="9" height="18" rx="1"/>
  </g>
  <rect x="29" y="14" width="40" height="6" rx="3" fill="currentColor"/>
  <line x1="33" y1="20" x2="33" y2="33" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
  <line x1="65" y1="20" x2="65" y2="31" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>""",
    )

    # ---- m3 影像格 + 引用角标 ----
    write(
        "m3-plate-cite.svg",
        f"""  <!-- 一张影像，右上角挂一个引用编号 -->
  <rect x="14" y="24" width="52" height="52" rx="8" fill="currentColor"/>
  <rect x="66" y="10" width="18" height="18" rx="3" fill="{ACCENT}"/>""",
    )

    # ---- m4 取景框 + 轨迹 ----
    write(
        "m4-roi-trail.svg",
        f"""  <!-- 圈出来的那块区域，轨迹从缺口追进来 -->
  <g fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round">
    <path d="M 18 40 L 18 22 L 36 22"/>
    <path d="M 18 62 L 18 80 L 36 80"/>
    <path d="M 60 80 L 78 80 L 78 64"/>
  </g>
  <circle cx="90" cy="11" r="5" fill="currentColor"/>
  <circle cx="80" cy="19" r="3" fill="{ACCENT}"/>""",
    )

    # ---- m5 对位标记 ----
    write(
        "m5-register.svg",
        f"""  <!-- 对位：两个标记夹住一个目标 -->
  <g fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round">
    <path d="M 16 42 L 16 16 L 42 16"/>
    <path d="M 84 60 L 84 84 L 62 84"/>
  </g>
  <rect x="42" y="44" width="14" height="14" rx="2.5" fill="{ACCENT}"/>""",
    )

    # ---- m6 序列里抽出来的一片 ----
    write(
        "m6-series.svg",
        """  <!-- 一组片子，和被抽出来的那一片 -->
  <g fill="none" stroke="currentColor" stroke-width="3" stroke-linejoin="round">
    <rect x="14" y="12" width="24" height="24" rx="4"/>
    <rect x="42" y="12" width="24" height="24" rx="4"/>
    <rect x="14" y="40" width="24" height="24" rx="4"/>
    <rect x="42" y="40" width="24" height="24" rx="4"/>
  </g>
  <rect x="70" y="66" width="22" height="22" rx="4" fill="currentColor"/>""",
    )

    # ---- m7 病灶标记 ----
    write(
        "m7-lesion.svg",
        f"""  <!-- 圈住的病灶，和指向它的引线 -->
  <circle cx="44" cy="56" r="24" fill="none" stroke="currentColor" stroke-width="3.4"/>
  <circle cx="44" cy="56" r="6" fill="currentColor"/>
  <line x1="62" y1="40" x2="78" y2="26" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/>
  <rect x="76" y="12" width="16" height="16" rx="3" fill="{ACCENT}"/>""",
    )


if __name__ == "__main__":
    main()
