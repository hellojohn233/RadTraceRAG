#!/usr/bin/env python3
"""生成 RadTraceRAG 的横版 lockup 矢量图（无副标版本）。

字标用 Georgia Bold 的真实轮廓转成 path（不依赖字体文件，任何环境打开都一致），
标志图形取自 assets/logo/svg/01-window-level.svg。

    python make_lockup_svg.py --out assets/logo/final/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).parent.parent
GEORGIA_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
MARK_SRC = ROOT / "assets" / "logo" / "svg" / "01-window-level.svg"

INK = "#16202E"
CINNABAR = "#C0472B"
PAPER = "#FBFAF7"

MARK_H = 150.0      # 标志高度
CAP = 88.0          # 字标大写字高
GAP = 58.0          # 标志与字标间距
BASELINE = 131.0    # 字标基线：让字标视觉中心与标志内容中心对齐


def shape(text: str, font_path: str):
    """用 harfbuzz 排版，返回 (字形名, x 位移, y 位移, 前进宽度)。"""
    with open(font_path, "rb") as f:
        blob = hb.Blob(f.read())
    face = hb.Face(blob)
    font = hb.Font(face)
    hb.ot_font_set_funcs(font)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})

    infos = buf.glyph_infos
    positions = buf.glyph_positions
    out = []
    for info, pos in zip(infos, positions):
        out.append((info.codepoint, pos.x_offset, pos.y_offset, pos.x_advance))
    return out


def build() -> str:
    tt = TTFont(GEORGIA_BOLD)
    upm = tt["head"].unitsPerEm
    glyph_order = tt.getGlyphOrder()
    glyphset = tt.getGlyphSet()

    # 大写字高（以 H 为准）
    h_name = tt.getBestCmap()[ord("H")]
    cap_units = tt["glyf"][h_name].yMax
    scale = CAP / cap_units

    def run(text: str, x0: float, color: str) -> tuple[str, float]:
        parts = []
        x = x0
        for gid, xo, yo, adv in shape(text, GEORGIA_BOLD):
            name = glyph_order[gid]
            pen = SVGPathPen(glyphset)
            glyphset[name].draw(pen)
            d = pen.getCommands()
            if d:
                px = x + xo * scale
                py = BASELINE - yo * scale
                parts.append(
                    f'<path d="{d}" fill="{color}" '
                    f'transform="translate({px:.3f},{py:.3f}) scale({scale:.6f},{-scale:.6f})"/>'
                )
            x += adv * scale
        return "\n    ".join(parts), x

    ink_part, x_after = run("RadTrace", MARK_H + GAP, INK)
    red_part, x_end = run("RAG", x_after, CINNABAR)

    # 标志图形：把 currentColor 换成墨色，朱砂保留
    mark = MARK_SRC.read_text(encoding="utf-8")
    mark = re.sub(r"<!--.*?-->", "", mark, flags=re.S)
    mark = mark.replace("currentColor", INK)
    mark = re.sub(r"<svg[^>]*>", "", mark, count=1)
    mark = mark.replace("</svg>", "")
    mark = mark.replace("<title>", "<!--").replace("</title>", "-->")
    mark = mark.replace("<desc>", "<!--").replace("</desc>", "-->")
    mark_scale = MARK_H / 100.0

    width = x_end + 2.0
    height = MARK_H + 16.0

    # 标称尺寸给到 2400px 宽，viewBox 不变 —— 矢量，随便放大都不失真
    nom_w = 2400.0
    nom_h = round(nom_w * height / width, 2)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="0 0 {width:.2f} {height:.2f}" width="{nom_w:.0f}" height="{nom_h:.2f}"
     role="img" aria-label="RadTraceRAG">
  <title>RadTraceRAG</title>
  <desc>窗位标志 + RadTraceRAG 字标</desc>
  <g transform="translate(0,8) scale({mark_scale:.6f})">
    {mark.strip()}
  </g>
  {ink_part}
  {red_part}
</svg>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets" / "logo" / "final"))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    svg_path = outdir / "radtracerag-lockup.svg"
    svg_path.write_text(build(), encoding="utf-8")
    print("wrote", svg_path)


if __name__ == "__main__":
    main()
