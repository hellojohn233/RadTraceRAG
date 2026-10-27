#!/usr/bin/env python3
"""组装最终 logo 候选包。

  assets/logo/svg/<id>.svg      8 个候选（带 <title>，currentColor）
  assets/logo/png/<id>-512.png  透明底 512 导出
  assets/logo/showcase.html     自包含展示页（无外部依赖）

用法：python make_logo_set.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from render_svg import render  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "assets" / "logo"

INK = "#16202E"
CINNABAR = "#C0472B"
PAPER = "#FBFAF7"

# id, 名称, 源文件, 分组, 设计说明
VARIANTS = [
    (
        "01-monogram-r",
        "R 单体",
        "assets/marks4/p3-r.svg",
        "字标",
        "Georgia Bold 的 R 直接用做单体字标。和 UI 里 Georgia 衬线字标同族，"
        "整套品牌像一本医学期刊的刊头。零装饰，靠字形本身的质量说话。",
    ),
    (
        "02-rt-ligature",
        "RT 合字",
        "assets/marks4/p7-rt.svg",
        "字标",
        "T 的横杠压在 R 的顶线上，两个字共用一根竖笔。比单独的 R 多一层信息，"
        "但也更容易被读成 ℞（处方符号）。",
    ),
    (
        "03-citation-mark",
        "引用标记",
        "assets/marks4/p2-semicolon-square.svg",
        "字标",
        "分号是排版里「引文清单」的记号，句点改成实心方块让它在小尺寸下更硬朗。"
        "整个产品就是一份带引文的报告，这个记号直接对应。",
    ),
    (
        "04-citation-comma",
        "引用逗号",
        "assets/marks4/p6-comma.svg",
        "字标",
        "更极端的版本：一个逗号就是全部标志。黑白印刷里它最好用，"
        "但语义要靠文案补。",
    ),
    (
        "05-trace-spiral",
        "溯源",
        "assets/marks6/q1b-spiral.svg",
        "符号",
        "一条线绕回中心，中心是那个被追到的原点。溯源不是查一次就完，"
        "是能一路退回起点。纯几何，和任何医学影像没有相似性。",
    ),
    (
        "06-annotation-leader",
        "标注引线",
        "assets/marks5/q3-leader.svg",
        "符号",
        "点 + 折线 + 方块，是图表和论文里标注引线的结构。回答从图像出发，"
        "指向一条证据。",
    ),
    (
        "07-echo-arcs",
        "两次观察",
        "assets/marks5/q4-halfring.svg",
        "符号",
        "一段粗弧、一段细弧，像同一处被看过两次。RAG 前后两轮，"
        "第二次比第一次轻。",
    ),
    (
        "08-aperture-ring",
        "环缺",
        "assets/marks4/p5-gapring.svg",
        "符号",
        "厚环开一道缺口，节点从缺口探出。最抽象的一个，"
        "结构稳、缩到 16px 也认得出。",
    ),
]


def add_title(svg: str, title: str, desc: str) -> str:
    svg = svg.replace(">", f"><title>{title}</title><desc>{desc}</desc>", 1)
    return svg


def main() -> None:
    svg_dir = OUT / "svg"
    png_dir = OUT / "png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for vid, name, src, group, desc in VARIANTS:
        raw = (ROOT / src).read_text(encoding="utf-8")
        raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
        body = add_title(raw.strip(), f"RadTraceRAG — {name}", desc)
        (svg_dir / f"{vid}.svg").write_text(body + "\n", encoding="utf-8")

        png = png_dir / f"{vid}-512.png"
        render(svg_dir / f"{vid}.svg", png, 512, bg="transparent", color="16202E")
        print("rendered", png)

        cards.append(
            {
                "id": vid,
                "name": name,
                "group": group,
                "desc": desc,
                "svg": body,
            }
        )

    (OUT / "showcase.html").write_text(build_html(cards), encoding="utf-8")
    print("wrote", OUT / "showcase.html")


def build_html(cards) -> str:
    card_html = []
    for c in cards:
        card_html.append(
            f"""    <figure class="card" data-group="{c['group']}">
      <div class="mark">{c['svg']}</div>
      <div class="row">
        <span class="idx">{c['id'].split('-')[0]}</span>
        <span class="name">{c['name']}</span>
        <span class="group">{c['group']}</span>
      </div>
      <p class="desc">{c['desc']}</p>
      <div class="checks">
        <div class="check"><span class="k">16</span>{c['svg']}</div>
        <div class="check mono"><span class="k">16</span>{c['svg']}</div>
        <div class="check"><span class="k">32</span>{c['svg']}</div>
        <div class="check mono"><span class="k">32</span>{c['svg']}</div>
      </div>
      <div class="lockup">
        <span class="lk-mark">{c['svg']}</span>
        <span class="lk-word">RadTrace<em>RAG</em></span>
      </div>
    </figure>"""
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RadTraceRAG — Logo 候选集</title>
<style>
  :root {{
    --ink: {INK};
    --cinnabar: {CINNABAR};
    --paper: {PAPER};
    --rule: #E1DDD4;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 56px 40px 80px;
    background: var(--paper); color: var(--ink);
    font-family: Georgia, "Times New Roman", "Songti SC", serif;
    -webkit-font-smoothing: antialiased;
  }}
  header {{ max-width: 1180px; margin: 0 auto 44px; }}
  h1 {{ font-size: 34px; font-weight: 400; margin: 0 0 10px; letter-spacing: .01em; }}
  .sub {{ font-size: 15px; line-height: 1.75; color: #4A5563; max-width: 760px; margin: 0; }}
  .rules {{ margin: 22px 0 0; max-width: 760px; }}
  .rules li {{ font-size: 14px; line-height: 1.85; color: #4A5563; margin: 0 0 4px; }}
  .rules b {{ font-weight: 400; color: var(--cinnabar); }}
  .grid {{
    max-width: 1180px; margin: 0 auto;
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 1px; background: var(--rule); border: 1px solid var(--rule);
  }}
  .card {{ background: var(--paper); padding: 26px 26px 22px; margin: 0; }}
  .mark {{ height: 168px; display: flex; align-items: center; justify-content: center; }}
  .mark svg {{ width: 150px; height: 150px; color: var(--ink); transition: color .25s; }}
  .card:hover .mark svg {{ color: var(--cinnabar); }}
  .row {{ display: flex; align-items: baseline; gap: 10px; margin: 20px 0 8px;
          border-top: 1px solid var(--rule); padding-top: 12px; }}
  .idx {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11px;
          letter-spacing: .12em; color: #9AA3AE; }}
  .name {{ font-size: 17px; }}
  .group {{ margin-left: auto; font-family: "Helvetica Neue", Arial, sans-serif;
            font-size: 11px; letter-spacing: .12em; color: var(--cinnabar); }}
  .desc {{ font-size: 13px; line-height: 1.8; color: #4A5563; margin: 0 0 16px; min-height: 70px; }}
  .checks {{ display: flex; align-items: flex-end; gap: 18px; padding: 12px 0 14px;
             border-top: 1px solid var(--rule); }}
  .check {{ display: flex; align-items: center; gap: 6px; }}
  .check svg {{ width: 16px; height: 16px; color: var(--ink); }}
  .check:nth-child(3) svg {{ width: 32px; height: 32px; }}
  .check.mono svg {{ color: #000; }}
  .k {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9px; color: #A8AEB8; }}
  .lockup {{ display: flex; align-items: center; gap: 11px; padding-top: 13px;
             border-top: 1px solid var(--rule); }}
  .lk-mark svg {{ width: 34px; height: 34px; color: var(--ink); display: block; }}
  .lk-word {{ font-size: 19px; letter-spacing: .02em; }}
  .lk-word em {{ font-style: italic; color: var(--cinnabar); }}
  footer {{ max-width: 1180px; margin: 34px auto 0; font-size: 12px; color: #8A929C;
            font-family: "Helvetica Neue", Arial, sans-serif; }}
</style>
</head>
<body>
<header>
  <h1>RadTraceRAG — 标志候选集</h1>
  <p class="sub">八个方向，全部避开看片/影像类图形。墨色 #16202E 单色成立，
  几何尺寸统一 100×100，缩到 16px 仍可辨。</p>
  <ul class="rules">
    <li><b>01 R 单体</b> —— 和 Georgia 衬线字标同族，最像一本期刊的刊头。</li>
    <li><b>05 溯源</b> —— 纯符号，一条线绕回被追到的原点。</li>
    <li><b>03 引用标记</b> —— 分号是引文清单的记号，句点改方块。</li>
  </ul>
</header>
<div class="grid">
{chr(10).join(card_html)}
</div>
<footer>RadTraceRAG · 纸面报告风格 UI · 墨 #16202E / 朱砂 #C0472B / 米白 #FBFAF7</footer>
</body>
</html>
"""


if __name__ == "__main__":
    main()
