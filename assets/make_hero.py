#!/usr/bin/env python3
"""生成 README 首屏横幅：窗位 → 证据链。

    python assets/make_hero.py

输出：
  assets/style/hero.html   横幅源文件（自包含，便于手工微调）
  assets/hero.png          README 首屏，1600×560

图形语言与标志一致：直方图用细长条等宽底部对齐，选区是朱砂描边矩形且不填充，
分组靠 1px 分割线，描边端点方头。朱砂只出现在窗位与引用编号两处角色上。
配色与字阶全部取自 theme.py 的 token，不写字面量。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from radtracerag.ui import theme  # noqa: E402

OUT_HTML = ROOT / "assets" / "style" / "hero.html"
OUT_PNG = ROOT / "assets" / "hero.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

W, H = 1600, 560

# 直方图构造：细长条、等宽 9px、间距 4px（约为宽度的 0.45 倍）
BAR_W, GAP = 9, 4
N = 28
HIST_X, BASE_Y, MAXH = 104, 404, 276

# 医学 CT 的像素值分布：空气峰最高、软组织宽缓丘、骨小峰收尾
PROFILE = [
    1.00, 0.92, 0.63, 0.34, 0.26, 0.30, 0.40, 0.47,
    0.52, 0.55, 0.57, 0.59, 0.60, 0.60, 0.58, 0.55,
    0.51, 0.46, 0.40, 0.33, 0.27, 0.22, 0.26, 0.30,
    0.24, 0.17, 0.10, 0.05,
]

# 窗位框住的桶：软组织峰，器官在这段灰阶里才显影
WIN_LO, WIN_HI = 7, 15

# 证据链：取自演示数据（src/radtracerag/ui/demo_data.py）
EVID = [
    ("1", ("肝是人体最大的腺体，位于右季肋区。功能包括糖异生、糖原储存、"
           "尿素合成与氨解毒，胆汁由肝细胞分泌并储存于胆囊。"),
     "SLAKE knowledge_base · 肝脏条目"),
    ("2", ("胆汁经胆总管进入十二指肠，参与脂肪的乳化与脂溶性维生素的吸收；"
           "胆囊负责储存与浓缩。"),
     "SLAKE knowledge_base · 胆汁条目"),
    ("3", ("腹腔积液指腹腔内超出正常量的游离液体，可见于肝硬化、心衰、"
           "结核性腹膜炎、胰腺炎等多种情况。"),
     "SLAKE knowledge_base · 腹水条目"),
    ("4", "脾位于左季肋区，质地脆，被膜下出血易破裂，是人体最大的淋巴器官。",
     "SLAKE knowledge_base · 脾脏条目"),
]
COL_X, COL_W = 648, W - 648 - 96
ROW_H, ROW0 = 104, 62
SPINE_X, RAIL_Y = 570, 34


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html() -> str:
    c = theme.COLORS
    t = theme.TYPE
    serif = ", ".join(theme.SERIF_STACK)
    sans = ", ".join(theme.SANS_STACK)

    fx0 = HIST_X + WIN_LO * (BAR_W + GAP) - 6
    fx1 = HIST_X + (WIN_HI + 1) * (BAR_W + GAP) - 6
    fy0 = BASE_Y - MAXH * max(PROFILE[WIN_LO:WIN_HI + 1]) - 24
    fy1 = BASE_Y + 6
    ys = [ROW0 + i * ROW_H + 14 for i in range(len(EVID))]
    mid = (fx0 + fx1) / 2

    p: list[str] = [
        f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:{W}px;height:{H}px;overflow:hidden}}
body{{background:{c['bg']};font-family:{sans};color:{c['text']};position:relative}}
.bar{{position:absolute;background:{c['text']};width:{BAR_W}px}}
.frame{{position:absolute;border:3px solid {c['accent']};background:none}}
.hair{{position:absolute;background:{c['border']};height:1px}}
.rule{{position:absolute;background:{c['border_2']};height:1px}}
.spine{{position:absolute;background:{c['text']};width:1px}}
.tick{{position:absolute;background:{c['border_2']};height:1px}}
.no{{font-family:{serif};font-size:{t['no']}px;color:{c['accent']};position:absolute}}
.claim{{font-family:{serif};font-size:{t['lead']}px;line-height:{theme.LEADING['loose']};color:{c['text']};position:absolute}}
.src{{font-family:{sans};font-size:{t['sm']}px;line-height:{theme.LEADING['body']};color:{c['text_mute']};position:absolute}}
.label{{font-family:{sans};font-size:{t['sm']}px;letter-spacing:{theme.TRACKING['label']};color:{c['text_mute']};position:absolute}}
</style></head><body>"""
    ]

    for i, v in enumerate(PROFILE):
        x = HIST_X + i * (BAR_W + GAP)
        h = MAXH * v
        p.append(f'<div class="bar" style="left:{x}px;top:{BASE_Y - h:.0f}px;height:{h:.0f}px"></div>')
    p.append(f'<div class="hair" style="left:{HIST_X - 22}px;top:{BASE_Y + 1}px;width:{N * (BAR_W + GAP)}px"></div>')
    p.append(f'<div class="frame" style="left:{fx0}px;top:{fy0:.0f}px;width:{fx1 - fx0}px;height:{fy1 - fy0:.0f}px"></div>')
    p.append(f'<div class="label" style="left:{HIST_X - 22}px;top:{BASE_Y + 20}px">像素值分布</div>')
    p.append(f'<div class="label" style="left:{fx0}px;top:{BASE_Y + 20}px">窗位 · 检索到的灰阶</div>')

    for i, (no, claim, src) in enumerate(EVID):
        ry = ROW0 + i * ROW_H
        p.append(f'<div class="no" style="left:{COL_X}px;top:{ry + 3}px">[{no}]</div>')
        p.append(f'<div class="claim" style="left:{COL_X + 32}px;top:{ry}px;width:{COL_W - 32}px">{esc(claim)}</div>')
        p.append(f'<div class="src" style="left:{COL_X + 32}px;top:{ry + 64}px">{esc(src)}</div>')
        if i < len(EVID) - 1:
            p.append(f'<div class="hair" style="left:{COL_X}px;top:{ry + ROW_H - 16}px;width:{COL_W}px"></div>')

    # 追溯线：窗位上翻、越过直方图、右折到竖脊，再分到各行
    p.append(f'<div class="rule" style="left:{mid:.0f}px;top:{fy0:.0f}px;width:1px;height:{RAIL_Y - fy0:.0f}px"></div>')
    p.append(f'<div class="rule" style="left:{mid:.0f}px;top:{RAIL_Y}px;width:{SPINE_X - mid:.0f}px"></div>')
    p.append(f'<div class="spine" style="left:{SPINE_X}px;top:{RAIL_Y}px;height:{ys[-1] - RAIL_Y}px"></div>')
    for y in ys:
        p.append(f'<div class="tick" style="left:{SPINE_X}px;top:{y}px;width:{COL_X - SPINE_X}px"></div>')

    p.append("</body></html>")
    return "".join(p)


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    print("wrote", OUT_HTML)

    subprocess.run(
        [
            CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--screenshot={OUT_PNG}", f"--window-size={W},{H}",
            "--default-background-color=00000000",
            f"file://{OUT_HTML.resolve()}",
        ],
        capture_output=True,
        check=False,
    )
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
