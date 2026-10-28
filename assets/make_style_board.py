#!/usr/bin/env python3
"""导出设计 token（JSON）并生成风格视觉板（HTML + PNG）。

    python assets/make_style_board.py

输出：
  assets/style/design-tokens.json   机器可读 token，Figma / Style Dictionary 都能吃
  assets/style/style-board.html      视觉板源文件（自包含）
  assets/style-board.png             视觉板图片
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from radtracerag.ui import theme  # noqa: E402

OUT_DIR = ROOT / "assets" / "style"
BOARD_PNG = ROOT / "assets" / "style-board.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

MARK = ROOT / "assets" / "logo" / "svg" / "01-window-level.svg"


def tokens() -> dict:
    return {
        "color": {
            **theme.COLORS,
        },
        "spacing": {f"space-{k}": f"{v}px" for k, v in theme.SPACING.items()},
        "fontSize": {f"fs-{k}": f"{v}px" for k, v in theme.TYPE.items()},
        "radius": {f"radius-{k}": f"{v}px" for k, v in theme.RADIUS.items()},
        "lineWidth": {f"line-{k}": f"{v}px" for k, v in theme.LINE.items()},
        "lineHeight": {f"leading-{k}": v for k, v in theme.LEADING.items()},
        "letterSpacing": {f"tracking-{k}": f"{v}em" for k, v in theme.TRACKING.items()},
        "layout": {f"layout-{k}": f"{v}px" for k, v in theme.LAYOUT.items()},
        "fontFamily": {
            "serif": theme.SERIF_STACK,
            "sans": theme.SANS_STACK,
            "mono": theme.MONO_STACK,
        },
    }


def mark_svg(color: str = "currentColor") -> str:
    raw = MARK.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"<title>.*?</title>", "", raw, flags=re.S)
    raw = re.sub(r"<desc>.*?</desc>", "", raw, flags=re.S)
    raw = raw.replace("currentColor", color)
    return raw.replace('xmlns="http://www.w3.org/2000/svg"', "").strip()


def build_html() -> str:
    c = theme.COLORS
    t = theme.TYPE

    swatches = [
        ("墨", c["text"], "正文、结构线、影像切片", "#FBFAF7"),
        ("朱砂", c["accent"], "引用编号、命中、唯一强调", "#FBFAF7"),
        ("藏青", c["navy"], "次级强调（备用，默认不用）", "#FBFAF7"),
        ("纸白", c["bg"], "页面底", "#16202E"),
        ("浅灰", c["surface"], "灯箱底、侧边栏块", "#16202E"),
        ("白", c["surface_2"], "输入框、证据区", "#16202E"),
    ]
    sw_html = "".join(
        f"""      <div class="sw">
        <div class="chip" style="background:{hexv};border:1px solid {c['border']};"></div>
        <div class="sw-t"><b>{name}</b><span>{hexv}</span></div>
        <div class="sw-d">{use}</div>
      </div>"""
        for name, hexv, use, _ in swatches
    )

    type_rows = "".join(
        f"""      <div class="tr">
        <span class="tk">fs-{k}</span>
        <span class="tv">{v}px</span>
        <span class="ts" style="font-size:{v}px;{serif}">{sample}</span>
      </div>"""
        for (k, v), sample, serif in zip(
            t.items(),
            [
                "4　8　4.5　s",
                "腹部轴位断层",
                "图中主要器官是肝脏",
                "肝是人体最大的腺体，位于右季肋区",
                "12",
                "SLAKE knowledge_base",
                "检索轨迹　图像观察",
                "RadTraceRAG",
            ],
            [
                "font-family:'Songti SC',Georgia,serif;font-weight:600;",
                "font-family:'Songti SC',Georgia,serif;font-weight:600;",
                "font-family:'Songti SC',Georgia,serif;font-weight:600;",
                "",
                "font-family:'Songti SC',Georgia,serif;font-weight:600;",
                "",
                "",
                "font-family:'Songti SC',Georgia,serif;font-weight:600;",
            ],
        )
    )

    space_html = "".join(
        f"""      <div class="sb">
        <span class="tk">space-{k}</span>
        <span class="bar" style="width:{v}px;"></span>
        <span class="tv">{v}px</span>
      </div>"""
        for k, v in theme.SPACING.items()
    )

    mark = mark_svg()
    mark_white = mark_svg("#FBFAF7")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RadTraceRAG 风格基调</title>
<style>
  :root {{
    --ink:{c['text']}; --accent:{c['accent']}; --paper:{c['bg']}; --surface:{c['surface']};
    --border:{c['border']}; --border2:{c['border_2']}; --mute:{c['text_mute']}; --dim:{c['text_dim']};
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:56px 56px 72px; background:var(--paper); color:var(--ink);
         font-family:'Inter',-apple-system,'PingFang SC','Helvetica Neue',Arial,sans-serif; }}
  .sheet {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-family:'Songti SC',Georgia,serif; font-weight:600; font-size:34px; margin:0 0 8px; }}
  .lede {{ font-size:15px; line-height:1.85; color:var(--dim); max-width:760px; margin:0 0 6px; }}
  .keys {{ display:flex; gap:10px; margin:18px 0 0; }}
  .key {{ border:1px solid var(--border2); border-radius:999px; padding:4px 14px; font-size:12px;
          letter-spacing:.06em; color:var(--accent); }}
  section {{ margin-top:52px; }}
  .sh {{ display:flex; align-items:baseline; gap:14px; border-bottom:2px solid var(--ink);
        padding-bottom:10px; margin-bottom:22px; }}
  .sh h2 {{ font-family:'Songti SC',Georgia,serif; font-size:21px; font-weight:600; margin:0; }}
  .sh p {{ margin:0; font-size:12.5px; color:var(--mute); }}
  .cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:18px; }}
  .sw-t {{ display:flex; align-items:baseline; gap:8px; margin-top:9px; }}
  .sw-t b {{ font-size:13.5px; }}
  .sw-t span {{ font-size:11px; color:var(--mute); font-family:ui-monospace,Menlo,monospace; }}
  .sw-d {{ font-size:11.5px; color:var(--dim); margin-top:2px; line-height:1.6; }}
  .chip {{ height:74px; border-radius:4px; }}
  .tr,.sb {{ display:grid; grid-template-columns:104px 62px 1fr; gap:14px; align-items:baseline;
            padding:7px 0; border-bottom:1px solid var(--border); }}
  .sb {{ grid-template-columns:104px 1fr 62px; }}
  .tk {{ font-family:ui-monospace,Menlo,monospace; font-size:11px; color:var(--mute); }}
  .tv {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--dim); }}
  .ts {{ color:var(--ink); line-height:1.4; }}
  .bar {{ height:10px; background:var(--ink); display:block; }}
  .note {{ font-size:12.5px; color:var(--dim); line-height:1.8; margin:16px 0 0; }}
  .rules {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:26px; }}
  .rule {{ border:1px solid var(--border); border-radius:4px; padding:16px 18px; background:#fff; }}
  .rule h3 {{ font-size:13px; margin:0 0 8px; font-weight:600; }}
  .rule p {{ font-size:12px; line-height:1.75; color:var(--dim); margin:0; }}
  .demo {{ display:flex; align-items:center; justify-content:center; gap:16px; height:120px;
          background:var(--surface); border-radius:4px; margin-top:10px; }}
  .demo.onink {{ background:var(--ink); }}
  .demo svg {{ width:64px; height:64px; }}
  .demo.onink svg {{ color:#FBFAF7; }}
  .demo.sm svg {{ width:28px; height:28px; }}
  .demo.xs svg {{ width:16px; height:16px; }}
  .box {{ display:inline-flex; padding:18px; border:1px dashed var(--border2); border-radius:4px; }}
  .ban {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
  .ban div {{ width:76px; height:76px; border:1px solid var(--border); border-radius:4px;
              display:flex; align-items:center; justify-content:center; position:relative;
              background:#fff; }}
  .ban div svg {{ width:40px; height:40px; }}
  .ban div::after {{ content:"✕"; position:absolute; right:4px; bottom:2px; color:#B3261E; font-size:13px; }}
  .kv {{ display:grid; grid-template-columns:132px 1fr; gap:8px 16px; font-size:12.5px;
         line-height:1.8; color:var(--dim); }}
  .kv b {{ color:var(--ink); font-weight:600; }}
</style>
</head>
<body><div class="sheet">

<h1>RadTraceRAG 风格基调</h1>
<p class="lede">这是一份可以核查的文档，不是灵感板。页面上出现的每个颜色、字号、间距都能在
<code>src/radtracerag/ui/theme.py</code> 里找到同名 token；改代码时改 token，不改数值。
本文件的数字与代码由 <code>assets/make_style_board.py</code> 同源生成，不会各写各的。</p>
<div class="keys"><span class="key">纸面</span><span class="key">可核查</span><span class="key">克制</span></div>

<section>
  <div class="sh"><h2>色彩</h2><p>一色为主，朱砂为唯一强调。界面上出现没定义过的颜色就是设计出错。</p></div>
  <div class="cols">{sw_html}</div>
  <p class="note">朱砂只给三样东西：<b>引用编号</b>、<b>命中</b>、<b>主操作</b>。一屏之内朱砂出现超过
  5 处就说明层级乱了——先砍强调，再想文案。状态色（ok / bad）只在真实链路里出现，
  演示界面不用。</p>
</section>

<section>
  <div class="sh"><h2>字阶</h2><p>衬线做标题、答案和数值，无衬线做正文与元信息。字号只有这几档。</p></div>
  {type_rows}
  <p class="note">数字用衬线（读数要像读数，不像代码）。<b>证据 ID 不用等宽字体</b>——
  等宽是控制台的做法，编号用朱砂上标就够了。中文正文与英文来源的字号相同（14.5px），
  不要为了「英文看起来小」再降一档。</p>
</section>

<section>
  <div class="sh"><h2>间距 · 圆角 · 线宽</h2><p>4px 基数。间距用标尺，不用随手写的偶数。</p></div>
  {space_html}
  <p class="note">圆角只有三档：影像 <b>2px</b>、容器 <b>4px</b>、药丸 <b>999px</b>。
  线宽只有两档：发丝 <b>1px</b>（分区、表格行）、强调 <b>2px</b>（顶栏底线、证据首条）。
  圆角大于 4px 的东西不属于这套语言。</p>
</section>

<section>
  <div class="sh"><h2>标志使用</h2><p>窗位：直方图 + 朱砂窗位框。留白取窗位框宽的 1/2。</p></div>
  <div class="rules">
    <div class="rule">
      <h3>留白</h3>
      <p>标志四周至少留 1/2 个窗位框宽的空白，四周等距。</p>
      <div class="demo"><span class="box">{mark}</span></div>
    </div>
    <div class="rule">
      <h3>最小尺寸</h3>
      <p>印刷 ≥ 8mm，屏幕 ≥ 16px。低于 16px 时直方图会糊成一条线，改用纯色块版本。</p>
      <div class="demo sm"><span class="box">{mark}</span></div>
      <div class="demo xs" style="height:60px;"><span class="box">{mark}</span></div>
    </div>
    <div class="rule">
      <h3>反白</h3>
      <p>深底上把墨色部分换成纸白，朱砂框保持朱砂不变。</p>
      <div class="demo onink">{mark_white}</div>
    </div>
  </div>
  <p class="note">禁止事项</p>
  <div class="ban">
    <div style="transform:scaleX(1.8)">{mark}</div>
    <div style="transform:rotate(-18deg)">{mark}</div>
    <div style="filter:drop-shadow(0 3px 5px rgba(0,0,0,.35))">{mark}</div>
    <div style="border-radius:50%;overflow:hidden;background:#C0472B">{mark}</div>
  </div>
  <p class="note">不拉伸、不旋转、不加阴影、不加渐变、不换色、不加圆角容器。标志是印在纸上的，
  不是贴在界面上的贴纸。</p>
</section>

<section>
  <div class="sh"><h2>图形语言</h2><p>图表、图标、示意图都跟标志同一套构造。</p></div>
  <div class="kv">
    <b>直方图</b><span>细长条、等宽、底部对齐，宽度约 9px 视觉，间距约为宽度 0.45 倍</span>
    <b>描边</b><span>2–3px，端点方头，不用圆头</span>
    <b>选区</b><span>朱砂描边矩形，不填充——和标志里的窗位框一致</span>
    <b>图形分组</b><span>靠 1px 分割线，不靠卡片阴影</span>
    <b>图标</b><span>线性、单色、24px 网格、2px 描边，不混用多色</span>
    <b>影像</b><span>不加彩色边框、不加圆角以外的处理、不做美化滤镜，保留原始灰阶</span>
  </div>
</section>

<section>
  <div class="sh"><h2>文案语气</h2><p>报告语体：陈述，不推销，不兜售。</p></div>
  <div class="kv">
    <b>陈述</b><span>「腹部轴位断层，可见右上腹大片实质脏器影。」——说看到了什么</span>
    <b>不确定</b><span>「不能仅凭单张图定性。」——不确定就说不确定，不写模糊的肯定句</span>
    <b>没有证据</b><span>「没有检索到证据，答案没有可回溯的支撑。」——空结果也要占位，不留白</span>
    <b>禁止</b><span>不用「智能」「高效」「一键」「赋能」；不用感叹号；不写「本系统」</span>
  </div>
</section>

</div></body></html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "design-tokens.json").write_text(
        json.dumps(tokens(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", OUT_DIR / "design-tokens.json")

    html = OUT_DIR / "style-board.html"
    html.write_text(build_html(), encoding="utf-8")
    print("wrote", html)

    subprocess.run(
        [
            CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--screenshot={BOARD_PNG}", "--window-size=1400,2660",
            "--default-background-color=00000000",
            f"file://{html.resolve()}",
        ],
        capture_output=True,
        check=False,
    )
    print("wrote", BOARD_PNG)


if __name__ == "__main__":
    main()
