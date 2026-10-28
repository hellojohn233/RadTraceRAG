"""RadTraceRAG 界面设计 token 与主题。

视觉基调是**纸面**：米白纸底、墨色字、朱砂红强调。参照物不是 SaaS 控制台，
而是放射科报告和学术论文——因为「可追溯」这件事在纸面上就是引用制度：
答案带脚注号，证据按编号列在下方，每条写明来源。

配色只有三色加中性色，logo 和界面共用一套：

    墨色  #16202E   文字、结构、影像切片
    朱砂  #C0472B   引用编号、命中的那一点、强调
    米白  #FBFAF7   纸
    藏青  #234A6E   次级强调
"""

from __future__ import annotations

from string import Template

import streamlit as st

# ---------------------------------------------------------------- 设计 token

COLORS = {
    # 纸与墨
    "bg": "#FBFAF7",
    "surface": "#F1EFE9",
    "surface_2": "#FFFFFF",
    "border": "#E2E0DA",
    "border_2": "#D3D8DF",
    "text": "#16202E",
    "text_dim": "#3C4654",
    "text_mute": "#8A94A2",
    # 强调
    "accent": "#C0472B",
    "accent_soft": "#E8B9AC",
    "navy": "#234A6E",
    # 状态
    "ok": "#2E7D5B",
    "bad": "#B3261E",
}

SERIF_STACK = (
    "'Songti SC', 'STSong', 'Source Han Serif SC', 'Noto Serif CJK SC', "
    "Georgia, 'Times New Roman', serif"
)

SANS_STACK = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'PingFang SC', "
    "'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif"
)

MONO_STACK = (
    "'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', 'Courier New', monospace"
)

# ---------------------------------------------------------------- CSS

CSS = """
<style>
:root {
  --rt-bg: $bg;
  --rt-surface: $surface;
  --rt-surface-2: $surface_2;
  --rt-border: $border;
  --rt-border-2: $border_2;
  --rt-text: $text;
  --rt-text-dim: $text_dim;
  --rt-text-mute: $text_mute;
  --rt-accent: $accent;
  --rt-accent-soft: $accent_soft;
  --rt-navy: $navy;
  --rt-sans: $sans;
  --rt-serif: $serif;
  --rt-mono: $mono;
}

/* ---------- 底色与字体 ---------- */
.stApp { background: var(--rt-bg); }
[data-testid="stAppViewContainer"] {
  --primary-color: $accent;
  --primary-color-hover: #A93C23;
  --primary-color-active: #8E3019;
  --background-color: var(--rt-bg);
  --secondary-background-color: var(--rt-surface);
  --text-color: var(--rt-text);
}
html, body, [class*="css"] { font-family: var(--rt-sans); color: var(--rt-text); }
[data-testid="stMarkdownContainer"] p { color: var(--rt-text); line-height: 1.75; }
h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {
  font-family: var(--rt-serif);
  letter-spacing: -0.005em;
  color: var(--rt-text);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }

/* 侧边栏也用纸白，让 logo 图标融进页面（图标底色是米白），
   用一条右边线区分左右两栏 */
[data-testid="stSidebar"] { background: var(--rt-bg); border-right: 1px solid var(--rt-border); }
[data-testid="stSidebarContent"] { padding-top: 1.4rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { font-family: var(--rt-serif); }

/* ---------- 顶栏 ---------- */
.rt-topbar {
  display: flex; align-items: center; gap: 15px;
  padding: 0 2px 15px 2px; margin-bottom: 26px;
  border-bottom: 2px solid var(--rt-text);
}
.rt-topbar img { height: 40px; border-radius: 5px; }
.rt-topbar-title { font-family: var(--rt-serif); font-size: 19px; font-weight: 600; }
.rt-topbar-spacer { flex: 1; }

/* ---------- 分区：靠分割线而不是卡片 ---------- */
/* 用 padding-top 而不是 margin-top：Streamlit 的 vertical block 会吃掉相邻
   元素的外边距，margin 撑不开间距。
   注意别写 :first-child 清零——每个 section 都是它所在容器的首个子元素，
   那样会把所有 section 的上间距都清掉。 */
.rt-section { padding-top: 30px; }
.rt-section-head {
  display: flex; align-items: center; gap: 12px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.16em;
  color: var(--rt-text-mute); margin-bottom: 13px;
}
.rt-section-head::after { content: ""; flex: 1; height: 1px; background: var(--rt-border); }
.rt-section-head.is-accent { color: var(--rt-accent); }

/* ---------- 问答 ---------- */
.rt-q { font-size: 19px; line-height: 1.6; margin-bottom: 20px; }
.rt-q b { font-weight: 600; }
.rt-a { font-size: 17px; line-height: 1.85; }
.rt-a b { font-weight: 600; }
.rt-ref {
  color: var(--rt-accent); font-weight: 600; font-size: 11px;
  margin-left: 1px; white-space: nowrap; vertical-align: super;
}

/* ---------- 影像灯箱 ---------- */
.rt-plate { background: var(--rt-surface); padding: 16px; border-radius: 4px; }
.rt-plate img { width: 100%; display: block; border-radius: 2px; }
.rt-plate-cap {
  display: flex; justify-content: space-between; gap: 12px;
  font-size: 11px; color: var(--rt-text-mute); margin-top: 10px;
}

/* ---------- 轨迹行 ---------- */
.rt-rows { font-size: 14px; }
.rt-row {
  display: grid; grid-template-columns: 96px 1fr auto; gap: 16px;
  align-items: baseline; padding: 8px 0;
  border-bottom: 1px dotted var(--rt-border);
}
.rt-row:last-child { border-bottom: 0; }
.rt-row-key { font-weight: 600; }
.rt-row-val { color: var(--rt-text-dim); }
.rt-row-meta { color: var(--rt-text-mute); font-size: 12px; white-space: nowrap; }
.rt-row.is-hit .rt-row-key { color: var(--rt-accent); }

/* ---------- 证据列表 ---------- */
.rt-refs { font-size: 14.5px; }
.rt-ref-item {
  display: grid; grid-template-columns: 38px 1fr; gap: 14px;
  padding: 12px 0; border-top: 1px solid var(--rt-border);
}
.rt-ref-item:first-child { border-top: 2px solid var(--rt-text); }
.rt-ref-item.is-missed { opacity: 0.5; }
.rt-ref-no { color: var(--rt-accent); font-weight: 600; font-size: 13px; }
.rt-ref-item.is-missed .rt-ref-no { color: var(--rt-text-mute); }
.rt-ref-body p { margin: 0 0 4px 0; line-height: 1.7; }
.rt-ref-src { font-size: 11.5px; color: var(--rt-text-mute); }

/* ---------- 指标（细排，不做卡片块） ---------- */
.rt-stats { display: flex; gap: 26px; flex-wrap: wrap; }
.rt-stat-v { font-family: var(--rt-serif); font-size: 21px; font-weight: 600; }
.rt-stat-l { font-size: 11px; color: var(--rt-text-mute); margin-top: 1px; }

/* ---------- 对照组切换 ---------- */
[data-testid="stRadio"] > div { gap: 0; }
[data-testid="stRadio"] label {
  border: 1px solid var(--rt-border-2); border-radius: 999px;
  padding: 5px 14px; background: transparent; margin-right: 0;
  transition: all 0.15s ease;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: var(--rt-text); border-color: var(--rt-text);
}
[data-testid="stRadio"] label:has(input:checked) p { color: var(--rt-bg) !important; }
[data-testid="stRadio"] label:hover { border-color: var(--rt-text-mute); }
[data-testid="stRadio"] input[type="radio"] { accent-color: var(--rt-accent); }
/* 药丸样式已经用底色区分选中态，原生圆点可以去掉，省出横向空间 */
[data-testid="stRadio"] [data-baseweb="radio"] { display: none !important; }

/* ---------- 控件 ---------- */
.stTextInput input, .stTextArea textarea {
  background: var(--rt-surface-2) !important;
  border: 1px solid var(--rt-border-2) !important;
  border-radius: 3px !important;
  color: var(--rt-text) !important;
  font-family: var(--rt-sans) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--rt-text) !important; box-shadow: none !important;
}
.stButton > button {
  background: var(--rt-text); color: var(--rt-bg);
  border: 1px solid var(--rt-text); border-radius: 3px;
  font-weight: 600; width: 100%; letter-spacing: 0.04em;
  transition: opacity 0.15s ease;
}
/* 按钮文字是 markdown 容器里的 p，会被上面的 p 颜色规则改成墨色，
   墨字压在墨底上看不见，这里显式改回来。 */
.stButton > button p,
.stButton > button span { color: var(--rt-bg) !important; }
.stButton > button:hover { opacity: 0.85; color: var(--rt-bg); }
[data-testid="stFileUploader"] section {
  background: var(--rt-surface-2); border: 1px dashed var(--rt-border-2); border-radius: 3px;
}
[data-testid="stFileUploaderDropzoneInstructions"] > * { visibility: hidden; height: 0; }
[data-testid="stFileUploader"] button { color: var(--rt-text-dim) !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * { font-size: 0 !important; }
[data-testid="stFileUploader"] button::after {
  content: "选择文件"; font-size: 13px !important; font-weight: 500;
}
[data-testid="stFileUploader"] small { color: var(--rt-text-mute) !important; }
[data-testid="stSlider"] [role="slider"] { background-color: var(--rt-accent) !important; }
[data-testid="stSlider"] [data-testid="stSliderTickBarMin"] { background: var(--rt-accent) !important; }
[data-testid="stCaptionContainer"] { color: var(--rt-text-mute); }
[data-testid="stExpander"] { border: 1px solid var(--rt-border); border-radius: 3px; }
hr { border-color: var(--rt-border); }
a { color: var(--rt-accent); }
</style>
"""

CSS = Template(CSS).safe_substitute(
    **COLORS,
    sans=SANS_STACK,
    serif=SERIF_STACK,
    mono=MONO_STACK,
)


def apply_theme() -> None:
    """把主题注入到当前 Streamlit 页面。"""
    st.markdown(CSS, unsafe_allow_html=True)
