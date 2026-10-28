"""RadTraceRAG 界面设计 token 与主题。

视觉基调是**纸面**：米白纸底、墨色字、朱砂红强调。参照物不是 SaaS 控制台，
而是放射科报告和学术论文——因为「可追溯」这件事在纸面上就是引用制度：
答案带脚注号，证据按编号列在下方，每条写明来源。

配色只有三色加中性色，logo 和界面共用一套：

    墨色  #16202E   文字、结构、影像切片
    朱砂  #C0472B   引用编号、命中的那一点、强调
    米白  #FBFAF7   纸
    藏青  #234A6E   次级强调

尺度（间距 / 字阶 / 圆角 / 线宽）全部定义在这里，CSS 只允许引用变量，
不允许出现字面量。规范见 `doc/visual-language.md`。
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

# 间距标尺：4px 基数，2px 只用于发丝级微调
SPACING = {
    "s1": 2,
    "s2": 4,
    "s3": 8,
    "s4": 12,
    "s5": 16,
    "s6": 24,
    "s7": 32,
    "s8": 48,
}

# 字阶：每一档都有角色，不允许在同一段里混用
TYPE = {
    "display": 21,   # 统计数值
    "title": 19,     # 提问、顶栏标题
    "lead": 17,      # 答案
    "body": 14.5,    # 正文、证据条目
    "no": 13,        # 引用编号
    "sm": 12,        # 次要元信息、来源
    "label": 11,     # 小标题、字段标签
    "brand": 16,     # 侧边栏字标
}

# 圆角：影像 2、容器 4、药丸 999
RADIUS = {
    "sm": 2,
    "md": 4,
    "pill": 999,
}

# 线宽：发丝 1、结构强调 2
LINE = {
    "hair": 1,
    "strong": 2,
}

# 行高
LEADING = {
    "tight": 1.6,
    "body": 1.75,
    "loose": 1.85,
}

# 字间距
TRACKING = {
    "head": -0.005,
    "label": 0.16,
    "btn": 0.04,
}

# 版面常量：内容最大宽、几处固定列宽、标志高度
LAYOUT = {
    "max_w": 1400,
    "w_key": 96,
    "w_no": 38,
    "logo_h": 40,
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

  /* 间距标尺 */
  --rt-s1: ${s1}px; --rt-s2: ${s2}px; --rt-s3: ${s3}px; --rt-s4: ${s4}px;
  --rt-s5: ${s5}px; --rt-s6: ${s6}px; --rt-s7: ${s7}px; --rt-s8: ${s8}px;

  /* 字阶 */
  --rt-fs-display: ${fs_display}px;
  --rt-fs-title: ${fs_title}px;
  --rt-fs-lead: ${fs_lead}px;
  --rt-fs-body: ${fs_body}px;
  --rt-fs-no: ${fs_no}px;
  --rt-fs-sm: ${fs_sm}px;
  --rt-fs-label: ${fs_label}px;
  --rt-fs-brand: ${fs_brand}px;

  /* 圆角与线宽 */
  --rt-r-sm: ${r_sm}px;
  --rt-r-md: ${r_md}px;
  --rt-r-pill: ${r_pill}px;
  --rt-line: ${line_hair}px;
  --rt-line-strong: ${line_strong}px;

  /* 行高与字间距 */
  --rt-lh-tight: ${lh_tight};
  --rt-lh-body: ${lh_body};
  --rt-lh-loose: ${lh_loose};
  --rt-ls-head: ${ls_head}em;
  --rt-ls-label: ${ls_label}em;
  --rt-ls-btn: ${ls_btn}em;

  /* 版面常量 */
  --rt-max-w: ${max_w}px;
  --rt-w-key: ${w_key}px;
  --rt-w-no: ${w_no}px;
  --rt-logo-h: ${logo_h}px;

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
[data-testid="stMarkdownContainer"] p { color: var(--rt-text); line-height: var(--rt-lh-body); }
h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {
  font-family: var(--rt-serif);
  letter-spacing: var(--rt-ls-head);
  color: var(--rt-text);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: var(--rt-s6); padding-bottom: var(--rt-s7); max-width: var(--rt-max-w); }

/* 侧边栏也用纸白，让 logo 图标融进页面（图标底色是米白），
   用一条右边线区分左右两栏 */
[data-testid="stSidebar"] { background: var(--rt-bg); border-right: var(--rt-line) solid var(--rt-border); }
[data-testid="stSidebarContent"] { padding-top: var(--rt-s5); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { font-family: var(--rt-serif); }

/* ---------- 顶栏 ---------- */
.rt-topbar {
  display: flex; align-items: center; gap: var(--rt-s5);
  padding: 0 var(--rt-s1) var(--rt-s5) var(--rt-s1); margin-bottom: var(--rt-s6);
  border-bottom: var(--rt-line-strong) solid var(--rt-text);
}
.rt-topbar img { height: var(--rt-logo-h); border-radius: var(--rt-r-md); }
.rt-topbar-title { font-family: var(--rt-serif); font-size: var(--rt-fs-title); font-weight: 600; }
.rt-topbar-spacer { flex: 1; }

/* ---------- 分区：靠分割线而不是卡片 ---------- */
/* 用 padding-top 而不是 margin-top：Streamlit 的 vertical block 会吃掉相邻
   元素的外边距，margin 撑不开间距。
   注意别写 :first-child 清零——每个 section 都是它所在容器的首个子元素，
   那样会把所有 section 的上间距都清掉。 */
.rt-section { padding-top: var(--rt-s7); }
.rt-section-head {
  display: flex; align-items: center; gap: var(--rt-s4);
  font-size: var(--rt-fs-label); font-weight: 600; letter-spacing: var(--rt-ls-label);
  color: var(--rt-text-mute); margin-bottom: var(--rt-s4);
}
.rt-section-head::after {
  content: ""; flex: 1; height: var(--rt-line); background: var(--rt-border);
}
.rt-section-head.is-accent { color: var(--rt-accent); }

/* ---------- 问答 ---------- */
.rt-q { font-size: var(--rt-fs-title); line-height: var(--rt-lh-tight); margin-bottom: var(--rt-s6); }
.rt-q b { font-weight: 600; }
.rt-a { font-size: var(--rt-fs-lead); line-height: var(--rt-lh-loose); }
.rt-a b { font-weight: 600; }
.rt-ref {
  color: var(--rt-accent); font-weight: 600; font-size: var(--rt-fs-no);
  margin-left: var(--rt-s1); white-space: nowrap; vertical-align: super;
}

/* ---------- 影像灯箱 ---------- */
.rt-plate {
  background: var(--rt-surface); padding: var(--rt-s5);
  border-radius: var(--rt-r-md);
}
.rt-plate img { width: 100%; display: block; border-radius: var(--rt-r-sm); }
.rt-plate-cap {
  display: flex; justify-content: space-between; gap: var(--rt-s4);
  font-size: var(--rt-fs-sm); color: var(--rt-text-mute); margin-top: var(--rt-s4);
}

/* ---------- 轨迹行 ---------- */
.rt-rows { font-size: var(--rt-fs-body); }
.rt-row {
  display: grid; grid-template-columns: var(--rt-w-key) 1fr auto; gap: var(--rt-s5);
  align-items: baseline; padding: var(--rt-s3) 0;
  border-bottom: var(--rt-line) dotted var(--rt-border);
}
.rt-row:last-child { border-bottom: 0; }
.rt-row-key { font-weight: 600; }
.rt-row-val { color: var(--rt-text-dim); }
.rt-row-meta { color: var(--rt-text-mute); font-size: var(--rt-fs-sm); white-space: nowrap; }
.rt-row.is-hit .rt-row-key { color: var(--rt-accent); }

/* ---------- 证据列表 ---------- */
.rt-refs { font-size: var(--rt-fs-body); }
.rt-ref-item {
  display: grid; grid-template-columns: var(--rt-w-no) 1fr; gap: var(--rt-s5);
  padding: var(--rt-s4) 0; border-top: var(--rt-line) solid var(--rt-border);
}
.rt-ref-item:first-child { border-top: var(--rt-line-strong) solid var(--rt-text); }
.rt-ref-item.is-missed { opacity: 0.5; }
.rt-ref-no { color: var(--rt-accent); font-weight: 600; font-size: var(--rt-fs-no); }
.rt-ref-item.is-missed .rt-ref-no { color: var(--rt-text-mute); }
.rt-ref-body p { margin: 0 0 var(--rt-s2) 0; line-height: var(--rt-lh-body); }
.rt-ref-src { font-size: var(--rt-fs-sm); color: var(--rt-text-mute); }

/* ---------- 指标（细排，不做卡片块） ---------- */
.rt-stats { display: flex; gap: var(--rt-s6); flex-wrap: wrap; }
.rt-stat-v {
  font-family: var(--rt-serif); font-size: var(--rt-fs-display); font-weight: 600;
}
.rt-stat-l { font-size: var(--rt-fs-label); color: var(--rt-text-mute); margin-top: var(--rt-s1); }

/* ---------- 对照组切换 ---------- */
[data-testid="stRadio"] > div { gap: 0; }
[data-testid="stRadio"] label {
  border: var(--rt-line) solid var(--rt-border-2); border-radius: var(--rt-r-pill);
  padding: var(--rt-s2) var(--rt-s5); background: transparent; margin-right: 0;
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
  border: var(--rt-line) solid var(--rt-border-2) !important;
  border-radius: var(--rt-r-sm) !important;
  color: var(--rt-text) !important;
  font-family: var(--rt-sans) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--rt-text) !important; box-shadow: none !important;
}
.stButton > button {
  background: var(--rt-text); color: var(--rt-bg);
  border: var(--rt-line) solid var(--rt-text); border-radius: var(--rt-r-sm);
  font-weight: 600; width: 100%; letter-spacing: var(--rt-ls-btn);
  transition: opacity 0.15s ease;
}
/* 按钮文字是 markdown 容器里的 p，会被上面的 p 颜色规则改成墨色，
   墨字压在墨底上看不见，这里显式改回来。 */
.stButton > button p,
.stButton > button span { color: var(--rt-bg) !important; }
.stButton > button:hover { opacity: 0.85; color: var(--rt-bg); }
[data-testid="stFileUploader"] section {
  background: var(--rt-surface-2);
  border: var(--rt-line) dashed var(--rt-border-2);
  border-radius: var(--rt-r-sm);
}
[data-testid="stFileUploaderDropzoneInstructions"] > * { visibility: hidden; height: 0; }
[data-testid="stFileUploader"] button { color: var(--rt-text-dim) !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * { font-size: 0 !important; }
[data-testid="stFileUploader"] button::after {
  content: "选择文件"; font-size: var(--rt-fs-no) !important; font-weight: 500;
}
[data-testid="stFileUploader"] small { color: var(--rt-text-mute) !important; }
[data-testid="stSlider"] [role="slider"] { background-color: var(--rt-accent) !important; }
[data-testid="stSlider"] [data-testid="stSliderTickBarMin"] { background-color: var(--rt-accent) !important; }
[data-testid="stCaptionContainer"] { color: var(--rt-text-mute); }
[data-testid="stExpander"] { border: var(--rt-line) solid var(--rt-border); border-radius: var(--rt-r-sm); }
hr { border-color: var(--rt-border); }
a { color: var(--rt-accent); }
</style>
"""


CSS = Template(CSS).safe_substitute(
    **COLORS,
    **{
        **{f"s{k[1:]}": v for k, v in SPACING.items()},
        **{f"fs_{k}": v for k, v in TYPE.items()},
        **{f"r_{k}": v for k, v in RADIUS.items()},
        **{f"line_{k}": v for k, v in LINE.items()},
        **{f"lh_{k}": v for k, v in LEADING.items()},
        **{f"ls_{k}": v for k, v in TRACKING.items()},
        **LAYOUT,
    },
    sans=SANS_STACK,
    serif=SERIF_STACK,
    mono=MONO_STACK,
)


def apply_theme() -> None:
    """把主题注入到当前 Streamlit 页面。"""
    st.markdown(CSS, unsafe_allow_html=True)
