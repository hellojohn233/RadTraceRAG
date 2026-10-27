"""RadTraceRAG 界面设计 token 与主题。

颜色语义和 logo 严格一致，这是整套设计语言的核心：

    青（trace）  检索路径、轨迹、系统动作 —— 对应 logo 里被点亮的那段弧
    琥珀（hit）  命中的证据、引用          —— 对应 logo 里的琥珀点
    灰蓝（scan） 图像观察、看片视野         —— 对应 logo 里的中心准星

界面默认深色。医学影像是灰阶的，深底上对比度和细节都更好；看片软件（PACS）
也是这个惯例，不是为了好看而选的。
"""

from __future__ import annotations

from string import Template

import streamlit as st

# ---------------------------------------------------------------- 设计 token

COLORS = {
    "bg": "#0A0F18",
    "surface": "#111927",
    "surface_2": "#0E1622",
    "border": "#1F2C3D",
    "border_2": "#2A3A4F",
    "text": "#E6EDF7",
    "text_dim": "#93A7C0",
    "text_mute": "#61748D",
    # 检索 / 轨迹
    "trace": "#38BDF8",
    "trace_2": "#22D3EE",
    "trace_3": "#8B93F8",
    # 命中的证据
    "hit": "#FBBF24",
    "hit_2": "#F59E0B",
    # 图像观察
    "scan": "#7FA0C0",
    "ok": "#34D399",
    "bad": "#F87171",
    "warn": "#FB923C",
}

FONT_STACK = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', "
    "'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif"
)

MONO_STACK = (
    "'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', "
    "'Liberation Mono', 'Courier New', monospace"
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
  --rt-trace: $trace;
  --rt-trace-2: $trace_2;
  --rt-trace-3: $trace_3;
  --rt-hit: $hit;
  --rt-hit-2: $hit_2;
  --rt-scan: $scan;
  --rt-ok: $ok;
  --rt-bad: $bad;
  --rt-font: $font;
  --rt-mono: $mono;
}

/* ---------- 底色与字体 ---------- */
.stApp { background: var(--rt-bg); }
[data-testid="stAppViewContainer"] {
  --primary-color: $trace;
  --primary-color-hover: $trace_2;
  --primary-color-active: #0EA5E9;
  --background-color: var(--rt-bg);
  --secondary-background-color: var(--rt-surface-2);
  --text-color: var(--rt-text);
}
html, body, [class*="css"], [data-testid="stMarkdownContainer"] p, h1, h2, h3, h4 {
  font-family: var(--rt-font);
  color: var(--rt-text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 0; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1440px; }

[data-testid="stSidebar"] {
  background: #0C131F;
  border-right: 1px solid var(--rt-border);
}
[data-testid="stSidebarContent"] { padding-top: 1.4rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  letter-spacing: -0.01em;
}

/* ---------- 顶栏 ---------- */
.rt-topbar {
  display: flex; align-items: center; gap: 16px;
  padding: 0 4px 18px 4px; margin-bottom: 18px;
  border-bottom: 1px solid var(--rt-border);
}
.rt-topbar img { height: 44px; border-radius: 10px; }
.rt-topbar-title { font-size: 19px; font-weight: 650; letter-spacing: -0.02em; }
.rt-topbar-sub { font-size: 12.5px; color: var(--rt-text-mute); letter-spacing: 0.14em; text-transform: uppercase; }
.rt-topbar-spacer { flex: 1; }

/* ---------- 卡片 ---------- */
.rt-card {
  background: var(--rt-surface);
  border: 1px solid var(--rt-border);
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 12px;
}
.rt-card-head {
  display: flex; align-items: center; gap: 9px;
  font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--rt-text-mute); margin-bottom: 12px;
}
.rt-card-head .rt-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--rt-trace); }
.rt-card-head.is-hit .rt-dot { background: var(--rt-hit); }
.rt-card-head.is-scan .rt-dot { background: var(--rt-scan); }

/* ---------- 答案 ---------- */
.rt-answer {
  font-size: 19px; line-height: 1.62; color: var(--rt-text);
  letter-spacing: -0.005em;
}
.rt-answer-mark {
  background: linear-gradient(180deg, transparent 62%, rgba(56,189,248,0.28) 62%);
  padding: 0 2px;
}
.rt-answer-cites { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; }

/* ---------- 徽章 ---------- */
.rt-badge {
  display: inline-block; padding: 3px 9px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600; line-height: 1.5;
  border: 1px solid var(--rt-border-2); color: var(--rt-text-dim);
  background: var(--rt-surface-2);
}
.rt-badge.is-hit { color: #1A1206; background: var(--rt-hit); border-color: var(--rt-hit); }
.rt-badge.is-trace { color: #05121E; background: var(--rt-trace); border-color: var(--rt-trace); }
.rt-badge.is-mono { font-family: var(--rt-mono); font-weight: 500; letter-spacing: -0.01em; }
.rt-badge.is-ok { color: #052E20; background: var(--rt-ok); border-color: var(--rt-ok); }
.rt-badge.is-bad { color: #2B0A0A; background: var(--rt-bad); border-color: var(--rt-bad); }
.rt-badge.is-dim { opacity: 0.72; }

/* ---------- 证据卡 ---------- */
.rt-evidence {
  background: var(--rt-surface-2);
  border: 1px solid var(--rt-border);
  border-left: 3px solid var(--rt-hit);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 9px;
}
.rt-evidence.is-missed { border-left-color: var(--rt-border-2); opacity: 0.62; }
.rt-evidence-head {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  margin-bottom: 7px;
}
.rt-evidence-id {
  font-family: var(--rt-mono); font-size: 12px; font-weight: 600;
  color: var(--rt-hit); letter-spacing: -0.01em;
}
.rt-evidence.is-missed .rt-evidence-id { color: var(--rt-text-mute); }
.rt-evidence-text { font-size: 14.5px; line-height: 1.58; color: var(--rt-text); }
.rt-evidence-source { font-size: 12px; color: var(--rt-text-mute); margin-top: 8px; }

/* ---------- 检索轨迹 ---------- */
.rt-trace { position: relative; padding-left: 20px; }
.rt-trace::before {
  content: ""; position: absolute; left: 4px; top: 8px; bottom: 10px; width: 2px;
  background: linear-gradient(180deg, var(--rt-trace-2), var(--rt-trace-3));
  opacity: 0.45;
}
.rt-step { position: relative; margin-bottom: 15px; }
.rt-step:last-child { margin-bottom: 0; }
.rt-dot {
  position: absolute; left: -20px; top: 4px; width: 10px; height: 10px;
  border-radius: 50%; background: var(--rt-trace);
  box-shadow: 0 0 0 4px rgba(56,189,248,0.14);
}
.rt-step.is-hit .rt-dot { background: var(--rt-hit); box-shadow: 0 0 0 4px rgba(251,191,36,0.14); }
.rt-step-title { font-size: 13.5px; font-weight: 600; color: var(--rt-text); }
.rt-step-detail { font-size: 13px; color: var(--rt-text-dim); margin-top: 3px; line-height: 1.5; }
.rt-step-meta { font-size: 11.5px; color: var(--rt-text-mute); margin-top: 4px; font-family: var(--rt-mono); }

/* ---------- 影像查看器 ---------- */
.rt-viewer {
  position: relative; background: #04070C;
  border: 1px solid var(--rt-border); border-radius: 14px;
  padding: 12px; overflow: hidden;
}
.rt-viewer img { width: 100%; display: block; border-radius: 8px; }
.rt-crosshair-h, .rt-crosshair-v {
  position: absolute; background: rgba(127,160,192,0.34); pointer-events: none;
}
.rt-crosshair-h { left: 12px; right: 12px; top: 50%; height: 1px; }
.rt-crosshair-v { top: 12px; bottom: 12px; left: 50%; width: 1px; }
.rt-viewer-caption {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 11.5px; color: var(--rt-text-mute); margin-top: 9px;
  font-family: var(--rt-mono);
}

/* ---------- 指标 ---------- */
.rt-metrics { display: flex; gap: 10px; flex-wrap: wrap; }
.rt-metric {
  flex: 1 1 120px; background: var(--rt-surface-2);
  border: 1px solid var(--rt-border); border-radius: 10px; padding: 10px 12px;
}
.rt-metric-value { font-size: 20px; font-weight: 650; letter-spacing: -0.02em; }
.rt-metric-label { font-size: 11.5px; color: var(--rt-text-mute); margin-top: 2px; }

/* ---------- 对照组切换 ---------- */
[data-testid="stRadio"] > div { gap: 6px; }
[data-testid="stRadio"] label {
  border: 1px solid var(--rt-border); border-radius: 9px;
  padding: 6px 12px; background: var(--rt-surface-2);
  transition: all 0.15s ease;
}
[data-testid="stRadio"] label:has(input:checked) {
  border-color: var(--rt-trace); background: rgba(56,189,248,0.10);
}
[data-testid="stRadio"] label:hover { border-color: var(--rt-border-2); }
[data-testid="stRadio"] input[type="radio"] { accent-color: var(--rt-trace); }
[data-testid="stRadio"] input[type="radio"] + div { color: var(--rt-text); }

/* ---------- 滑块 ---------- */
[data-testid="stSlider"] [role="slider"] {
  background-color: var(--rt-trace) !important;
  box-shadow: 0 0 0 4px rgba(56,189,248,0.16);
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTickBarMin"] {
  background: var(--rt-trace) !important;
}

/* ---------- 上传区 ---------- */
[data-testid="stFileUploader"] section {
  background: var(--rt-surface-2); border: 1px dashed var(--rt-border-2); border-radius: 12px;
}
/* Streamlit 的上传提示是英文，藏掉换成中文。
   按钮里既有 icon 又有文字，还有多个并列 span，逐个匹配会漏。
   干脆把整块文字压成 0，再在 button 上加一个 ::after。 */
[data-testid="stFileUploaderDropzoneInstructions"] > * { visibility: hidden; height: 0; }
[data-testid="stFileUploader"] button { color: var(--rt-text-dim) !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * { font-size: 0 !important; }
[data-testid="stFileUploader"] button::after {
  content: "选择文件"; font-size: 13px !important; font-weight: 500;
}
[data-testid="stFileUploader"] small { color: var(--rt-text-mute) !important; }

/* ---------- 窄屏堆叠 ---------- */
@media (max-width: 1250px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stColumn"] { min-width: 100% !important; }
}

/* ---------- 控件 ---------- */
.stTextInput input, .stTextArea textarea {
  background: var(--rt-surface-2) !important;
  border: 1px solid var(--rt-border-2) !important;
  border-radius: 10px !important;
  color: var(--rt-text) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--rt-trace) !important; box-shadow: 0 0 0 1px var(--rt-trace) !important;
}
.stButton > button {
  background: linear-gradient(135deg, var(--rt-trace-2), var(--rt-trace));
  color: #04121D; border: none; border-radius: 10px;
  font-weight: 650; letter-spacing: -0.01em; width: 100%;
  transition: filter 0.15s ease;
}
.stButton > button:hover { filter: brightness(1.08); color: #04121D; }
.stDownloadButton > button {
  background: var(--rt-surface-2); color: var(--rt-text-dim);
  border: 1px solid var(--rt-border-2); border-radius: 10px; width: 100%;
}
[data-testid="stFileUploader"] section {
  background: var(--rt-surface-2); border: 1px dashed var(--rt-border-2); border-radius: 12px;
}
[data-testid="stExpander"] {
  border: 1px solid var(--rt-border); border-radius: 12px; background: var(--rt-surface);
}
[data-testid="stExpander"] summary { font-size: 13.5px; color: var(--rt-text-dim); }

/* ---------- 杂项 ---------- */
hr { border-color: var(--rt-border); }
[data-testid="stCaptionContainer"] { color: var(--rt-text-mute); }
a { color: var(--rt-trace); }
</style>
"""

CSS = Template(CSS).safe_substitute(
    **COLORS,
    font=FONT_STACK,
    mono=MONO_STACK,
)


def apply_theme() -> None:
    """把主题注入到当前 Streamlit 页面。"""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .stApp [data-testid="stMetricValue"] { font-family: var(--rt-mono); }
        </style>
        """,
        unsafe_allow_html=True,
    )
