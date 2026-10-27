"""界面组件。

每个组件都只负责渲染，不做业务判断。数据来自 radtracerag.contracts。

两个坑值得记一下：

1. Streamlit 用 CommonMark 渲染 HTML，遇到空行就结束当前 HTML 块，后面全部
   当成字面量显示。所以这里所有 HTML 片段都先 strip() 再拼接，块与块之间
   不留空行。
2. Streamlit 的 markdown 组件不放行 file:// 图片，图标要转成 data URI。
"""

from __future__ import annotations

import base64
import html
import mimetypes
import re
from pathlib import Path

import streamlit as st

from radtracerag.contracts import Answer, Evidence, ImageObservation, RetrievalStep
from radtracerag.ui.theme import COLORS

_CITE = re.compile(r"\[(E\d+)\]")


def _esc(text: str) -> str:
    return html.escape(str(text))


def _join(parts: list[str]) -> str:
    """拼 HTML 片段。逐个 strip 掉首尾换行，块间不留空行。"""
    return "".join(p.strip() for p in parts)


def img_uri(path: Path | str | None) -> str:
    """把本地图片转成 data URI。file:// 在 Streamlit 里会被拦。"""
    if not path:
        return ""
    path = Path(path)
    if not path.exists():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


# ---------------------------------------------------------------- 顶栏


def topbar(logo: Path | None = None, subtitle: str = "TRACEABLE MULTIMODAL RAG") -> None:
    logo_html = ""
    uri = img_uri(logo)
    if uri:
        logo_html = f'<img src="{uri}" alt="RadTraceRAG">'
    st.markdown(
        f"""
        <div class="rt-topbar">
          {logo_html}
          <div>
            <div class="rt-topbar-title">RadTraceRAG</div>
            <div class="rt-topbar-sub">{_esc(subtitle)}</div>
          </div>
          <div class="rt-topbar-spacer"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 影像查看器


def image_viewer(image_path: Path | None, caption: str = "", meta: str = "") -> None:
    """深底影像查看器，中间叠一个十字准星（呼应 logo）。"""
    uri = img_uri(image_path)
    if not uri:
        st.markdown(
            f"""
            <div class="rt-viewer" style="min-height:260px;display:flex;
                        align-items:center;justify-content:center;">
              <span style="color:{COLORS['text_mute']};font-size:13px;">
                等待上传影像
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="rt-viewer">
          <img src="{uri}">
          <div class="rt-crosshair-h"></div>
          <div class="rt-crosshair-v"></div>
        </div>
        <div class="rt-viewer-caption">
          <span>{_esc(caption)}</span>
          <span>{_esc(meta)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 答案


def answer_card(answer: Answer, evidence: list[Evidence]) -> None:
    """答案卡。答案里 [E1] 形式的引用会高亮，引用列表单列在下面。"""
    cited = set(answer.evidence_ids)

    def mark(match: re.Match) -> str:
        eid = match.group(1)
        color = COLORS["hit"] if eid in cited else COLORS["text_mute"]
        return (
            f'<span class="rt-answer-mark" style="border-bottom:2px solid {color}">'
            f"{match.group(0)}</span>"
        )

    body = _CITE.sub(mark, _esc(answer.text))
    chips = "".join(
        f'<span class="rt-badge is-hit is-mono">{_esc(eid)}</span>'
        for eid in answer.evidence_ids
    )

    st.markdown(
        f"""
        <div class="rt-card">
          <div class="rt-card-head is-hit"><span class="rt-dot"></span>答案</div>
          <div class="rt-answer">{body}</div>
          <div class="rt-answer-cites">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 图像观察


def observation_card(observation: ImageObservation | None) -> None:
    if observation is None:
        return
    chips = "".join(
        f'<span class="rt-badge is-dim">{_esc(e)}</span>' for e in observation.entities
    )
    st.markdown(
        f"""
        <div class="rt-card">
          <div class="rt-card-head is-scan"><span class="rt-dot"></span>图像观察</div>
          <div style="font-size:14.5px;line-height:1.6;color:{COLORS['text']};">
            {_esc(observation.text)}
          </div>
          <div class="rt-answer-cites">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 证据


def evidence_list(evidence: list[Evidence], cited_ids: tuple[str, ...] = ()) -> None:
    """证据卡列表。被答案引用的证据高亮，没被引用的压暗。"""
    cited = set(cited_ids)
    if not evidence:
        st.markdown(
            f"""
            <div class="rt-card">
              <div class="rt-card-head is-hit"><span class="rt-dot"></span>证据</div>
              <div style="color:{COLORS['text_mute']};font-size:13.5px;">
                没有检索到证据
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    blocks = []
    for ev in evidence:
        hit = ev.evidence_id in cited
        cls = "rt-evidence" if hit else "rt-evidence is-missed"
        mark = (
            '<span class="rt-badge is-hit">已引用</span>'
            if hit
            else '<span class="rt-badge is-dim">未引用</span>'
        )
        blocks.append(
            f"""
            <div class="{cls}">
              <div class="rt-evidence-head">
                <span class="rt-evidence-id">{_esc(ev.evidence_id)}</span>
                {mark}
              </div>
              <div class="rt-evidence-text">{_esc(ev.content)}</div>
              <div class="rt-evidence-source">来源：{_esc(ev.source)}</div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="rt-card">
          <div class="rt-card-head is-hit"><span class="rt-dot"></span>证据</div>
          {_join(blocks)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 检索轨迹


def retrieval_trace(steps: list[RetrievalStep]) -> None:
    """检索轨迹时间线。这是「答案怎么来的」的展开。"""
    if not steps:
        return
    blocks = []
    for step in steps:
        is_hit = step.stage in {"命中", "证据", "evidence"}
        cls = "rt-step is-hit" if is_hit else "rt-step"
        meta = []
        if step.count:
            meta.append(f"{step.count} 条")
        if step.elapsed_ms:
            meta.append(f"{step.elapsed_ms} ms")
        meta_html = (
            f'<div class="rt-step-meta">{" · ".join(_esc(m) for m in meta)}</div>'
            if meta
            else ""
        )
        blocks.append(
            f"""
            <div class="{cls}">
              <span class="rt-dot"></span>
              <div class="rt-step-title">{_esc(step.stage)}</div>
              <div class="rt-step-detail">{_esc(step.detail)}</div>
              {meta_html}
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="rt-card">
          <div class="rt-card-head"><span class="rt-dot"></span>检索轨迹</div>
          <div class="rt-trace">{_join(blocks)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 指标


def metrics_row(items: list[tuple[str, str, str]]) -> None:
    """指标行。items = [(值, 标签, 颜色), ...]"""
    blocks = []
    for value, label, color in items:
        blocks.append(
            f"""
            <div class="rt-metric">
              <div class="rt-metric-value" style="color:{color}">{_esc(value)}</div>
              <div class="rt-metric-label">{_esc(label)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="rt-metrics">{_join(blocks)}</div>', unsafe_allow_html=True)
