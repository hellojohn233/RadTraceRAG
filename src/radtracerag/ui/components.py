"""界面组件。

每个组件只负责渲染，不做业务判断。数据来自 radtracerag.contracts。

排版参照的是放射科报告和论文：靠分割线分区，靠编号列表列证据，
不堆卡片。这样一屏读下来就是一份「答案 + 它凭什么这么说」的文档。

两个 Streamlit 的坑值得记一下：

1. HTML 块之间不能有空行。Streamlit 用 CommonMark 渲染 HTML，遇到空行就
   结束当前 HTML 块，后面全部当字面量显示。所以所有片段先 strip() 再拼接。
2. `file://` 图片不显示。markdown 组件不放行本地文件协议，图标和影像
   都要转成 data URI。
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

_CITE = re.compile(r"\[(\d+)\]")


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


# ---------------------------------------------------------------- 影像灯箱


def image_plate(image_path: Path | None, caption: str = "", meta: str = "") -> None:
    """影像放在浅灰灯箱上，像摊在阅片灯上的胶片。"""
    uri = img_uri(image_path)
    if not uri:
        st.markdown(
            f"""
            <div class="rt-plate" style="min-height:300px;display:flex;
                        align-items:center;justify-content:center;">
              <span style="color:{COLORS['text_mute']};font-size:13px;">等待上传影像</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="rt-plate">
          <img src="{uri}">
        </div>
        <div class="rt-plate-cap">
          <span>{_esc(caption)}</span>
          <span>{_esc(meta)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 问答


def question_line(question: str) -> None:
    st.markdown(
        f'<div class="rt-q"><b>问</b>　{_esc(question)}</div>',
        unsafe_allow_html=True,
    )


def answer_block(answer: Answer, evidence: list[Evidence]) -> None:
    """答案段。引用编号用上标，脚注色和 logo 里的朱砂点一致。"""
    cited = set(answer.evidence_ids)

    def mark(match: re.Match) -> str:
        no = match.group(1)
        cls = "rt-ref" if no in cited else "rt-ref-missed"
        style = "" if no in cited else "color:#8A94A2;"
        return f'<span class="{cls}" style="{style}">[{no}]</span>'

    body = _CITE.sub(mark, _esc(answer.text))
    st.markdown(f'<div class="rt-a"><b>答</b>　{body}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- 图像观察


def observation_block(observation: ImageObservation | None) -> None:
    if observation is None:
        return
    st.markdown(
        f"""
        <div class="rt-section">
          <div class="rt-section-head">图像观察</div>
          <div style="font-size:15px;line-height:1.8;color:{COLORS['text_dim']};">
            {_esc(observation.text)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 检索轨迹


def trace_rows(steps: list[RetrievalStep]) -> None:
    """轨迹用表格式行：阶段 | 说明 | 条数与耗时。"""
    if not steps:
        return
    rows = []
    for step in steps:
        is_hit = step.stage in {"命中", "证据", "evidence"}
        cls = "rt-row is-hit" if is_hit else "rt-row"
        meta = []
        if step.count:
            meta.append(f"{step.count} 条")
        if step.elapsed_ms:
            meta.append(f"{step.elapsed_ms} ms")
        rows.append(
            f"""
            <div class="{cls}">
              <div class="rt-row-key">{_esc(step.stage)}</div>
              <div class="rt-row-val">{_esc(step.detail)}</div>
              <div class="rt-row-meta">{_esc(" · ".join(meta))}</div>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="rt-section">
          <div class="rt-section-head">检索轨迹</div>
          <div class="rt-rows">{_join(rows)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 证据


def reference_list(evidence: list[Evidence], cited_ids: tuple[str, ...] = ()) -> None:
    """证据按引用编号列，形态和论文参考文献一致。

    没有检索到证据时也显示这一节——「没证据」本身是要被看见的结果。
    """
    if not evidence:
        st.markdown(
            f"""
            <div class="rt-section">
              <div class="rt-section-head is-accent">证据</div>
              <div style="font-size:14px;color:{COLORS['text_mute']};">
                没有检索到证据，答案没有可回溯的支撑。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cited = set(cited_ids)
    items = []
    for ev in evidence:
        hit = ev.evidence_id in cited
        cls = "rt-ref-item" if hit else "rt-ref-item is-missed"
        items.append(
            f"""
            <div class="{cls}">
              <div class="rt-ref-no">{_esc(ev.evidence_id)}</div>
              <div class="rt-ref-body">
                <p>{_esc(ev.content)}</p>
                <div class="rt-ref-src">{_esc(ev.source)}</div>
              </div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="rt-section">
          <div class="rt-section-head is-accent">证据</div>
          <div class="rt-refs">{_join(items)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 指标


def stat_row(items: list[tuple[str, str]]) -> None:
    """一行细排指标。items = [(值, 标签), ...]"""
    blocks = []
    for value, label in items:
        blocks.append(
            f"""
            <div>
              <div class="rt-stat-v">{_esc(value)}</div>
              <div class="rt-stat-l">{_esc(label)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="rt-stats">{_join(blocks)}</div>', unsafe_allow_html=True)
