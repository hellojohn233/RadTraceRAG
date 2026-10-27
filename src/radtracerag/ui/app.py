"""RadTraceRAG 演示界面。

    uv run radtracerag-ui          # 启动后打开 http://localhost:8501

界面的叙事顺序就是产品的核心承诺：先给答案，再把「它看到了什么、
检索到了什么、命中了哪些证据」一层层摊开。

真实问答链路接上之前，界面跑的是 demo_data 里的假数据。
接入点在 `run_question()` —— 把里面换成 pipeline 调用即可，界面不动。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from radtracerag.contracts import Answer, Evidence, ImageObservation, RetrievalStep
from radtracerag.ui import components as C
from radtracerag.ui.demo_data import MODES, load_demo
from radtracerag.ui.theme import COLORS, apply_theme

ASSETS = Path(__file__).resolve().parents[3] / "assets"


st.set_page_config(
    page_title="RadTraceRAG",
    page_icon=str(ASSETS / "png" / "favicon-180.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


def save_upload(uploaded) -> Path | None:
    """把上传的影像存到临时目录。"""
    if uploaded is None:
        return None
    target = Path(tempfile.gettempdir()) / "radtracerag_uploads" / uploaded.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded.getvalue())
    return target


def run_question(
    image_path: Path | None,
    question: str,
    mode: str,
) -> tuple[
    Answer,
    ImageObservation | None,
    list[Evidence],
    list[RetrievalStep],
    list[tuple[str, str, str]],
]:
    """问一次。真实链路接好后，这里改成 pipeline 调用。"""
    case = load_demo(mode)
    if image_path is not None:
        case = type(case)(
            image=image_path,
            question=question,
            answer=case.answer,
            observation=case.observation,
            evidence=case.evidence,
            trace=case.trace,
            metrics=case.metrics,
            desc=case.desc,
        )
    return case.answer, case.observation, case.evidence, case.trace, case.metrics


# ---------------------------------------------------------------- 侧边栏

with st.sidebar:
    logo_uri = C.img_uri(ASSETS / "png" / "logo-mark-128.png")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:11px;margin-bottom:18px;">
          <img src="{logo_uri}"
               style="height:38px;border-radius:9px;">
          <div>
            <div style="font-size:16px;font-weight:650;letter-spacing:-0.02em;">RadTraceRAG</div>
            <div style="font-size:10.5px;color:{COLORS['text_mute']};letter-spacing:0.13em;">
              TRACEABLE MULTIMODAL RAG
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**影像**")
    uploaded = st.file_uploader(
        "上传 X 光 / CT / MRI",
        type=["png", "jpg", "jpeg", "dcm", "nii"],
        label_visibility="collapsed",
    )
    st.caption("支持 PNG / JPG / DICOM，单张不超过 200 MB")

    st.markdown("")
    st.markdown("**问题**")
    question = st.text_area(
        "对这张影像提问",
        value="图中主要器官的作用是什么？",
        height=90,
        label_visibility="collapsed",
    )

    st.markdown("")
    st.markdown("**检索模式**")
    mode = st.radio(
        "检索模式",
        list(MODES.keys()),
        index=2,
        label_visibility="collapsed",
        key="sidebar_mode",
    )
    st.caption(MODES[mode]["desc"])

    top_k = st.slider("Top-K", 1, 10, 4)

    st.markdown("")
    ask = st.button("提问", type="primary", use_container_width=True)

    st.markdown("")
    st.markdown(
        f"""
        <div style="font-size:11.5px;color:{COLORS['text_mute']};line-height:1.65;
                    border-top:1px solid {COLORS['border']};padding-top:12px;">
          演示模式：数据为合成样例<br>
          非真实患者数据，不用于诊断
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- 主区

C.topbar(logo=ASSETS / "png" / "logo-mark-128.png")

image_path = save_upload(uploaded)
if image_path is None:
    image_path = load_demo(mode).image

if ask or st.session_state.get("rt_ran"):
    st.session_state["rt_ran"] = True
    answer, observation, evidence, trace, metrics = run_question(image_path, question, mode)
else:
    answer, observation, evidence, trace, metrics = run_question(image_path, question, mode)

left, right = st.columns([5, 7], gap="large")

with left:
    C.image_viewer(
        image_path,
        caption=Path(image_path).name,
        meta="512 × 512 · 合成影像" if uploaded is None else "已上传",
    )
    st.markdown("")
    C.metrics_row(metrics)

with right:
    C.answer_card(answer, evidence)

    col_a, col_b = st.columns([1, 1], gap="medium")
    with col_a:
        C.observation_card(observation)
    with col_b:
        C.retrieval_trace(trace)

    C.evidence_list(evidence, answer.evidence_ids)

st.markdown("")
st.markdown(
    f"""
    <div style="font-size:12px;color:{COLORS['text_mute']};text-align:center;
                padding-top:18px;border-top:1px solid {COLORS['border']};">
      RadTraceRAG · 面向医学影像问答的可追溯多模态 RAG ·
      演示数据仅用于界面预览，不构成任何诊断建议
    </div>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """给 `radtracerag-ui` 入口用。Streamlit 自己会执行模块级代码。"""


if __name__ == "__main__":
    main()
