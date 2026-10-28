"""RadTraceRAG 演示界面。

    uv run radtracerag-ui          # 启动后打开 http://localhost:8501

界面的排版参照放射科报告和论文：左边摊着片子，右边从上到下是
「问 → 答 → 图像观察 → 检索轨迹 → 证据」，证据按脚注编号列在最后。
整页读下来是一份「答案 + 它凭什么这么说」的文档。

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
from radtracerag.ui.theme import apply_theme

ASSETS = Path(__file__).resolve().parents[3] / "assets"

st.set_page_config(
    page_title="RadTraceRAG",
    page_icon=str(ASSETS / "png" / "logo-mark-favicon-180.png"),
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
    list[tuple[str, str]],
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
        <div style="display:flex;align-items:center;gap:var(--rt-s4);margin-bottom:var(--rt-s6);">
          <img src="{logo_uri}" style="height:36px;border-radius:var(--rt-r-md);">
          <div style="font-family:'Songti SC',Georgia,serif;font-size:var(--rt-fs-brand);font-weight:600;">
            RadTraceRAG
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
    st.caption("PNG / JPG / DICOM，单张不超过 200 MB")

    st.markdown("")
    st.markdown("**问题**")
    question = st.text_area(
        "对这张影像提问",
        value="图中主要器官的作用是什么？",
        height=88,
        label_visibility="collapsed",
    )

    st.markdown("")
    st.markdown("**检索深度**")
    top_k = st.slider("Top-K", 1, 10, 4)
    st.caption(MODES["C · 图增强 RAG"]["desc"])

    st.markdown("")
    ask = st.button("提问", type="primary", use_container_width=True)

    st.markdown("")
    st.markdown(
        """
        <div style="font-size:11.5px;color:#8A94A2;line-height:1.7;
                    border-top:1px solid #E2E0DA;padding-top:12px;">
          演示模式：影像与答案均为合成样例<br>
          非真实患者数据，不用于诊断
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- 顶栏 + 对照组

head, ctrl = st.columns([3, 2], gap="medium", vertical_alignment="center")
with head:
    C.topbar(logo=ASSETS / "png" / "logo-mark-128.png")
with ctrl:
    mode = st.radio(
        "检索模式",
        list(MODES.keys()),
        index=2,
        horizontal=True,
        label_visibility="collapsed",
        key="rt_mode",
    )

# ---------------------------------------------------------------- 主区

image_path = save_upload(uploaded)
if image_path is None:
    image_path = load_demo(mode).image

answer, observation, evidence, trace, metrics = run_question(image_path, question, mode)

left, right = st.columns([1, 1.62], gap="large")

with left:
    C.image_plate(
        image_path,
        caption=Path(image_path).name,
        meta="512 × 512 · 合成影像" if uploaded is None else "已上传",
    )
    st.markdown("")
    C.stat_row(metrics)

with right:
    C.question_line(question)
    C.answer_block(answer, evidence)
    C.observation_block(observation)
    C.trace_rows(trace)
    C.reference_list(evidence, answer.evidence_ids)

st.markdown("")
st.markdown(
    """
    <div style="font-size:12px;color:#8A94A2;text-align:center;
                padding-top:20px;border-top:1px solid #E2E0DA;">
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
