"""演示数据。

真实链路还没接上，界面先要能跑起来看效果。这里用一组贴近 SLAKE 风格的
假数据把三组对照跑通，影像用程序生成的合成腹部切片。

接上真实链路后，把 load_demo() 换成实际的 pipeline 调用即可，界面不用动。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from radtracerag.contracts import Answer, Evidence, ImageObservation, RetrievalStep
from radtracerag.ui.theme import COLORS

ASSETS = Path(__file__).resolve().parents[3] / "assets"

MODES = {
    "A · 无 RAG": {
        "label": "无 RAG",
        "desc": "只看影像和问题，让模型直接回答",
        "answer": Answer(
            text=(
                "图中腹部横断面影像的主要器官是肝脏，位于右上腹。肝脏负责代谢、解毒和"
                "合成蛋白质，同时生成胆汁帮助脂肪消化。"
            ),
            evidence_ids=(),
        ),
        "observation": ImageObservation(
            text=(
                "腹部轴位断层，可见右上腹大片实质脏器影，密度较周围脂肪高，"
                "边缘清晰，未见明显钙化。周围可见少量低密度积液。"
            ),
            entities=("腹部", "轴位断层", "肝", "腹腔积液"),
        ),
        "evidence": [],
        "trace": [
            RetrievalStep(
                stage="图像理解",
                detail="VLM 读取影像，产出与问题相关的观察",
                elapsed_ms=1840,
            ),
            RetrievalStep(
                stage="直接回答",
                detail="没有检索，仅凭模型参数里的知识作答",
                elapsed_ms=620,
            ),
        ],
        "metrics": [("0", "引用证据", COLORS["text_mute"]), ("1", "模型调用", COLORS["text"])],
    },
    "B · 向量 RAG": {
        "label": "向量 RAG",
        "desc": "LightRAG naive 模式，纯文本向量检索",
        "answer": Answer(
            text=(
                "图中主要器官是肝脏，位于右上腹 [E1]。肝脏承担代谢与解毒、合成白蛋白"
                "与凝血因子，并分泌胆汁参与脂肪消化 [E1]。"
                "同时可见腹腔少量低密度积液，常见于多种病因，不能仅凭单张图定性 [E3]。"
            ),
            evidence_ids=("E1", "E3"),
        ),
        "observation": ImageObservation(
            text=(
                "腹部轴位断层，可见右上腹大片实质脏器影，密度较周围脂肪高，"
                "边缘清晰，未见明显钙化。周围可见少量低密度积液。"
            ),
            entities=("腹部", "轴位断层", "肝", "腹腔积液"),
        ),
        "evidence": [
            Evidence(
                evidence_id="E1",
                content=(
                    "肝是人体最大的腺体，位于右季肋区。功能包括糖异生、糖原储存、"
                    "尿素合成与氨解毒、胆汁分泌，以及合成白蛋白、凝血因子等血浆蛋白。"
                ),
                source="SLAKE knowledge_base · 肝脏条目",
            ),
            Evidence(
                evidence_id="E2",
                content=(
                    "胆汁由肝细胞分泌、储存在胆囊，经胆总管进入十二指肠，"
                    "参与脂肪的乳化与脂溶性维生素吸收。"
                ),
                source="SLAKE knowledge_base · 胆汁条目",
            ),
            Evidence(
                evidence_id="E3",
                content=(
                    "腹腔积液指腹腔内超出正常量的游离液体，可见于肝硬化、心衰、"
                    "结核性腹膜炎、胰腺炎等多种情况，影像表现相似，"
                    "需结合临床与化验鉴别。"
                ),
                source="SLAKE knowledge_base · 腹水条目",
            ),
            Evidence(
                evidence_id="E4",
                content=(
                    "脾位于左季肋区，质地脆，被膜下出血易破裂，"
                    "是人体最大的淋巴器官。"
                ),
                source="SLAKE knowledge_base · 脾脏条目",
            ),
        ],
        "trace": [
            RetrievalStep(
                stage="图像理解",
                detail="VLM 读取影像，产出与问题相关的观察",
                elapsed_ms=1840,
            ),
            RetrievalStep(
                stage="查询构造",
                detail="把图像观察中的实体拼成检索词：肝脏 / 腹腔积液",
                elapsed_ms=95,
            ),
            RetrievalStep(
                stage="向量检索",
                detail="BGE-M3 编码后取 top-k，命中 4 条候选事实",
                elapsed_ms=410,
                count=4,
            ),
            RetrievalStep(
                stage="命中",
                detail="E1、E3 进入答案上下文，E2、E4 落榜",
                elapsed_ms=12,
                count=2,
            ),
            RetrievalStep(
                stage="基于证据回答",
                detail="原图 + 2 条证据生成答案，并回传引用 ID",
                elapsed_ms=1250,
            ),
        ],
        "metrics": [
            ("2", "引用证据", COLORS["text"]),
            ("4", "候选事实", COLORS["text_dim"]),
            ("3.6", "总耗时 s", COLORS["text_dim"]),
        ],
    },
    "C · 图增强 RAG": {
        "label": "图增强 RAG",
        "desc": "LightRAG mix 模式，图关系 + 向量联合检索",
        "answer": Answer(
            text=(
                "图中主要器官是肝脏，位于右上腹 [E1]。肝脏承担代谢与解毒、合成白蛋白"
                "与凝血因子，并分泌胆汁参与脂肪消化 [E1][E2]。"
                "同时可见腹腔少量低密度积液，常见于多种病因，不能仅凭单张图定性 [E3]。"
                "脾区未见明显异常 [E4]。"
            ),
            evidence_ids=("E1", "E2", "E3", "E4"),
        ),
        "observation": ImageObservation(
            text=(
                "腹部轴位断层，可见右上腹大片实质脏器影，密度较周围脂肪高，"
                "边缘清晰，未见明显钙化。周围可见少量低密度积液。"
            ),
            entities=("腹部", "轴位断层", "肝", "胆囊", "脾", "腹腔积液"),
        ),
        "evidence": [
            Evidence(
                evidence_id="E1",
                content=(
                    "肝是人体最大的腺体，位于右季肋区。功能包括糖异生、糖原储存、"
                    "尿素合成与氨解毒、胆汁分泌，以及合成白蛋白、凝血因子等血浆蛋白。"
                ),
                source="SLAKE knowledge_base · 肝脏条目",
            ),
            Evidence(
                evidence_id="E2",
                content=(
                    "胆汁由肝细胞分泌、储存在胆囊，经胆总管进入十二指肠，"
                    "参与脂肪的乳化与脂溶性维生素吸收。"
                ),
                source="SLAKE knowledge_base · 胆汁条目",
            ),
            Evidence(
                evidence_id="E3",
                content=(
                    "腹腔积液指腹腔内超出正常量的游离液体，可见于肝硬化、心衰、"
                    "结核性腹膜炎、胰腺炎等多种情况，影像表现相似，"
                    "需结合临床与化验鉴别。"
                ),
                source="SLAKE knowledge_base · 腹水条目",
            ),
            Evidence(
                evidence_id="E4",
                content=(
                    "脾位于左季肋区，质地脆，被膜下出血易破裂，"
                    "是人体最大的淋巴器官。"
                ),
                source="SLAKE knowledge_base · 脾脏条目",
            ),
        ],
        "trace": [
            RetrievalStep(
                stage="图像理解",
                detail="VLM 读取影像，产出与问题相关的观察",
                elapsed_ms=1840,
            ),
            RetrievalStep(
                stage="查询构造",
                detail="把图像观察中的实体拼成检索词：肝脏 / 胆囊 / 脾 / 腹腔积液",
                elapsed_ms=95,
            ),
            RetrievalStep(
                stage="向量检索",
                detail="BGE-M3 编码后取 top-k",
                elapsed_ms=390,
                count=4,
            ),
            RetrievalStep(
                stage="图扩展",
                detail="沿实体关系扩一跳，带出胆汁、脾脏等间接相关事实",
                elapsed_ms=540,
                count=4,
            ),
            RetrievalStep(
                stage="命中",
                detail="4 条证据全部进入答案上下文",
                elapsed_ms=12,
                count=4,
            ),
            RetrievalStep(
                stage="基于证据回答",
                detail="原图 + 4 条证据生成答案，并回传引用 ID",
                elapsed_ms=1620,
            ),
        ],
        "metrics": [
            ("4", "引用证据", COLORS["text"]),
            ("8", "候选事实", COLORS["text_dim"]),
            ("4.5", "总耗时 s", COLORS["text_dim"]),
        ],
    },
}


@dataclass(frozen=True)
class DemoCase:
    image: Path
    question: str
    answer: Answer
    observation: ImageObservation
    evidence: list[Evidence]
    trace: list[RetrievalStep]
    metrics: list[tuple[str, str, str]]
    desc: str


def make_demo_image(path: Path | None = None) -> Path:
    """生成一张合成的腹部断层影像。

    不是真片子，只是有 CT 观感的灰阶结构：黑背景、分层的体壁轮廓、
    右上腹一片均匀高密度的肝、两侧肾影、中后方的椎体亮点和椎前主动脉。
    用软边椭圆遮罩而不是高斯叠加，才能得到 CT 那种边界清晰的结构。
    """
    path = path or (ASSETS / "demo" / "abdominal-ct.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]

    def soft_ellipse(cy_: float, cx_: float, ry: float, rx: float, edge: float = 0.10) -> np.ndarray:
        d = ((yy - h * cy_) / (h * ry)) ** 2 + ((xx - w * cx_) / (w * rx)) ** 2
        return np.clip((1.0 - d) / edge, 0, 1).astype(np.float32)

    def layer(mask: np.ndarray, value: float) -> np.ndarray:
        return mask * value

    img = np.zeros((h, w), dtype=np.float32)

    # 腹内软组织底
    body = soft_ellipse(0.50, 0.50, 0.45, 0.41, 0.30)
    img = np.maximum(img, layer(body, 0.20))
    # 皮下脂肪层：身体轮廓再往外一圈
    fat = soft_ellipse(0.50, 0.50, 0.485, 0.445, 0.20) - body
    img = np.maximum(img, layer(np.clip(fat, 0, 1), 0.34))
    # 肝：右上腹大片均匀高密度
    img = np.maximum(img, layer(soft_ellipse(0.40, 0.60, 0.285, 0.235, 0.42), 0.30))
    # 脾：左上腹
    img = np.maximum(img, layer(soft_ellipse(0.44, 0.36, 0.195, 0.155, 0.42), 0.26))
    # 两侧肾影
    for kx in (0.255, 0.745):
        img = np.maximum(img, layer(soft_ellipse(0.505, kx, 0.115, 0.078, 0.44), 0.40))
    # 椎体：骨，最亮
    img = np.maximum(img, layer(soft_ellipse(0.600, 0.50, 0.062, 0.055, 0.34), 0.70))
    # 椎体棘突
    img = np.maximum(img, layer(soft_ellipse(0.655, 0.50, 0.045, 0.016, 0.40), 0.55))
    # 腹主动脉：椎体前方的亮点
    img = np.maximum(img, layer(soft_ellipse(0.508, 0.50, 0.034, 0.029, 0.45), 0.50))
    # 胃肠气腔：低密度黑斑
    gas = soft_ellipse(0.455, 0.425, 0.115, 0.085, 0.50)
    img = np.clip(img - gas * 0.17, 0, 1)

    rng = np.random.default_rng(7)
    img += rng.normal(0, 0.016, img.shape).astype(np.float32)

    img = np.clip(img, 0, 1)
    # CT 重建影像不是锐利的，轻度模糊
    blurred = Image.fromarray((img * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.6)
    )
    img = np.asarray(blurred, dtype=np.float32) / 255.0
    # 窗宽窗位：提中间调，让软组织层次出来
    img = (img - 0.30) * 1.22 + 0.26
    img = np.clip(img, 0, 1)

    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)
    return path


def load_demo(mode: str) -> DemoCase:
    spec = MODES.get(mode, MODES["C · 图增强 RAG"])
    image = make_demo_image()
    return DemoCase(
        image=image,
        question="图中主要器官的作用是什么？",
        answer=spec["answer"],
        observation=spec["observation"],
        evidence=spec["evidence"],
        trace=spec["trace"],
        metrics=spec["metrics"],
        desc=spec["desc"],
    )


__all__ = ["DemoCase", "MODES", "load_demo", "make_demo_image"]
