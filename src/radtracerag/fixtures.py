"""演示用的三组对照数据。

界面（Streamlit）和 API 的演示端点共用这一份，避免两边各写一套假数据。
这些内容是合成样例，不是真实医学结论。
"""

from __future__ import annotations

from radtracerag.contracts import (
    Answer,
    Evidence,
    ImageObservation,
    Metric,
    RetrievalStep,
)

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
        "metrics": [Metric("0", "引用证据"), Metric("1", "模型调用")],
    },
    "B · 向量 RAG": {
        "label": "向量 RAG",
        "desc": "LightRAG naive 模式，纯文本向量检索",
        "answer": Answer(
            text=(
                "图中主要器官是肝脏，位于右上腹 [1]。肝脏承担代谢与解毒、合成白蛋白"
                "与凝血因子，并分泌胆汁参与脂肪消化 [1]。"
                "同时可见腹腔少量低密度积液，常见于多种病因，不能仅凭单张图定性 [3]。"
            ),
            evidence_ids=("1", "3"),
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
                evidence_id="1",
                content=(
                    "肝是人体最大的腺体，位于右季肋区。功能包括糖异生、糖原储存、"
                    "尿素合成与氨解毒、胆汁分泌，以及合成白蛋白、凝血因子等血浆蛋白。"
                ),
                source="SLAKE knowledge_base · 肝脏条目",
            ),
            Evidence(
                evidence_id="2",
                content=(
                    "胆汁由肝细胞分泌、储存在胆囊，经胆总管进入十二指肠，"
                    "参与脂肪的乳化与脂溶性维生素吸收。"
                ),
                source="SLAKE knowledge_base · 胆汁条目",
            ),
            Evidence(
                evidence_id="3",
                content=(
                    "腹腔积液指腹腔内超出正常量的游离液体，可见于肝硬化、心衰、"
                    "结核性腹膜炎、胰腺炎等多种情况，影像表现相似，"
                    "需结合临床与化验鉴别。"
                ),
                source="SLAKE knowledge_base · 腹水条目",
            ),
            Evidence(
                evidence_id="4",
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
                detail="1、3 进入答案上下文，2、4 落榜",
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
            Metric("2", "引用证据"),
            Metric("4", "候选事实"),
            Metric("3.6", "总耗时 s"),
        ],
    },
    "C · 图增强 RAG": {
        "label": "图增强 RAG",
        "desc": "LightRAG mix 模式，图关系 + 向量联合检索",
        "answer": Answer(
            text=(
                "图中主要器官是肝脏，位于右上腹 [1]。肝脏承担代谢与解毒、合成白蛋白"
                "与凝血因子，并分泌胆汁参与脂肪消化 [1][2]。"
                "同时可见腹腔少量低密度积液，常见于多种病因，不能仅凭单张图定性 [3]。"
                "脾区未见明显异常 [4]。"
            ),
            evidence_ids=("1", "2", "3", "4"),
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
                evidence_id="1",
                content=(
                    "肝是人体最大的腺体，位于右季肋区。功能包括糖异生、糖原储存、"
                    "尿素合成与氨解毒、胆汁分泌，以及合成白蛋白、凝血因子等血浆蛋白。"
                ),
                source="SLAKE knowledge_base · 肝脏条目",
            ),
            Evidence(
                evidence_id="2",
                content=(
                    "胆汁由肝细胞分泌、储存在胆囊，经胆总管进入十二指肠，"
                    "参与脂肪的乳化与脂溶性维生素吸收。"
                ),
                source="SLAKE knowledge_base · 胆汁条目",
            ),
            Evidence(
                evidence_id="3",
                content=(
                    "腹腔积液指腹腔内超出正常量的游离液体，可见于肝硬化、心衰、"
                    "结核性腹膜炎、胰腺炎等多种情况，影像表现相似，"
                    "需结合临床与化验鉴别。"
                ),
                source="SLAKE knowledge_base · 腹水条目",
            ),
            Evidence(
                evidence_id="4",
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
            Metric("4", "引用证据"),
            Metric("8", "候选事实"),
            Metric("4.5", "总耗时 s"),
        ],
    },
}

__all__ = ["MODES"]
