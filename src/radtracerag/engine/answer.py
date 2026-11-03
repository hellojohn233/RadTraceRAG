"""基于证据生成回答，并校验引用。

检索与生成在这里是**解耦**的：检索走 ``aquery_data`` 拿结构化证据，
生成由本模块自己调 LLM 完成。这样做的好处是引用格式完全由我们掌控 ——
可以要求模型用 ``[E1]`` 标注依据，然后逐个校验这些编号是否真实存在。
模型编造的编号会被移除并记录在 ``unsupported_markers`` 里。
"""

from __future__ import annotations

from typing import Sequence

from radtracerag.engine.contracts import Answer, Evidence
from radtracerag.engine.llm import build_llm_func
from radtracerag.engine.markers import validate_markers
from radtracerag.engine.settings import EngineSettings

#: 回答阶段的系统提示词
ANSWER_SYSTEM_PROMPT = """你是严谨的知识问答助手。你只能依据下面给出的证据回答问题。

规则：
1. 每条结论后面用 [E1]、[E2] 这样的方括号编号标注它依据的是哪条证据。
2. 只能使用证据中出现过的编号，不要编造不存在的编号。
3. 证据不足以回答时，直接说明哪些信息无法从证据中确认，不要猜测。
4. 回答要简短，先给结论再给依据。"""


def _build_prompt(question: str, evidences: Sequence[Evidence]) -> str:
    """把证据与问题拼成提示词。"""
    blocks = [
        f"[{evidence.evidence_id}] (来源: {evidence.source})\n{evidence.content}"
        for evidence in evidences
    ]
    return "可用证据：\n\n" + "\n\n".join(blocks) + f"\n\n问题：{question}"


async def _complete(
    prompt: str, system_prompt: str | None, settings: EngineSettings
) -> str:
    """调用 LLM 生成原始回答。测试通过 monkeypatch 本函数替换。"""
    llm_func = build_llm_func(settings)
    return await llm_func(prompt=prompt, system_prompt=system_prompt)


async def answer_question(
    question: str, evidences: Sequence[Evidence], settings: EngineSettings
) -> Answer:
    """基于证据回答问题。

    返回的 ``Answer.unsupported_markers`` 记录模型引用了不存在的证据编号 ——
    这些编号已从正文中移除，但留下痕迹以便观测模型的可信度。
    """
    if not evidences:
        return Answer(
            text="没有检索到可用证据，无法回答。",
            evidence_ids=(),
            unsupported_markers=(),
        )

    prompt = _build_prompt(question, evidences)
    raw = await _complete(prompt, ANSWER_SYSTEM_PROMPT, settings)

    valid_ids = [evidence.evidence_id for evidence in evidences]
    clean, valid, unsupported = validate_markers(raw, valid_ids)

    return Answer(
        text=clean.strip(),
        evidence_ids=valid,
        unsupported_markers=unsupported,
    )
