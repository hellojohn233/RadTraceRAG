"""模型适配器：统一走 OpenAI 兼容接口。

本项目不绑定任何一家模型服务——本地 vLLM / Ollama / LM Studio，或者云上的
兼容服务，配置好 base_url 就能接。视觉模型负责看片，文本模型负责基于证据作答。

两个模型分开配：视觉模型看图，文本模型答题。证据引用由回答模型自己用 [n] 标出，
适配器负责把 [n] 映射回 evidence_id——映射不上就不标，宁可少报引用，
也不要编一个对不上的编号。
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from functools import partial
from pathlib import Path

from openai import AsyncOpenAI

from radtracerag.contracts import Answer, Evidence, ImageObservation, QueryMode

CITE_PATTERN = re.compile(r"\[(\d+)\]")

OBSERVE_PROMPT = """你是影像科医生，正在看一张医学影像。

只回答与问题直接相关的可见信息，不要泛泛描述整张图，不要给诊断结论。
用两三句话说清楚：这是哪个部位、什么模态、看到了什么、与问题相关的结构在哪。
如果看不到与问题有关的内容，就直接说"未见与问题相关的结构"。

最后另起一行，用 "实体：" 开头，列出从影像里识别出的、与问题相关的解剖或影像学实体，
用顿号分隔；没有就写"实体：无"。

问题：{question}"""

ANSWER_PROMPT = """你是影像科医生，根据给定的医学影像和检索到的知识事实回答问题。

规则：
1. 只用证据里写明的内容，证据没写的不许补充
2. 每个来自证据的论断后面用 [编号] 标注来源，编号就是证据前面的序号
3. 证据不足以定论时，直接说不能仅凭现有资料定性
4. 不要写"根据证据""参考资料"这类话，直接给结论
5. 答案控制在三句话以内

证据：
{evidence}

问题：{question}"""


def image_data_uri(image_path: Path) -> str:
    """把影像转成 data URI，OpenAI 兼容接口直接吃。"""
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class ChatClient:
    """OpenAI 兼容的 chat 接口薄封装。"""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        image: Path | None = None,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_uri(image)}}
            )
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content or ""


class VLMObserver:
    """看片：产出与当前问题相关的观察，以及用于检索的实体。"""

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    async def observe(self, image_path: Path, question: str) -> ImageObservation:
        raw = await self.client.complete(
            OBSERVE_PROMPT.format(question=question), image=image_path
        )
        text, entities = _split_entities(raw)
        return ImageObservation(text=text, entities=tuple(entities))


class EvidenceAnswerer:
    """基于证据作答，并把 [n] 引用映射回 evidence_id。"""

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    async def answer(
        self,
        image_path: Path,
        question: str,
        observation: ImageObservation | None,
        evidence: list[Evidence],
        mode: QueryMode,
    ) -> Answer:
        if not evidence:
            # A 组没有证据可引，要求模型明确说明依据不足
            prompt = (
                ANSWER_PROMPT.format(evidence="（本组没有检索到任何知识证据）", question=question)
            )
        else:
            listed = "\n".join(
                f"[{i}] {item.content}（来源：{item.source}）"
                for i, item in enumerate(evidence, start=1)
            )
            prompt = ANSWER_PROMPT.format(evidence=listed, question=question)

        raw = await self.client.complete(prompt, image=image_path)
        cited = _map_citations(raw, evidence)
        return Answer(text=raw.strip(), evidence_ids=cited)


def _split_entities(raw: str) -> tuple[str, list[str]]:
    """把「实体：肝、胆囊」那一行拆出来，正文留给观察描述。"""
    text_lines: list[str] = []
    entities: list[str] = []
    for line in raw.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("实体"):
            _, _, tail = stripped.partition("：")
            entities = [e.strip() for e in tail.split("、") if e.strip() and e.strip() != "无"]
        elif stripped:
            text_lines.append(stripped)
    return " ".join(text_lines).strip(), entities


def _map_citations(raw: str, evidence: list[Evidence]) -> tuple[str, ...]:
    """把答案里的 [n] 映射成 evidence_id；对不上的编号丢掉。"""
    if not evidence:
        return ()
    cited: list[str] = []
    for number in CITE_PATTERN.findall(raw):
        index = int(number) - 1
        if 0 <= index < len(evidence):
            evidence_id = evidence[index].evidence_id
            if evidence_id not in cited:
                cited.append(evidence_id)
    return tuple(cited)


def build_embedding_func():
    """LightRAG 需要的嵌入函数，走 OpenAI 兼容接口。

    维度必须和 Milvus 集合一致——建集合时用的是这个值，写错会在插入时报错。
    """
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    func = partial(
        openai_embed,
        model=os.environ.get("RADTRACERAG_EMBEDDING_MODEL", "bge-m3"),
        base_url=os.environ["RADTRACERAG_EMBEDDING_BASE_URL"],
        api_key=os.environ["RADTRACERAG_EMBEDDING_API_KEY"],
    )
    return EmbeddingFunc(
        embedding_dim=int(os.environ.get("RADTRACERAG_EMBEDDING_DIM", "1024")),
        func=func,
        model_name=os.environ.get("RADTRACERAG_EMBEDDING_MODEL", "bge-m3"),
    )
