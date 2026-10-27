"""LightRAG integration point; model configuration is supplied by callers."""

from typing import Literal

from lightrag import LightRAG, QueryParam


QueryMode = Literal["naive", "mix"]


async def query_context(
    rag: LightRAG, question: str, mode: QueryMode = "mix"
) -> str:
    """Query the same LightRAG index with either planned baseline mode."""
    return await rag.aquery(question, param=QueryParam(mode=mode, only_need_context=True))
