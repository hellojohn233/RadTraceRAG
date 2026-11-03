"""命令行入口。

四个子命令：

``doctor``
    检查配置完整性与服务连通性 —— 真调一次 embedding 核对维度，
    真调一次 LLM 测延迟。缺配置时逐项列出缺哪个变量，而不是抛栈。
``ingest <path>``
    把语料目录写入索引。幂等：重复摄入算「跳过」不算失败。
``query <question>``
    提问，得到带证据编号的回答，并打印检索轨迹。
``eval``
    评测（评测集尚未构建）。
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import time
from importlib.metadata import version
from pathlib import Path

from radtracerag import __version__
from radtracerag.engine.answer import answer_question
from radtracerag.engine.ingest import ingest, load_path_map
from radtracerag.engine.llm import build_llm_func, embed_once
from radtracerag.engine.rag import rag_session
from radtracerag.engine.retrieve import retrieve
from radtracerag.engine.settings import EngineSettings

#: 对外暴露的检索模式
VALID_MODES = ("naive", "local", "global", "mix")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radtracerag",
        description="Traceable multimodal RAG engine CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="检查配置与依赖连通性")

    ingest_parser = subcommands.add_parser("ingest", help="把语料写入索引")
    ingest_parser.add_argument("path", help="语料目录")

    query_parser = subcommands.add_parser("query", help="提问并得到带证据的回答")
    query_parser.add_argument("question", help="要问的问题")
    query_parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="mix",
        help="检索模式（默认 mix）",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="覆盖 top_k",
    )

    subcommands.add_parser("eval", help="评测")
    return parser


def _load_settings() -> tuple[EngineSettings | None, list[str]]:
    """读配置。返回 ``(settings, [])`` 或 ``(None, 错误信息列表)``。"""
    try:
        settings = EngineSettings.from_env()
    except Exception as exc:
        return None, [f"配置读取失败: {type(exc).__name__}: {exc}"]

    missing = settings.missing_keys()
    if missing:
        return None, [f"缺少配置: {name}" for name in missing]
    return settings, []


async def cmd_doctor(args: argparse.Namespace) -> int:
    """检查配置与服务连通性。"""
    print(f"RadTraceRAG {__version__}")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    try:
        print(f"LightRAG {version('lightrag-hku')}")
    except Exception:  # pragma: no cover - 仅在依赖异常时触发
        print("LightRAG 未安装")

    settings, errors = _load_settings()
    if settings is None:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"LLM      {settings.llm_model} @ {settings.llm_base_url}")
    print(f"Embedding {settings.embedding_model} @ {settings.embedding_base_url}")

    # 真实调一次 embedding，核对服务商返回的维度
    try:
        _, actual_dim = await embed_once(settings, "dimension probe")
    except Exception as exc:
        print(f"Embedding 调用失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Embedding 维度: 实际 {actual_dim} / 配置 {settings.embedding_dim}")
    if actual_dim != settings.embedding_dim:
        print(
            f"维度不一致: 服务商返回 {actual_dim} 维，EMBEDDING_DIM 配的是 "
            f"{settings.embedding_dim}。请修正 EMBEDDING_DIM 后重试。",
            file=sys.stderr,
        )
        return 1

    # 真实调一次 LLM，测连通性与延迟
    try:
        llm_func = build_llm_func(settings)
        started = time.perf_counter()
        reply = await llm_func(prompt="ping", system_prompt="只回复 OK")
        elapsed = int((time.perf_counter() - started) * 1000)
    except Exception as exc:
        print(f"LLM 调用失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"LLM 响应: {reply.strip()[:60]} ({elapsed} ms)")

    if settings.working_dir.exists():
        print(f"索引目录: {settings.working_dir}（已存在）")
    else:
        print(f"索引目录: {settings.working_dir}（尚未创建，先跑 ingest）")
    return 0


async def cmd_ingest(args: argparse.Namespace) -> int:
    """把语料目录写入索引。"""
    settings, errors = _load_settings()
    if settings is None:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    root = Path(args.path)
    if not root.exists():
        print(f"语料目录不存在: {root}", file=sys.stderr)
        return 1

    async with rag_session(settings) as rag:
        report = await ingest(rag, root, settings)

    print(f"新增 {len(report.inserted)} · 跳过 {len(report.skipped)} · 失败 {len(report.failed)}")
    for relative in report.inserted:
        print(f"  + {relative}")
    for relative in report.skipped:
        print(f"  = {relative}（已入库）")
    for relative, reason in report.failed:
        print(f"  ! {relative}: {reason}", file=sys.stderr)

    return 1 if report.failed else 0


async def cmd_query(args: argparse.Namespace) -> int:
    """提问，得到带证据的回答。"""
    settings, errors = _load_settings()
    if settings is None:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if args.top_k:
        settings = dataclasses.replace(settings, top_k=args.top_k)

    path_map = load_path_map(settings.working_dir)
    async with rag_session(settings) as rag:
        result = await retrieve(rag, args.question, args.mode, settings, path_map)
        answer = await answer_question(args.question, result.evidences, settings)

    print("答案")
    print(f"  {answer.text}")
    print()
    print("证据")
    if not result.evidences:
        print("  （无）")
    for evidence in result.evidences:
        print(f"  [{evidence.evidence_id}] {evidence.source}  (chunk_id={evidence.chunk_id}, reference_id={evidence.reference_id})")
        print(f"       {evidence.content}")
    print()
    print("检索轨迹")
    for step in result.steps:
        print(f"  {step.stage}  {step.detail}  {step.elapsed_ms} ms · {step.count} 条")
    print()
    if answer.unsupported_markers:
        print(
            f"引用校验  {len(answer.evidence_ids)} 个有效，"
            f"已剔除 {len(answer.unsupported_markers)} 个不存在的编号: "
            f"{', '.join(answer.unsupported_markers)}"
        )
    else:
        print(f"引用校验  {len(answer.evidence_ids)} 个标记全部有效")
    return 0


async def cmd_eval(args: argparse.Namespace) -> int:
    """评测入口。评测集尚未构建。"""
    print("评测集尚未构建：当前仓库只交付引擎层，评估集与标注规则见设计文档。")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    handlers = {
        "doctor": cmd_doctor,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "eval": cmd_eval,
    }
    return await handlers[args.command](args)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。返回进程退出码。"""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse 的用法错误（未知子命令、非法取值）统一转成退出码
        return int(exc.code or 0)

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"执行失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
