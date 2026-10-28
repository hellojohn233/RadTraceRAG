"""FastAPI 应用：React 前端的入口。

传输层只做四件事：收请求、校验、调 pipeline、吐 JSON。
业务逻辑在 `radtracerag.pipeline`，改业务不动这里；
换传输层（CLI / Streamlit）也不动 pipeline。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from radtracerag import __version__
from radtracerag.api.engine import Engine, build_engine
from radtracerag.api.schemas import (
    AnswerOut,
    EvidenceOut,
    HealthOut,
    MetricOut,
    ModeOut,
    ObservationOut,
    QueryResultOut,
    TraceStepOut,
    mode_payload,
)
from radtracerag.config import Settings
from radtracerag.contracts import ImageQuestion, QueryMode
from radtracerag.pipeline import AnswerResult, answer_question

# 单张影像上限，和界面上的提示一致
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ASSETS = Path(__file__).resolve().parents[3] / "assets"


def to_result(result: AnswerResult) -> QueryResultOut:
    """领域结果 → HTTP 响应模型。字段和界面上的叙事流一一对应。"""
    observation = result.observation
    return QueryResultOut(
        mode=result.mode.value,
        mode_group=result.mode.group,
        mode_label=result.mode.label,
        answer=AnswerOut(
            text=result.answer.text,
            evidence_ids=list(result.answer.evidence_ids),
        ),
        observation=(
            ObservationOut(
                text=observation.text,
                entities=list(observation.entities),
            )
            if observation is not None
            else None
        ),
        evidence=[
            EvidenceOut(
                evidence_id=item.evidence_id,
                content=item.content,
                source=item.source,
            )
            for item in result.evidence
        ],
        trace=[
            TraceStepOut(
                stage=step.stage,
                detail=step.detail,
                elapsed_ms=step.elapsed_ms,
                count=step.count,
            )
            for step in result.trace
        ],
        metrics=[MetricOut(value=m.value, label=m.label) for m in result.metrics],
    )


def _save_upload(upload: UploadFile, target_dir: Path) -> Path:
    target = target_dir / Path(upload.filename or "image.png").name
    with target.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh, length=1024 * 1024)
    return target


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.engine = engine or build_engine(settings)

        # 真实链路：LightRAG 的存储只初始化一次，整个进程复用
        rag = getattr(getattr(app.state.engine, "retriever", None), "rag", None)
        if rag is not None:
            await rag.initialize_storages()
        try:
            yield
        finally:
            if rag is not None:
                await rag.finalize_storages()

    app = FastAPI(
        title="RadTraceRAG API",
        version=__version__,
        description=(
            "面向医学影像问答的可追溯多模态 RAG。"
            "答案、证据、检索轨迹一次返回，前端按问→答→观察→轨迹→证据的顺序渲染。"
        ),
        lifespan=lifespan,
    )

    origins = [
        o.strip()
        for o in os.environ.get("RADTRACERAG_CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthOut, tags=["系统"])
    async def health() -> HealthOut:
        current: Engine = app.state.engine
        return HealthOut(
            status="ok",
            version=__version__,
            engine=current.kind,
            engine_note=current.note,
            workspace=settings.workspace,
            vector_storage=settings.vector_storage,
            graph_storage=settings.graph_storage,
            milvus_uri=settings.milvus_uri,
            milvus_db_name=settings.milvus_db_name,
            modes=mode_payload(),
        )

    @app.get("/api/modes", response_model=list[ModeOut], tags=["系统"])
    async def modes() -> list[ModeOut]:
        """三组对照。A 不检索，B 向量，C 图增强。"""
        return mode_payload()

    @app.post("/api/query", response_model=QueryResultOut, tags=["问答"])
    async def query(
        question: str = Form(..., description="要问的问题"),
        mode: str = Form("mix", description="none / naive / mix，也接受 A / B / C"),
        image: UploadFile | None = File(None, description="影像文件；不传则用合成样例"),
    ) -> QueryResultOut:
        try:
            query_mode = QueryMode.parse(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")

        current: Engine = app.state.engine

        with tempfile.TemporaryDirectory(prefix="radtracerag_") as tmp:
            if image is not None:
                image.file.seek(0, 0)
                payload = image.file.read(MAX_UPLOAD_BYTES + 1)
                if len(payload) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="影像超过 200 MB")
                image_path = Path(tmp) / Path(image.filename or "image.png").name
                image_path.write_bytes(payload)
            else:
                # 没传影像就用合成样例，方便前端先把界面做出来
                image_path = ASSETS / "demo" / "abdominal-ct.png"
                if not image_path.exists():
                    from radtracerag.ui.demo_data import make_demo_image

                    image_path = make_demo_image()

            result = await answer_question(
                ImageQuestion(image_path=image_path, question=question.strip()),
                query_mode,
                observer=current.observer,
                answerer=current.answerer,
                retriever=current.retriever,
            )
        return to_result(result)

    return app


app = create_app()
