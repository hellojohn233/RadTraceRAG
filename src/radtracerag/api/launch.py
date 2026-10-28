"""API 进程入口。

    uv run radtracerag-api                 # 默认 127.0.0.1:8000
    uv run radtracerag-api --port 8080
    uv run radtracerag-api --host 0.0.0.0 --reload
"""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="radtracerag-api",
        description="RadTraceRAG HTTP API：给 React 前端用。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="改代码自动重启，开发用")
    parser.add_argument(
        "--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )
    args = parser.parse_args()

    uvicorn.run(
        "radtracerag.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
