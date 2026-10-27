"""启动演示界面。

    uv run radtracerag-ui
    uv run radtracerag-ui --server.port 8600

用 Streamlit 自己的 CLI 起，这样脚本在包上下文里跑，相对导入和资源路径都正常。
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def main() -> None:
    from streamlit.web import cli as stcli

    argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    sys.argv = argv
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
