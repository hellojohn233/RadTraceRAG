"""Small CLI for checking the scaffold before data and models are configured."""

import argparse
from importlib.metadata import version
import sys

from radtracerag import __version__
from radtracerag.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radtracerag",
        description="Medical image questions with traceable RAG evidence.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="Check the Python and LightRAG setup")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        from lightrag import LightRAG  # noqa: F401 - verify the SDK can be imported

        settings = Settings.from_environment()
        print(f"RadTraceRAG {__version__}")
        print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
        print(f"LightRAG {version('lightrag-hku')}")
        print(f"Project: {settings.project_root}")
        print(f"Data: {settings.data_dir}")
        print(f"Storage: {settings.storage_dir}")
        return 0
    return 2
