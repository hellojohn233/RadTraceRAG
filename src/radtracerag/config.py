"""Project paths shared by data, retrieval, and evaluation modules."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    storage_dir: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(
            os.environ.get("RADTRACERAG_DATA_DIR", project_root / "data")
        ).expanduser()
        storage_dir = Path(
            os.environ.get("RADTRACERAG_STORAGE_DIR", project_root / "rag_storage")
        ).expanduser()
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            storage_dir=storage_dir,
        )
