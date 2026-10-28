"""Project paths and storage backends shared by data, retrieval, and evaluation.

存储后端的选择在这里定。B/C 两组对照必须共用同一份向量底座，
否则测出来的「图增强增量」不可比——所以向量库交给 LightRAG 的 storage 层
（默认 Milvus），不要在 LightRAG 之外另建一条平行检索链路。
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    storage_dir: Path
    # LightRAG 存储后端：向量走 Milvus，其余用 LightRAG 自带的本地实现
    workspace: str = "radtracerag"
    vector_storage: str = "MilvusVectorDBStorage"
    kv_storage: str = "JsonKVStorage"
    graph_storage: str = "NetworkXStorage"
    doc_status_storage: str = "JsonDocStatusStorage"
    # Milvus 连接信息由 LightRAG 自己在初始化时从环境变量读，
    # 这里只镜像一份用于 /api/health 展示，不含密码
    milvus_uri: str = ""
    milvus_db_name: str = ""

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
            workspace=os.environ.get("RADTRACERAG_WORKSPACE", "radtracerag"),
            vector_storage=os.environ.get(
                "RADTRACERAG_VECTOR_STORAGE", "MilvusVectorDBStorage"
            ),
            kv_storage=os.environ.get("RADTRACERAG_KV_STORAGE", "JsonKVStorage"),
            graph_storage=os.environ.get(
                "RADTRACERAG_GRAPH_STORAGE", "NetworkXStorage"
            ),
            doc_status_storage=os.environ.get(
                "RADTRACERAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
            ),
            milvus_uri=os.environ.get("MILVUS_URI", ""),
            milvus_db_name=os.environ.get("MILVUS_DB_NAME", ""),
        )
