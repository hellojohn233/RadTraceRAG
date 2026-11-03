from pathlib import Path

from conftest import write_full_env

from radtracerag.engine.settings import EngineSettings


def test_from_env_reads_all_variables(tmp_path):
    env_file = write_full_env(tmp_path)
    s = EngineSettings.from_env(env_file)
    assert s.llm_model == "chat-placeholder"
    assert s.embedding_dim == 1024
    assert s.top_k == 40 and s.chunk_top_k == 20
    assert s.enable_rerank is False
    assert s.llm_max_async == 4 and s.llm_timeout == 240
    assert s.embedding_max_token_size == 8192


def test_working_dir_is_absolute_relative_to_project_root(tmp_path):
    s = EngineSettings.from_env(write_full_env(tmp_path))
    assert s.working_dir.is_absolute()
    assert s.working_dir.name == "rag_storage"


def test_missing_keys_lists_absent_variables(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BASE_URL=https://llm.example/v1\n", encoding="utf-8")
    s = EngineSettings.from_env(env_file)
    missing = s.missing_keys()
    assert "LLM_API_KEY" in missing
    assert "LLM_MODEL" in missing
    assert "EMBEDDING_API_KEY" in missing
    assert "LLM_BASE_URL" not in missing
