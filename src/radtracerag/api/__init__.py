"""RadTraceRAG 的 HTTP API 层。

这一层只负责传输：校验请求、调 pipeline、返回 JSON。
业务逻辑在 `radtracerag.pipeline`，模型与存储在 `radtracerag.integrations`。
"""
