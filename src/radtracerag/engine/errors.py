"""引擎层的错误类型。

为什么要单独一个异常类型
------------------------
LightRAG 的很多接口**不抛异常**，而是把失败编码进返回值里 ——
``aquery_data`` 返回 ``{"status": "failure", ...}``，``ainsert`` 把拒收
写进 ``doc_status``。这些失败如果被当成「正常但没结果」，用户看到的就是
「没有检索到可用证据」加一个 0 退出码 —— 服务商挂了也看不出来。

引擎把这些情况显式转成 :class:`EngineError`，让 CLI 能给出可读的信息
和非 0 退出码，而不是把故障粉饰成空结果。
"""


class EngineError(RuntimeError):
    """引擎层可预期的错误。

    与「程序 bug」区分开：这类错误通常带有上游的原始信息，
    直接展示给用户比展示 traceback 更有用。
    """

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        #: 上游的原始信息（如服务商的错误文本）
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message}（{self.detail}）"
        return self.message
