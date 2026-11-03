"""引用标记的解析与校验。

模型按约定用 ``[E1]`` 这样的编号标注依据。本模块负责：

1. 找出正文里出现了哪些编号；
2. 区分「真实存在的证据」与「模型凭空编造的编号」；
3. 把编造的编号从正文里**移除**（而不是留着冒充证据）。

为什么不能只认 ``\\[E(\\d+)\\]``
--------------------------------
真实模型的输出远不止这一种写法：``[E1, E9]``（列表）、``[e1]``（小写）、
``[E 9]``（带空格）都很常见。只认精确形式的话，这些写法里的**编造编号**
会既留在最终答案里、又不进 ``unsupported_markers`` —— 正好是这个功能
要抓的那类失败。

反过来，**方括号之外**的 ``E1`` 不是引用：证据正文里写「E1 染色体」很常见，
把它当编号会凭空造出一条引用。所以识别范围限定在方括号组内。

组内编号的边界用字母数字断言收紧，避免把 ``ABC1`` 里的 ``1``、
``E12abc`` 里的 ``E12`` 误当成编号。
"""

from __future__ import annotations

import re
from typing import Iterable

#: 方括号组（不含嵌套方括号）
GROUP_PATTERN = re.compile(r"\[([^\[\]]*)\]")

#: 组内的证据编号，形如 ``E1`` / ``E 9`` / ``e1``
MARKER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])E\s*(\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _markers_in(inner: str) -> list[str]:
    """取出一段括号内容里的编号，归一成 ``E1`` 形式。"""
    return [f"E{m.group(1)}" for m in MARKER_PATTERN.finditer(inner)]


def extract_markers(text: str) -> list[str]:
    """返回按首次出现顺序去重的证据编号，如 ``["E2", "E1"]``。

    只认方括号里的编号 —— 正文里出现的 ``E1``（比如「E1 染色体」）不是引用。
    """
    seen: list[str] = []
    for group in GROUP_PATTERN.finditer(text):
        for marker in _markers_in(group.group(1)):
            if marker not in seen:
                seen.append(marker)
    return seen


def strip_markers(text: str) -> str:
    """移除正文里所有的证据标记，保留其余文字。

    整组移除（含编号的方括号），避免留下 ``[E1, ]`` 这种残骸。
    """
    def _drop(match: re.Match[str]) -> str:
        inner = match.group(1)
        return "" if MARKER_PATTERN.search(inner) else match.group(0)

    return GROUP_PATTERN.sub(_drop, text)


def validate_markers(
    text: str, valid_ids: Iterable[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """校验正文里的证据编号。

    返回 ``(清洗后的正文, 有效编号元组, 无效编号元组)``。

    - 有效编号**保留**在正文中，读者需要看到 ``[E1]`` 对应哪条证据；
    - 无效编号从正文移除，不给模型编造的引用留下位置；
    - 列表写法（``[E1, E9]``）会被重写成只含有效编号的形式（``[E1]``）；
    - 正文里的编号统一成大写形式（``[e1]`` → ``[E1]``）。
    """
    allowed = set(valid_ids)
    found = extract_markers(text)
    valid = tuple(marker for marker in found if marker in allowed)
    unsupported = tuple(marker for marker in found if marker not in allowed)

    def _rewrite(match: re.Match[str]) -> str:
        inner = match.group(1)
        if not MARKER_PATTERN.search(inner):
            return match.group(0)
        kept = [marker for marker in _markers_in(inner) if marker in allowed]
        if not kept:
            return ""
        return "[" + ", ".join(kept) + "]"

    clean = GROUP_PATTERN.sub(_rewrite, text)
    return clean, valid, unsupported
