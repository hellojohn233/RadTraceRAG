"""引用标记的解析与校验。

模型按约定用 ``[E1]`` 这样的编号标注依据。本模块负责：

1. 找出正文里出现了哪些编号；
2. 区分「真实存在的证据」与「模型凭空编造的编号」；
3. 把编造的编号从正文里**移除**（而不是留着冒充证据）。
"""

from __future__ import annotations

import re
from typing import Iterable

#: 证据编号标记，形如 [E1] / [E12]
MARKER_PATTERN = re.compile(r"\[E(\d+)\]")


def extract_markers(text: str) -> list[str]:
    """返回按首次出现顺序去重的证据编号，如 ``["E2", "E1"]``。"""
    seen: list[str] = []
    for number in MARKER_PATTERN.findall(text):
        marker = f"E{number}"
        if marker not in seen:
            seen.append(marker)
    return seen


def strip_markers(text: str) -> str:
    """移除正文里所有的证据标记，保留其余文字。"""
    return MARKER_PATTERN.sub("", text)


def validate_markers(
    text: str, valid_ids: Iterable[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """校验正文里的证据编号。

    返回 ``(清洗后的正文, 有效编号元组, 无效编号元组)``。

    - 有效编号**保留**在正文中，读者需要看到 ``[E1]`` 对应哪条证据；
    - 无效编号连同其前面的空白一起从正文移除，不给模型编造的引用留下位置。
    """
    allowed = set(valid_ids)
    found = extract_markers(text)
    valid = tuple(marker for marker in found if marker in allowed)
    unsupported = tuple(marker for marker in found if marker not in allowed)

    clean = text
    for marker in unsupported:
        clean = re.sub(rf"\s*\[{re.escape(marker)}\]", "", clean)
    return clean, valid, unsupported
