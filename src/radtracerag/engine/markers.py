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


#: 只由分隔符与空白组成的内容
#:
#: 这些字符在编号之间只起连接作用 —— 摘掉无效编号之后它们不该留下残骸。
#: ``&`` 和 ``/`` 也在内：``[E1 & E9]`` 里的 ``&``、``[E9/E10]`` 里的 ``/``
#: 都是分隔符而不是正文。
SEPARATOR_ONLY = re.compile(r"^[\s,;、&/]*$")


def _is_residue(text: str) -> bool:
    """判断括号里剩下的东西是不是「残骸」。

    只剩空白或分隔符时，这组括号对读者没有信息量 —— 留着只会让答案看起来
    像坏掉了。

    ⚠️ 判定用「有没有非空白非分隔符的字符」，**不是**「有没有字母数字」：
    emoji（``[😀]``）、全角标点（``[㎡]``）、Markdown 链接（``[](url)``）
    都是**正文**，删掉它们就是在毁掉答案。
    """
    return not text.strip() or SEPARATOR_ONLY.match(text) is not None



def _split_group(inner: str) -> list[tuple[str | None, str]]:
    """把方括号内容拆成 ``[(编号 or None, 文字), ...]``。"""
    pieces: list[tuple[str | None, str]] = []
    pos = 0
    for match in MARKER_PATTERN.finditer(inner):
        if match.start() > pos:
            pieces.append((None, inner[pos : match.start()]))
        pieces.append((f"E{match.group(1)}", match.group(0)))
        pos = match.end()
    if pos < len(inner):
        pieces.append((None, inner[pos:]))
    return pieces


def validate_markers(
    text: str, valid_ids: Iterable[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """校验正文里的证据编号。

    返回 ``(清洗后的正文, 有效编号元组, 无效编号元组)``。

    - 有效编号**保留**在正文中，读者需要看到 ``[E1]`` 对应哪条证据；
    - 无效编号从正文移除，不给模型编造的引用留下位置；
    - 纯编号的组（``[E1, E9]``）重写成只含有效编号的形式（``[E1]``）；
    - 混着别的文字的组（``[依据 E1 的数据]``）**只摘掉无效编号**，
      其余文字原样保留 —— 那是答案正文里的内容，不能被吃掉；
    - 摘完之后只剩空白/分隔符的组**整组删掉**（``[]``、``[ ]``、``[/]``）——
      留着空壳括号只会让答案看起来像坏掉了；
    - emoji、全角标点、Markdown 链接这类**正文**不会被误删。
    """
    allowed = set(valid_ids)
    found = extract_markers(text)
    valid = tuple(marker for marker in found if marker in allowed)
    unsupported = tuple(marker for marker in found if marker not in allowed)

    def _rewrite(match: re.Match[str]) -> str:
        inner = match.group(1)
        markers = _markers_in(inner)
        if not markers:
            if not _is_residue(inner):
                return match.group(0)
            # 后面紧跟 ``(`` → 很可能是 Markdown 链接 ``[](url)``：
            # 空的是链接文字，不是残骸。删掉它会把链接拆成裸 URL。
            if match.string[match.end() : match.end() + 1] == "(":
                return match.group(0)
            # 组里根本没有编号：空括号（``[]`` / ``[ ]``）是残骸，整组删掉；
            # 有内容的组（``[注]`` / ``[1]``）原样保留。
            return ""

        pieces = _split_group(inner)
        marker_only = all(
            marker is not None or SEPARATOR_ONLY.match(text_piece)
            for marker, text_piece in pieces
        )
        canonical = "[" + ", ".join(markers) + "]"

        if all(marker in allowed for marker in markers):
            # 写法不规范的（[e1] / [E 9]）归一成 [E1]，其余原样保留
            if marker_only and canonical != match.group(0):
                return canonical
            return match.group(0)

        if marker_only:
            kept = [
                marker
                for marker, _piece in pieces
                if marker is not None and marker in allowed
            ]
            if not kept:
                # 整组摘空。后面紧跟 ``(`` → 很可能是 Markdown 链接
                # ``[E9](url)``：留着空壳 ``[]`` 才能保住链接结构，
                # 整组删掉会把链接拆成裸 URL。
                if match.string[match.end() : match.end() + 1] == "(":
                    return "[]"
                return ""
            return "[" + ", ".join(kept) + "]"

        # 组里还有别的文字 → 只把无效编号摘掉，其余原样保留
        cleaned = "".join(
            piece_text
            if marker is None or marker in allowed
            else ""
            for marker, piece_text in pieces
        ).strip()
        # 摘完只剩空白/分隔符 → 整组删掉，别在答案里留一个空壳括号
        if _is_residue(cleaned):
            return ""
        return "[" + cleaned + "]"

    clean = GROUP_PATTERN.sub(_rewrite, text)
    return clean, valid, unsupported
