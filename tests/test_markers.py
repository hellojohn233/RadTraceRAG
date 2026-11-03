from radtracerag.engine.markers import extract_markers, strip_markers, validate_markers


def test_extract_markers_returns_unique_in_order():
    assert extract_markers("先看 [E2]，再看 [E1]，还是 [E2]。") == ["E2", "E1"]


def test_extract_markers_empty_when_none():
    assert extract_markers("没有引用的普通文本") == []


def test_strip_markers_removes_all_markers():
    assert strip_markers("答案 [E1] 内容 [E2]").strip() == "答案  内容"


def test_validate_markers_keeps_valid_and_reports_unsupported():
    text = "结论 [E1] 依据 [E9] 还有 [E2]。"
    clean, valid, unsupported = validate_markers(text, ["E1", "E2"])
    assert "[E9]" not in clean
    assert valid == ("E1", "E2")
    assert unsupported == ("E9",)


def test_validate_markers_keeps_valid_markers_in_text():
    clean, valid, _ = validate_markers("结论 [E1] 成立。", ["E1"])
    assert "[E1]" in clean


def test_validate_markers_with_no_valid_ids_removes_everything():
    clean, valid, unsupported = validate_markers("全是 [E3] 和 [E4]", [])
    assert "[E3]" not in clean and "[E4]" not in clean
    assert valid == ()
    assert unsupported == ("E3", "E4")


# --- 列表式 / 大小写 / 带空格的标记写法 ---------------------------------


def test_extract_markers_handles_comma_list():
    """真实模型经常写成 [E1, E9] —— 这种列表里的编号必须能被认出来。"""
    assert extract_markers("结论 [E1, E9] 成立") == ["E1", "E9"]


def test_extract_markers_is_case_insensitive():
    assert extract_markers("见 [e1]") == ["E1"]


def test_extract_markers_allows_space_before_number():
    assert extract_markers("见 [E 9]") == ["E9"]


def test_extract_markers_ignores_marker_outside_brackets():
    """证据正文里出现 'E1 染色体' 这类字样不是引用，不能当编号。"""
    assert extract_markers("E1 染色体与 E2 的关系") == []


def test_validate_markers_removes_unsupported_inside_list():
    """列表里混进的编造编号，必须被剔除并计数 —— 否则它会留在最终答案里。"""
    clean, valid, unsupported = validate_markers("结论 [E1, E9]。", ["E1"])
    assert unsupported == ("E9",)
    assert valid == ("E1",)
    assert "E9" not in clean
    assert "[E1]" in clean


def test_validate_markers_normalizes_case_in_body():
    clean, valid, _ = validate_markers("见 [e1]。", ["E1"])
    assert clean == "见 [E1]。"
    assert valid == ("E1",)


def test_validate_markers_keeps_non_marker_text_in_group():
    """方括号里的非标记文字不能被吃掉。

    模型写成「[依据 E1 的数据]」时，把整组换成「[E1]」会把「依据」和
    「的数据」一起丢掉 —— 那是答案正文里的内容。
    """
    clean, valid, _ = validate_markers("见 [依据 E1 的数据]。", ["E1"])
    assert "依据" in clean
    assert "的数据" in clean
    assert "E1" in clean
    assert valid == ("E1",)


# --- 空括号残骸 ---------------------------------------------------------


def test_validate_markers_removes_empty_brackets():
    """模型自己写出来的空括号要整组删掉。

    ``[]`` / ``[ ]`` 里没有任何信息，留着只会让答案看起来像坏掉了。
    """
    clean, valid, unsupported = validate_markers("结论 [] 成立 [ ]。", ["E1"])
    assert "[]" not in clean
    assert "[ ]" not in clean
    assert "结论" in clean and "成立" in clean
    assert valid == ()
    assert unsupported == ()


def test_validate_markers_removes_group_that_became_empty():
    """摘掉无效编号后只剩空壳的组，整组删掉。"""
    clean, _valid, unsupported = validate_markers("依据 [E9] 还有正文。", ["E1"])
    assert "[]" not in clean
    assert "[ ]" not in clean
    assert "还有正文" in clean
    assert unsupported == ("E9",)


def test_validate_markers_keeps_group_with_real_text():
    """摘掉无效编号后还剩着文字的组要留下 —— 那是答案正文。"""
    clean, _valid, unsupported = validate_markers("依据 [E9 的数据] 结论。", ["E1"])
    assert "的数据" in clean
    assert unsupported == ("E9",)
