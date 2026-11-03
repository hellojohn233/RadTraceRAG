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
