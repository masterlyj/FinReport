"""graph state 的 reducer 与 schema 纯逻辑测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from company_report_kit.graph.state import (
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearchQuestion,
    ResearcherOutputState,
    last_value,
    override_reducer,
)


def test_override_reducer_covers() -> None:
    """override 标记整体覆盖,不走默认 append。"""
    # brief / final_report 期望后写覆盖前写,故用 override 标记而非 append。
    assert override_reducer(["old"], {"type": "override", "value": ["new"]}) == ["new"]


def test_override_reducer_appends_by_default() -> None:
    """无 override 标记时走 LangGraph 默认 append 语义。"""
    assert override_reducer(["a"], ["b"]) == ["a", "b"]


def test_override_reducer_missing_value_returns_dict() -> None:
    """override 标记但无 value 键时,get 回退到整个 new_value。"""
    assert override_reducer(["old"], {"type": "override"}) == {"type": "override"}


def test_override_reducer_value_none() -> None:
    """override value 显式为 None 时整体置空(brief 清空场景)。"""
    assert override_reducer(["old"], {"type": "override", "value": None}) is None


def test_last_value_overwrites() -> None:
    """last_value reducer 直接取新值,丢弃旧值(research_iterations 类计数器)。"""
    assert last_value(0, 5) == 5
    assert last_value("old", "new") == "new"


def test_researcher_output_state_defaults() -> None:
    """ResearcherOutputState 仅 compressed_research 必填,raw_notes 默认空。"""
    o = ResearcherOutputState(compressed_research="摘要")
    assert o.compressed_research == "摘要"
    assert o.raw_notes == []


def test_researcher_output_state_raw_notes_settable() -> None:
    """raw_notes 可显式注入(供 supervisor_tools 汇聚)。"""
    o = ResearcherOutputState(compressed_research="x", raw_notes=["n1", "n2"])
    assert o.raw_notes == ["n1", "n2"]


def test_conductresearch_requires_topic() -> None:
    """ConductResearch 必须带 research_topic(supervisor 派发 schema)。"""
    with pytest.raises(ValidationError):
        ConductResearch()  # type: ignore[call-arg]
    assert ConductResearch(research_topic="电池产业链").research_topic == "电池产业链"


def test_researchcomplete_no_fields() -> None:
    """ResearchComplete 无字段,仅作完成信号。"""
    assert ResearchComplete().model_dump() == {}


def test_clarifywithuser_extra_forbid() -> None:
    """ClarifyWithUser 禁止额外字段,配合 strict tool_choice。"""
    # strict 下额外字段会导致工具调用失败,故 extra=forbid 提前在实例化时拒绝。
    # 用 **dict 传额外字段,触发运行时 ValidationError 而非 pyright 静态报。
    with pytest.raises(ValidationError):
        ClarifyWithUser(
            need_clarification=True, question="q", verification="v", **{"unknown": "x"}
        )


def test_researchquestion_extra_forbid() -> None:
    """ResearchQuestion 同样禁止额外字段。"""
    with pytest.raises(ValidationError):
        ResearchQuestion(research_brief="b", **{"unknown": "x"})
