"""graph state 的 reducer 与 schema 纯逻辑测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from company_report_kit.graph.state import (
    ClarifyWithUser,
    ResearchQuestion,
    override_reducer,
)


def test_override_reducer_covers() -> None:
    """override 标记整体覆盖,不走默认 append。"""
    # brief / final_report 期望后写覆盖前写,故用 override 标记而非 append。
    assert override_reducer(["old"], {"type": "override", "value": ["new"]}) == ["new"]


def test_override_reducer_appends_by_default() -> None:
    """无 override 标记时走 LangGraph 默认 append 语义。"""
    assert override_reducer(["a"], ["b"]) == ["a", "b"]


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
