"""graph 节点测试(mock LLM 与 researcher_subgraph,测 Command goto 与状态更新逻辑)。

覆盖 supervisor_tools 路由(完成/无调用/派发/overflow/异常/exceeded)、
clarify_with_user 启用路径、assemble_report 拼接与 polish_report 润色。
"""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.nodes import (
    assemble_report,
    clarify_with_user,
    polish_report,
    supervisor_tools,
)
from company_report_kit.graph.state import (
    AgentState,
    ClarifyWithUser,
)
from tests.conftest import FakeModel


class FakeSubgraph:
    """researcher_subgraph 替身:ainvoke 返回预设 dict 或抛异常。"""

    def __init__(self, result: dict | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc

    async def ainvoke(self, _inp, _config=None):  # noqa: ANN001
        if self._exc is not None:
            raise self._exc
        return self._result


def _sup_messages(tool_calls: list[dict], n_ai: int = 1) -> list:
    """构造 supervisor_messages:末位 AIMessage 带 tool_calls,前缀可加空 AIMessage 凑轮数。"""
    msgs: list = [SystemMessage(content="sys"), HumanMessage(content="brief")]
    for _ in range(n_ai - 1):
        msgs.append(AIMessage(content="prev"))
    msgs.append(AIMessage(content="", tool_calls=tool_calls))
    return msgs


def _config() -> RunnableConfig:
    return cast(RunnableConfig, {"configurable": {}})


def _tc(name: str, args: dict | None = None, cid: str = "c1") -> dict:
    return {"name": name, "args": args or {}, "id": cid}


# ──────────────────────────────────────────────────────────────
# supervisor_tools 路由
# ──────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_supervisor_tools_research_complete_exits() -> None:
    """调 ResearchComplete 时直接退出,notes 取已有 ToolMessage。"""
    state = cast(AgentState, {
        "supervisor_messages": _sup_messages([_tc("ResearchComplete")]),
        "research_brief": "b",
    })
    cmd = await supervisor_tools(state, _config())
    assert cmd.goto == "__end__"
    assert cmd.update is not None
    assert cmd.update["research_brief"] == "b"


@pytest.mark.anyio
async def test_supervisor_tools_no_tool_calls_exits() -> None:
    """无 tool_calls(模型给最终答案)时退出。"""
    state = cast(AgentState, {
        "supervisor_messages": _sup_messages([]),
        "research_brief": "b",
    })
    cmd = await supervisor_tools(state, _config())
    assert cmd.goto == "__end__"


@pytest.mark.anyio
async def test_supervisor_tools_dispatches_and_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    """ConductResearch 派发:researcher 结果压成 ToolMessage 回 supervisor 继续循环。"""
    fake = FakeSubgraph(result={"section_text": "压缩笔记", "raw_notes": ["raw1"]})
    monkeypatch.setattr("company_report_kit.graph.nodes.researcher_subgraph", fake)
    state = cast(AgentState, {
        "supervisor_messages": _sup_messages([_tc("ConductResearch", {"research_topic": "电池"})]),
        "research_brief": "b",
    })
    cmd = await supervisor_tools(state, _config())
    assert cmd.goto == "supervisor"
    assert cmd.update is not None
    msgs = cmd.update["supervisor_messages"]
    assert any(isinstance(m, ToolMessage) and m.content == "压缩笔记" for m in msgs)
    assert cmd.update["raw_notes"] == ["raw1"]


@pytest.mark.anyio
async def test_supervisor_tools_overflow_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """5 个 ConductResearch 超 max_concurrent(3),多余 2 个标「超出并行上限」。"""
    fake = FakeSubgraph(result={"section_text": "n", "raw_notes": []})
    monkeypatch.setattr("company_report_kit.graph.nodes.researcher_subgraph", fake)
    calls = [_tc("ConductResearch", {"research_topic": f"t{i}"}, cid=f"c{i}") for i in range(5)]
    state = cast(AgentState, {
        "supervisor_messages": _sup_messages(calls),
        "research_brief": "b",
    })
    cmd = await supervisor_tools(state, _config())
    assert cmd.goto == "supervisor"
    assert cmd.update is not None
    msgs = cmd.update["supervisor_messages"]
    overflow = [m for m in msgs if isinstance(m, ToolMessage) and m.content == "超出并行上限"]
    assert len(overflow) == 2


@pytest.mark.anyio
async def test_supervisor_tools_researcher_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """researcher 子图抛异常时压「研究失败」ToolMessage,不炸整轮。"""
    fake = FakeSubgraph(exc=RuntimeError("子图崩溃"))
    monkeypatch.setattr("company_report_kit.graph.nodes.researcher_subgraph", fake)
    state = cast(AgentState, {
        "supervisor_messages": _sup_messages([_tc("ConductResearch", {"research_topic": "t"})]),
        "research_brief": "b",
    })
    cmd = await supervisor_tools(state, _config())
    assert cmd.goto == "supervisor"
    assert cmd.update is not None
    fail = [m for m in cmd.update["supervisor_messages"] if isinstance(m, ToolMessage)]
    assert len(fail) == 1
    assert "研究失败" in fail[0].content


@pytest.mark.anyio
async def test_supervisor_tools_exceeded_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """research_iterations 超 max_researcher_iterations(6)时,执行完工具后退出。"""
    fake = FakeSubgraph(result={"section_text": "n", "raw_notes": []})
    monkeypatch.setattr("company_report_kit.graph.nodes.researcher_subgraph", fake)
    state = cast(AgentState, {
        # 7 个 AIMessage > 默认 max_researcher_iterations(6)
        "supervisor_messages": _sup_messages([_tc("ConductResearch", {"research_topic": "t"})], n_ai=7),
        "research_brief": "b",
    })
    cmd = await supervisor_tools(state, _config())
    assert cmd.goto == "__end__"


@pytest.mark.anyio
async def test_supervisor_tools_think_only_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    """仅 think_tool 无 ConductResearch:压反思 ToolMessage 后回 supervisor。"""
    fake = FakeSubgraph()  # 不会被调用(无 ConductResearch)
    monkeypatch.setattr("company_report_kit.graph.nodes.researcher_subgraph", fake)
    state = cast(AgentState, {
        "supervisor_messages": _sup_messages([_tc("think_tool", {"reflection": "需补查产能"})]),
        "research_brief": "b",
    })
    cmd = await supervisor_tools(state, _config())
    assert cmd.goto == "supervisor"
    assert cmd.update is not None
    think = [m for m in cmd.update["supervisor_messages"] if isinstance(m, ToolMessage)]
    assert len(think) == 1
    assert "需补查产能" in think[0].content


# ──────────────────────────────────────────────────────────────
# clarify_with_user 启用路径
# ──────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_clarify_skip_when_disabled() -> None:
    """allow_clarification=False 时直接跳 write_brief,不调 LLM。"""
    config = cast(RunnableConfig, {"configurable": {"allow_clarification": False}})
    cmd = await clarify_with_user(
        state=cast(AgentState, {"messages": []}), config=config
    )
    assert cmd.goto == "write_brief"


@pytest.mark.anyio
async def test_clarify_needs_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    """allow_clarification=True 且 LLM 判定需追问时,暂停图并回问。"""
    fake = FakeModel(default=ClarifyWithUser(need_clarification=True, question="范围?", verification="开始"))
    monkeypatch.setattr("company_report_kit.graph.nodes.configurable_model", fake)
    config = cast(RunnableConfig, {"configurable": {"allow_clarification": True}})
    cmd = await clarify_with_user(state=cast(AgentState, {"messages": [HumanMessage(content="写报告")]}), config=config)
    assert cmd.goto == "__end__"
    assert cmd.update is not None
    assert any("范围?" in m.content for m in cmd.update["messages"])


@pytest.mark.anyio
async def test_clarify_proceeds_to_write_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 判定无需追问时,放行 write_brief 并回确认信息。"""
    fake = FakeModel(default=ClarifyWithUser(need_clarification=False, question="q", verification="开始研究"))
    monkeypatch.setattr("company_report_kit.graph.nodes.configurable_model", fake)
    config = cast(RunnableConfig, {"configurable": {"allow_clarification": True}})
    cmd = await clarify_with_user(state=cast(AgentState, {"messages": [HumanMessage(content="写报告")]}), config=config)
    assert cmd.goto == "write_brief"
    assert cmd.update is not None
    assert any("开始研究" in m.content for m in cmd.update["messages"])


# ──────────────────────────────────────────────────────────────
# assemble_report 拼接 / polish_report 润色
# ──────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_assemble_report_concats_sections() -> None:
    """assemble_report 把 sections 按顺序拼接成 # 标题 + ## N. 章节。"""
    state = cast(AgentState, {
        "sections": ["### 融资历史\n内容A[^1]\n[^1]: [来源](https://a.com)"],
        "research_brief": "研究月之暗面",
        "messages": [HumanMessage(content="研究月之暗面")],
    })
    cmd = await assemble_report(state, _config())
    assert cmd.goto == "polish_report"
    assert cmd.update is not None
    report = cmd.update["final_report"]
    # 公司名从用户消息提取(去掉"研究"前缀)
    assert report.startswith("# 月之暗面研究报告")
    assert "## 1. 融资历史" in report
    # notes 被清空(拼接模式下不再用 notes 生成报告)
    assert cmd.update["notes"] == {"type": "override", "value": []}


@pytest.mark.anyio
async def test_assemble_report_empty_sections() -> None:
    """无 sections 时仍产出报告标题(占位)。"""
    state = cast(AgentState, {"sections": [], "research_brief": "研究X公司", "messages": []})
    cmd = await assemble_report(state, _config())
    assert cmd.goto == "polish_report"
    assert cmd.update["final_report"].startswith("# ")


@pytest.mark.anyio
async def test_polish_report_invokes_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """polish_report 调 LLM 润色,输出覆盖 final_report,不新增事实。"""
    fake = FakeModel(responses=[AIMessage(content="润色后的报告")])
    monkeypatch.setattr("company_report_kit.graph.nodes.configurable_model", fake)
    state = cast(AgentState, {"final_report": "# 草稿\n## 1. 章节\n内容", "messages": []})
    cmd = await polish_report(state, _config())
    assert cmd.goto == "__end__"
    assert cmd.update is not None
    assert cmd.update["final_report"] == "润色后的报告"
    assert len(fake.invocations) == 1
    # 草稿进入润色 prompt
    assert "# 草稿" in fake.invocations[0][0].content
