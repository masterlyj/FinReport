"""graph 节点测试(mock LLM 与 researcher_subgraph,测 Command goto 与状态更新逻辑)。

覆盖 supervisor_tools 路由(完成/无调用/派发/overflow/异常/exceeded)、
clarify_with_user 启用路径、final_report_generation 重试截断与耗尽兜底。
"""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.nodes import (
    clarify_with_user,
    final_report_generation,
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
    fake = FakeSubgraph(result={"compressed_research": "压缩笔记", "raw_notes": ["raw1"]})
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
    fake = FakeSubgraph(result={"compressed_research": "n", "raw_notes": []})
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
    fake = FakeSubgraph(result={"compressed_research": "n", "raw_notes": []})
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
# final_report_generation 重试与截断
# ──────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_final_report_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """一次成功:LLM 返回报告文本,直接落 final_report。"""
    fake = FakeModel(responses=[AIMessage(content="这是最终报告")])
    monkeypatch.setattr("company_report_kit.graph.nodes.configurable_model", fake)
    state = cast(AgentState, {"notes": ["发现1"], "research_brief": "b", "messages": []})
    cmd = await final_report_generation(state, _config())
    assert cmd.goto == "__end__"
    assert cmd.update is not None
    assert cmd.update["final_report"] == "这是最终报告"
    assert len(fake.invocations) == 1


@pytest.mark.anyio
async def test_final_report_token_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """首调 token 超限→截断 findings 重试,二次成功产出报告。"""
    fake = FakeModel(responses=[
        Exception("This request exceeds the context length"),
        AIMessage(content="截断后生成的报告"),
    ])
    monkeypatch.setattr("company_report_kit.graph.nodes.configurable_model", fake)
    state = cast(AgentState, {"notes": ["发现" * 100], "research_brief": "b", "messages": []})
    cmd = await final_report_generation(state, _config())
    assert cmd.goto == "__end__"
    assert cmd.update is not None
    assert cmd.update["final_report"] == "截断后生成的报告"
    assert len(fake.invocations) == 2


@pytest.mark.anyio
async def test_final_report_exhausted_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有重试均 token 超限→返回「生成报告失败」兜底而非抛错。"""
    fake = FakeModel(responses=[Exception("context length exceeded")] * 5)
    monkeypatch.setattr("company_report_kit.graph.nodes.configurable_model", fake)
    state = cast(AgentState, {"notes": ["发现"], "research_brief": "b", "messages": []})
    cmd = await final_report_generation(state, _config())
    assert cmd.goto == "__end__"
    assert cmd.update is not None
    assert "生成报告失败" in cmd.update["final_report"]
