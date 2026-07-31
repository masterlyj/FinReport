"""researcher 子图路由逻辑测试(mock LLM 与工具,测 Command goto 与状态更新)。

researcher_tools 节点不触 LLM(LLM 在 researcher 节点产出 AIMessage),
只测三态路由 + 工具安全执行;_execute_tool_safely 单独验异常兜底。
"""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.researcher import (
    _execute_tool_safely,
    researcher_tools,
)
from company_report_kit.graph.state import ResearcherState


class FakeTool:
    """假工具:带 name + async ainvoke,可注入返回值或异常。"""

    def __init__(self, name: str, return_value: str = "结果", exc: Exception | None = None) -> None:
        self.name = name
        self._return = return_value
        self._exc = exc

    async def ainvoke(self, args, config=None):  # noqa: ANN001
        if self._exc is not None:
            raise self._exc
        return self._return


def _tc(name: str = "duckduckgo_web_search", call_id: str = "c1") -> dict:
    """构造一条 tool_call dict(name/args/id)。"""
    return {"name": name, "args": {"query": "q", "max_results": 3}, "id": call_id}


def _config() -> RunnableConfig:
    """空 configurable → Configuration 全默认(max_react_tool_calls=10)。"""
    return cast(RunnableConfig, {"configurable": {}})


@pytest.mark.anyio
async def test_researcher_tools_no_tool_calls_goto_compress() -> None:
    """最近消息无 tool_calls 时直接跳 compress_research。"""
    state = cast(ResearcherState, {"researcher_messages": [AIMessage(content="无工具调用")]})
    cmd = await researcher_tools(state, _config())
    assert cmd.goto == "compress_research"


@pytest.mark.anyio
async def test_researcher_tools_normal_loop_goto_researcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """有 tool_calls 且未超上限时,执行工具后回 researcher 继续循环。"""
    fake = FakeTool("duckduckgo_web_search", return_value="搜索结果文本")
    monkeypatch.setattr(
        "company_report_kit.graph.researcher._RESEARCHER_TOOLS", [fake]
    )
    state = cast(ResearcherState, {
        "researcher_messages": [AIMessage(content="", tool_calls=[_tc()])],
        "tool_call_iterations": 0,  # < max_react_tool_calls(10)
    })
    cmd = await researcher_tools(state, _config())
    assert cmd.goto == "researcher"
    assert cmd.update is not None
    # tool_outputs 作为 researcher_messages 更新
    tool_msg = cmd.update["researcher_messages"][0]
    assert tool_msg.content == "搜索结果文本"
    assert tool_msg.name == "duckduckgo_web_search"


@pytest.mark.anyio
async def test_researcher_tools_exceeded_goto_compress(monkeypatch: pytest.MonkeyPatch) -> None:
    """tool_call_iterations >= max_react_tool_calls 时,执行完本轮工具后跳 compress。"""
    fake = FakeTool("duckduckgo_web_search")
    monkeypatch.setattr(
        "company_report_kit.graph.researcher._RESEARCHER_TOOLS", [fake]
    )
    state = cast(ResearcherState, {
        "researcher_messages": [AIMessage(content="", tool_calls=[_tc()])],
        "tool_call_iterations": 10,  # == max_react_tool_calls → exceeded
    })
    cmd = await researcher_tools(state, _config())
    assert cmd.goto == "compress_research"
    assert cmd.update is not None
    # 仍执行了工具(结果进 update),只是不再回 researcher
    assert cmd.update["researcher_messages"][0].content == "结果"


@pytest.mark.anyio
async def test_execute_tool_safely_returns_error_text() -> None:
    """工具抛异常时不向上传播,返回「工具执行错误」文本(防单工具炸整轮)。"""
    fake = FakeTool("boom", exc=RuntimeError("连接超时"))
    out = await _execute_tool_safely(fake, {}, cast(RunnableConfig, {}))
    assert isinstance(out, str)
    assert "工具执行错误" in out
    assert "连接超时" in out


@pytest.mark.anyio
async def test_execute_tool_safely_returns_result() -> None:
    """工具正常时原样返回结果。"""
    fake = FakeTool("ok", return_value="正常结果")
    assert await _execute_tool_safely(fake, {}, cast(RunnableConfig, {})) == "正常结果"
