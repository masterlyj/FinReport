"""researcher_subgraph 组件测试:调真编译子图(patch LLM 与工具),验证闭环与输出 schema。

用 FakeModel 注入节点 LLM,真实 think_tool 作被调工具(无网络),
驱动 researcher → researcher_tools → researcher → compress_research → END,
验证 ResearcherOutputState 输出字段(compressed_research / raw_notes)。
"""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.researcher import researcher_subgraph
from company_report_kit.graph.state import ResearcherState
from company_report_kit.utils import think_tool
from tests.conftest import FakeModel


def _config() -> RunnableConfig:
    return cast(RunnableConfig, {"configurable": {}})


@pytest.mark.anyio
async def test_researcher_subgraph_produces_compressed_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整闭环:researcher 调 think_tool → 再决策无调用 → 压缩成 notes。"""
    # 三次 ainvoke:① researcher 带 tool_call(think_tool)→ 执行后回 researcher;
    # ② researcher 无 tool_call → 跳 compress;③ compress 产出笔记。
    fake = FakeModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "think_tool",
            "args": {"reflection": "先搜一下"},
            "id": "c1",
        }]),
        AIMessage(content="无更多工具调用"),
        AIMessage(content="压缩笔记正文"),
    ])
    monkeypatch.setattr("company_report_kit.graph.researcher.configurable_model", fake)
    # think_tool 无网络、原样返回 reflection,作被调工具
    monkeypatch.setattr(
        "company_report_kit.graph.researcher._RESEARCHER_TOOLS", [think_tool]
    )

    out = await researcher_subgraph.ainvoke(
        cast(ResearcherState, {
            "researcher_messages": [HumanMessage(content="电池产业链")],
            "research_topic": "电池产业链",
        }),
        _config(),
    )

    # 输出 schema 仅暴露 compressed_research + raw_notes(内部字段被屏蔽)
    assert out["compressed_research"] == "压缩笔记正文"
    assert isinstance(out["raw_notes"], list)
    # raw_notes 含工具结果(think_tool 返回的 reflection)
    assert any("先搜一下" in n for n in out["raw_notes"])
    # 内部状态不外泄
    assert "researcher_messages" not in out
    assert "tool_call_iterations" not in out
