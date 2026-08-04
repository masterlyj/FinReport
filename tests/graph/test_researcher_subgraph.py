"""researcher_subgraph 组件测试:调真编译子图(patch LLM 与工具),验证闭环与输出 schema。

用 FakeModel 注入节点 LLM,FakeSearch 作被调搜索工具(无网络),
驱动 researcher → researcher_tools → researcher → compress_research → END,
验证 ResearcherOutputState 输出字段(compressed_research / raw_notes)。
"""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.researcher import (
    _clusters_to_notes,
    researcher_subgraph,
)
from company_report_kit.graph.state import (
    ResearcherState,
    SourceGrouping,
)
from tests.conftest import FakeModel


def _config() -> RunnableConfig:
    return cast(RunnableConfig, {"configurable": {}})


@pytest.mark.anyio
async def test_researcher_subgraph_produces_compressed_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整闭环:researcher 搜索(有来源)→ 再决策无调用 → 分组产出结构化 notes。"""
    class FakeSearch:
        """模拟 duckduckgo_web_search:返回 format_for_agent 格式的来源文本。"""
        name = "duckduckgo_web_search"
        async def ainvoke(self, args, config=None):  # noqa: ANN001
            return "来源:\n1. 标题A — https://a.com/x\n   片段A"

    # 三次 ainvoke:① researcher 带 tool_call(搜索)→ 执行后回 researcher;
    # ② researcher 无 tool_call → 跳 compress;③ compress 分组产出 notes。
    fake = FakeModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "duckduckgo_web_search",
            "args": {"query": "电池产业链"},
            "id": "c1",
        }]),
        AIMessage(content="无更多工具调用"),
        # compress_research 用 with_structured_output,注入 SourceGroupingBatch
        _grouping_batch(),
    ])
    monkeypatch.setattr("company_report_kit.graph.researcher.configurable_model", fake)
    monkeypatch.setattr(
        "company_report_kit.graph.researcher._RESEARCHER_TOOLS", [FakeSearch()]
    )

    out = await researcher_subgraph.ainvoke(
        cast(ResearcherState, {
            "researcher_messages": [HumanMessage(content="电池产业链")],
            "research_topic": "电池产业链",
        }),
        _config(),
    )

    # 输出 schema 仅暴露 compressed_research + raw_notes(内部字段被屏蔽)
    # compressed_research 是按簇组织的 notes,含事件/正文引用/佐证来源
    assert "压缩笔记正文" in out["compressed_research"]
    assert "https://a.com/x" in out["compressed_research"]
    assert isinstance(out["raw_notes"], list)
    # raw_notes 含搜索结果文本(来源列表)
    assert any("来源:" in n for n in out["raw_notes"])
    # 内部状态不外泄
    assert "researcher_messages" not in out
    assert "tool_call_iterations" not in out


def _grouping_batch():
    """构造一个 SourceGroupingBatch(单一事件簇)。"""
    from company_report_kit.graph.state import SourceGroupingBatch
    return SourceGroupingBatch(clusters=[
        SourceGrouping(
            event_summary="压缩笔记正文",
            key_facts="关键事实细节",
            primary_url="https://a.com/x",
            supporting_urls=["https://b.com/y"],
        )
    ])


def test_clusters_to_notes_shape() -> None:
    """事件簇转 notes:每个簇含事件/关键事实/正文引用/佐证来源。"""
    clusters = [
        SourceGrouping(
            event_summary="月之暗面C+轮融资7亿美元",
            key_facts="2026-02由阿里、五源等老股东联合领投，估值100亿美元",
            primary_url="https://a.com",
            supporting_urls=["https://b.com", "https://c.com"],
        )
    ]
    notes = _clusters_to_notes(clusters)
    assert "事件: 月之暗面C+轮融资7亿美元" in notes
    assert "关键事实: 2026-02由阿里、五源等老股东联合领投" in notes
    assert "正文引用: https://a.com" in notes
    assert "佐证来源: https://b.com, https://c.com" in notes
