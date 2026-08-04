"""researcher_subgraph 组件测试:调真编译子图(patch LLM 与工具),验证闭环与输出 schema。

用 FakeModel 注入节点 LLM,FakeSearch 作被调搜索工具(无网络),
驱动 researcher → researcher_tools → compress_research → write_section → END,
验证 ResearcherOutputState 输出字段(section_text / raw_notes)。
"""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.researcher import (
    _clusters_to_text,
    researcher_subgraph,
)
from company_report_kit.graph.state import (
    ResearcherState,
    SourceGrouping,
)
from tests.conftest import FakeModel


def _config() -> RunnableConfig:
    """空 configurable → Configuration 全默认."""
    return cast(RunnableConfig, {"configurable": {}})


@pytest.mark.anyio
async def test_researcher_subgraph_produces_section_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整闭环:researcher 搜索 → compress 分组 → write_section 产出章节。"""

    class FakeSearch:
        """模拟 duckduckgo_web_search:返回 format_for_agent 格式的来源文本。"""

        name = "duckduckgo_web_search"

        async def ainvoke(self, args, config=None):  # noqa: ANN001
            return "来源:\n1. 标题A — https://a.com/x\n   片段A"

    # 四次 ainvoke:① researcher 带 tool_call(搜索)→ 执行后回 researcher;
    # ② researcher 无 tool_call → 跳 compress;③ compress 分组产出 clusters;
    # ④ write_section 产出章节文本。
    fake = FakeModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "duckduckgo_web_search",
            "args": {"query": "电池产业链"},
            "id": "c1",
        }]),
        AIMessage(content="无更多工具调用"),
        _grouping_batch(),
        AIMessage(content="## 电池产业链概览\n基于研究发现撰写的章节。"),
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

    # 输出 schema 仅暴露 section_text + raw_notes(内部字段被屏蔽)
    assert "电池产业链概览" in out["section_text"]
    assert isinstance(out["raw_notes"], list)
    assert any("来源:" in n for n in out["raw_notes"])
    # 内部状态不外泄
    assert "researcher_messages" not in out
    assert "tool_call_iterations" not in out
    assert "clusters" not in out


def _grouping_batch():
    """构造一个 SourceGroupingBatch(单一事件簇)。"""
    from company_report_kit.graph.state import SourceGroupingBatch

    return SourceGroupingBatch(clusters=[
        SourceGrouping(
            event_summary="电池产业链投资事件",
            key_facts="关键事实细节",
            primary_url="https://a.com/x",
            supporting_urls=["https://b.com/y"],
        )
    ])


def test_clusters_to_text_shape() -> None:
    """事件簇转文本:每个簇含编号/事件/关键事实/正文引用/佐证来源。"""
    clusters = [
        SourceGrouping(
            event_summary="月之暗面C+轮融资7亿美元",
            key_facts="2026-02由阿里、五源等老股东联合领投，估值100亿美元",
            primary_url="https://a.com",
            supporting_urls=["https://b.com", "https://c.com"],
        )
    ]
    text = _clusters_to_text(clusters)
    assert "事件 1: 月之暗面C+轮融资7亿美元" in text
    assert "关键事实: 2026-02由阿里、五源等老股东联合领投" in text
    assert "正文引用: https://a.com" in text
    assert "佐证来源: https://b.com, https://c.com" in text
