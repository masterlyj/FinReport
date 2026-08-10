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


def _grouping_batch():
    """构造一个 SourceGroupingBatch(单一事件簇)."""
    from company_report_kit.graph.state import SourceGroupingBatch

    return SourceGroupingBatch(clusters=[
        SourceGrouping(
            event_summary="电池产业链投资事件",
            key_facts="关键事实细节",
            primary_url="https://a.com/x",
            supporting_urls=["https://b.com/y"],
        )
    ])


@pytest.mark.anyio
async def test_researcher_subgraph_produces_section_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整闭环:researcher 搜索 → compress 分组 → write_section 产出章节。"""

    seen_args: list[dict] = []

    class FakeSearch:
        """模拟 duckduckgo_web_search:返回 format_for_agent 格式的来源文本。"""

        name = "duckduckgo_web_search"

        async def ainvoke(self, args, config=None):
            seen_args.append(args)
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

    # 输出 schema 暴露 section_text + raw_notes + clusters(事件骨架,供父图审查修正时重写章节)
    assert "电池产业链概览" in out["section_text"]
    assert isinstance(out["raw_notes"], list)
    assert any("来源:" in n for n in out["raw_notes"])
    # clusters 作为事件分组骨架暴露(父图审查修正重写章节的输入)
    assert isinstance(out["clusters"], list)
    # 真正的内部状态不外泄(消息流/迭代计数不污染父图)
    assert "researcher_messages" not in out
    assert "tool_call_iterations" not in out
    # 锁住节点调用次数:researcher×2 + compress×1 + write_section×1,防节点循环数变化静默错位
    assert len(fake.invocations) == 4
    # 搜索工具收到预期 query
    assert len(seen_args) == 1
    assert seen_args[0]["query"] == "电池产业链"


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


def test_clusters_to_text_multiple_and_no_supporting() -> None:
    """多簇递增编号;无佐证来源时不输出佐证行。"""
    clusters = [
        SourceGrouping(
            event_summary="A轮融资",
            key_facts="",
            primary_url="https://a.com",
            supporting_urls=[],
        ),
        SourceGrouping(
            event_summary="B轮融资",
            key_facts="B轮细节",
            primary_url="https://b.com",
            supporting_urls=["https://c.com"],
        ),
    ]
    text = _clusters_to_text(clusters)
    assert "事件 1: A轮融资" in text
    assert "事件 2: B轮融资" in text
    # 无佐证来源的簇不输出"佐证来源"行
    assert "佐证来源" in text  # 簇2有,总文本含
    assert "事件 1: A轮融资\n" in text
    # 事件1 无佐证来源 → 其块内无"佐证来源"
    block1 = text.split("事件 2")[0]
    assert "佐证来源" not in block1


@pytest.mark.anyio
async def test_researcher_subgraph_empty_clusters_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compress_research 无来源时,write_section 降级为"公开信息有限"。"""
    # 仅 1 次 ainvoke:researcher 无工具调用 → 跳 compress(无 tool 消息,不调 LLM)
    # → write_section 收到空 clusters(不调 LLM),直接降级输出。
    fake = FakeModel(responses=[AIMessage(content="无更多工具调用")])
    monkeypatch.setattr("company_report_kit.graph.researcher.configurable_model", fake)

    out = await researcher_subgraph.ainvoke(
        cast(ResearcherState, {
            "researcher_messages": [HumanMessage(content="无来源主题")],
            "research_topic": "无来源主题",
        }),
        _config(),
    )

    # 空 clusters → write_section 输出"公开信息有限"占位
    assert "公开信息有限" in out["section_text"]
    # 调用次数:researcher 仅 1 次;compress/write_section 均短路不触发 LLM。
    assert len(fake.invocations) == 1


def test_filter_raw_notes_by_urls_keeps_matching_entries() -> None:
    """过滤原始笔记:保留 keep_urls 内的条目,丢弃其余来源原文。"""
    from company_report_kit.graph.researcher import _filter_raw_notes_by_urls

    raw = (
        "来源:\n"
        "1. 营收稿 — https://a.com/rev\n"
        "   2025年收入近5亿元\n"
        "2. 融资稿 — https://a.com/fund\n"
        "   B轮融资7亿美元,腾讯领投"
    )
    out = _filter_raw_notes_by_urls(raw, {"https://a.com/rev"})
    # 保留营收条目
    assert "营收稿" in out
    assert "2025年收入近5亿元" in out
    # 丢弃融资条目(越界来源原文不进入写作上下文)
    assert "融资稿" not in out
    assert "B轮融资7亿美元" not in out
    # 非条目行"来源:"原样保留
    assert "来源:" in out


def test_filter_raw_notes_by_urls_no_match_returns_original() -> None:
    """无条目匹配时返回原文本,不误伤格式异常场景(安全侧)。"""
    from company_report_kit.graph.researcher import _filter_raw_notes_by_urls

    raw = "来源:\n1. 标题 — https://a.com/x\n   内容"
    assert _filter_raw_notes_by_urls(raw, {"https://none.com"}) == raw
    # 空 keep_urls 同样回退原文
    assert _filter_raw_notes_by_urls(raw, set()) == raw


def test_filter_raw_notes_by_urls_normalizes_url_drift() -> None:
    """URL 微漂移(协议/www/尾斜杠)归一化后仍能匹配,不误删合法条目。"""
    from company_report_kit.graph.researcher import _filter_raw_notes_by_urls

    raw = (
        "来源:\n"
        "1. 营收稿 - https://www.a.com/rev\n"
        "   2025年收入近5亿元\n"
        "2. 融资稿 - https://a.com/fund/\n"
        "   B轮融资7亿美元"
    )
    # keep_urls 用裸域名+尾斜杠,条目 URL 带 www/尾斜杠,归一化后应匹配
    out = _filter_raw_notes_by_urls(raw, {"https://a.com/rev/"})
    assert "营收稿" in out
    assert "2025年收入近5亿元" in out
    assert "融资稿" not in out
    assert "B轮融资7亿美元" not in out


@pytest.mark.anyio
async def test_researcher_subgraph_group_prompt_receives_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compress_research 把研究主题喂给分组 LLM,由 LLM 判断来源是否属本维度。

    维度边界的判断主体是分组 LLM(按主题语义),而非代码硬编码词典——
    分组 prompt 必须含研究主题,LLM 据此丢弃与本维度无关的来源。
    """
    from company_report_kit.graph.state import SourceGroupingBatch

    class FakeSearch:
        name = "duckduckgo_web_search"

        async def ainvoke(self, args, config=None):
            return "来源:\n1. 营收稿 — https://a.com/x\n   营收超10亿元"

    fake = FakeModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "duckduckgo_web_search",
            "args": {"query": "营收"},
            "id": "c1",
        }]),
        AIMessage(content="无更多工具调用"),
        SourceGroupingBatch(clusters=[
            SourceGrouping(
                event_summary="公司披露年度营收",
                key_facts="营收超10亿元，亏损收窄",
                primary_url="https://a.com/rev",
            ),
        ]),
        AIMessage(content="### 财务\n营收数据章节。"),
    ])
    monkeypatch.setattr("company_report_kit.graph.researcher.configurable_model", fake)
    monkeypatch.setattr(
        "company_report_kit.graph.researcher._RESEARCHER_TOOLS", [FakeSearch()]
    )

    out = await researcher_subgraph.ainvoke(
        cast(ResearcherState, {
            "researcher_messages": [HumanMessage(content="财务数据")],
            "research_topic": "研究{company}可得的财务数据",
        }),
        _config(),
    )

    # 分组 LLM 的 prompt 含研究主题(LLM 据此判断来源归属)
    group_msg = fake.invocations[2]
    content = group_msg[0].content
    assert "研究{company}可得的财务数据" in content
    # 分组 LLM 返回的簇原样进入写作端(不经过代码层词典过滤)
    write_msg = fake.invocations[3]
    assert "公司披露年度营收" in write_msg[0].content
    assert "营收数据章节" in out["section_text"]
