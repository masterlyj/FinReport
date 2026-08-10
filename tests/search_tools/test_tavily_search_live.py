"""Tavily 真实集成测试(标 live,需 TAVILY_API_KEY)。

沉淀手动跑过的真实调用:topic=finance 命中财经来源、include_raw_content="markdown"
拉到全文。CI 用 `-m "not live"` 跳过;本地 `pytest -m live` 跑。
"""

from __future__ import annotations

import pytest

from company_report_kit.search_tools.tavily_search import (
    TavilySearcher,
    tavily_web_search,
)


@pytest.mark.live
def test_tavily_search_returns_finance_sources(
    skip_without_tavily_key: None,
) -> None:
    """真实 search 返回非空来源,markdown 全文至少一条非空。"""
    r = TavilySearcher(include_raw_content="markdown").search(
        "宁德时代 2024 年动力电池装机量", max_results=3
    )
    assert r.answer is None  # include_answer 关闭
    assert len(r.sources) > 0
    assert r.sources[0].url
    # 至少一条拉到 markdown 全文(Yahoo 提不出时个别为空,故只要求至少一条)
    assert any(s.raw_content for s in r.sources)


@pytest.mark.live
def test_tavily_web_search_tool_returns_str(
    skip_without_tavily_key: None,
) -> None:
    """@tool invoke 返回非空 str(走 format_for_agent)。"""
    out = tavily_web_search.invoke({"query": "NVIDIA 2024 annual revenue", "max_results": 2})
    assert isinstance(out, str)
    assert len(out) > 0
    assert "来源:" in out
