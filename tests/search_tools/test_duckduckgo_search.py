"""DuckDuckGoSearcher 网络测试(真实调 ddgs,无需 key)。

标 network,CI 用 `-m "not network"` 跳过(DDG 不稳)。本地跑需联网。
"""

from __future__ import annotations

import pytest

from company_report_kit.search_tools.base import SearchResponse
from company_report_kit.search_tools.duckduckgo_search import DuckDuckGoSearcher


@pytest.mark.network
def test_ddg_search_returns_sources() -> None:
    """DDG 搜索返回 SearchResponse + 至少一条来源(auto backend)。"""
    r = DuckDuckGoSearcher().search("Beijing weather", max_results=3)
    assert isinstance(r, SearchResponse)
    assert len(r.sources) > 0
    assert r.sources[0].url
    # 原始结果型:DDG 无总结
    assert r.answer is None


@pytest.mark.network
def test_ddg_search_content_filled() -> None:
    """DDG body 映射到 Source.content(摘要片段)。"""
    r = DuckDuckGoSearcher().search("Beijing weather", max_results=2)
    if r.sources:
        assert r.sources[0].content is not None
