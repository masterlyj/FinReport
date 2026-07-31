"""TavilySearcher.search 的 mock 测试(不调真实 Tavily API)。

验证 results 映射、include_raw_content 开关与 raw_content 填充。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from company_report_kit.search_tools.tavily_search import TavilySearcher


def _make_searcher(monkeypatch: pytest.MonkeyPatch, resp: dict, **kwargs) -> tuple[TavilySearcher, MagicMock]:
    """构造 TavilySearcher,把 TavilyClient 替换为 mock(避免需真实 key)。"""
    mock_client = MagicMock()
    mock_client.search.return_value = resp
    monkeypatch.setattr(
        "company_report_kit.search_tools.tavily_search.TavilyClient",
        lambda **_: mock_client,
    )
    return TavilySearcher(**kwargs), mock_client


def test_search_maps_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """results 映射到 Source(url/title/content),answer 取 Tavily 内置摘要。"""
    resp = {
        "answer": "摘要",
        "results": [{"url": "https://a", "title": "A", "content": "片段"}],
    }
    searcher, _ = _make_searcher(monkeypatch, resp, max_results=2)
    r = searcher.search("q")
    assert r.answer == "摘要"
    assert r.sources[0].url == "https://a"
    assert r.sources[0].title == "A"
    assert r.sources[0].content == "片段"


def test_search_raw_content_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 include_raw_content=False,Tavily 不返回 raw_content,Source.raw_content 为 None。"""
    # 模拟 Tavily 关闭时的真实返回:无 raw_content 字段
    resp = {"answer": None, "results": [{"url": "https://a", "title": "A", "content": "片段"}]}
    searcher, mock_client = _make_searcher(monkeypatch, resp)
    r = searcher.search("q")
    assert r.sources[0].raw_content is None
    # 验证调用时传了 include_raw_content=False(省带宽)
    _, kwargs = mock_client.search.call_args
    assert kwargs["include_raw_content"] is False


def test_search_raw_content_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """开 include_raw_content=True 时,raw_content 填充(供入库)。"""
    resp = {"answer": None, "results": [{"url": "https://a", "title": "A", "raw_content": "全文"}]}
    searcher, mock_client = _make_searcher(monkeypatch, resp, include_raw_content=True)
    r = searcher.search("q")
    assert r.sources[0].raw_content == "全文"
    _, kwargs = mock_client.search.call_args
    assert kwargs["include_raw_content"] is True
