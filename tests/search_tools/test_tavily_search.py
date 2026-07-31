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
    """results 映射到 Source(url/title/content);不请求 Tavily 内置 answer,answer 恒为 None。"""
    resp = {
        "results": [{"url": "https://a", "title": "A", "content": "片段"}],
    }
    searcher, mock_client = _make_searcher(monkeypatch, resp, max_results=2)
    r = searcher.search("q")
    assert r.answer is None
    assert r.sources[0].url == "https://a"
    assert r.sources[0].title == "A"
    assert r.sources[0].content == "片段"
    # 验证不再请求 Tavily 内置摘要,且 topic 固定为 finance
    _, kwargs = mock_client.search.call_args
    assert "include_answer" not in kwargs
    assert kwargs["topic"] == "finance"


def test_search_raw_content_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 include_raw_content=False,Tavily 不返回 raw_content,Source.raw_content 为 None。"""
    # 模拟 Tavily 关闭时的真实返回:无 raw_content 字段
    resp = {"results": [{"url": "https://a", "title": "A", "content": "片段"}]}
    searcher, mock_client = _make_searcher(monkeypatch, resp)
    r = searcher.search("q")
    assert r.sources[0].raw_content is None
    # 验证调用时传了 include_raw_content=False(省带宽)
    _, kwargs = mock_client.search.call_args
    assert kwargs["include_raw_content"] is False


def test_search_raw_content_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """传 include_raw_content="markdown" 时,实际向 Tavily 请求 markdown 全文,raw_content 填充(供入库)。"""
    resp = {"results": [{"url": "https://a", "title": "A", "raw_content": "# 全文\n..."}]}
    searcher, mock_client = _make_searcher(monkeypatch, resp, include_raw_content="markdown")
    r = searcher.search("q")
    assert r.sources[0].raw_content == "# 全文\n..."
    _, kwargs = mock_client.search.call_args
    assert kwargs["include_raw_content"] == "markdown"


def test_tool_wires_markdown_and_finance(monkeypatch: pytest.MonkeyPatch) -> None:
    """@tool tavily_web_search 默认拉 markdown 全文 + topic=finance,且不请求 include_answer。"""
    from company_report_kit.search_tools.tavily_search import tavily_web_search

    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"url": "https://a", "title": "A", "content": "片段", "raw_content": "# 标题\n正文"}
        ]
    }
    monkeypatch.setattr(
        "company_report_kit.search_tools.tavily_search.TavilyClient",
        lambda **_: mock_client,
    )
    out = tavily_web_search.invoke({"query": "q", "max_results": 2})
    # 调用参数:topic 固定 finance、拉 markdown 全文、不请求 Tavily 内置 answer
    _, kwargs = mock_client.search.call_args
    assert kwargs["topic"] == "finance"
    assert kwargs["include_raw_content"] == "markdown"
    assert "include_answer" not in kwargs
    assert kwargs["max_results"] == 2
    # 返回文本含 raw 全文(format_for_agent 优先输出全文)
    assert "# 标题" in out and "正文" in out
