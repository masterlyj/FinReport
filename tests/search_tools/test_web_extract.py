"""web_extract 网络测试(真实取 URL 正文 markdown,无需 key)。

标 network,CI 跳。本地跑需联网。
"""

from __future__ import annotations

import pytest

from company_report_kit.search_tools import extract_url
from company_report_kit.search_tools.web_extract import PlaywrightExtractor


@pytest.mark.network
def test_extract_returns_markdown() -> None:
    """extract 取 Wikipedia 页,返回 markdown 全文(含标题关键词)。"""
    c = PlaywrightExtractor().extract(
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    )
    assert len(c) > 100
    assert "Python" in c


@pytest.mark.network
def test_extract_tool_ainvoke() -> None:
    """extract_url @tool ainvoke 返回 str 正文(async 工具支持并发)。"""
    import asyncio

    r = asyncio.run(
        extract_url.ainvoke("https://en.wikipedia.org/wiki/Python_(programming_language)")
    )
    assert isinstance(r, str)
    assert len(r) > 100
