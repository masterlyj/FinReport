"""DeepSeek 真实集成测试(标 live,需 DEEPSEEK_API_KEY)。

走 Anthropic 兼容端点的服务端 web_search 工具,验证 answer 与 sources 均有产出。
CI 用 `-m "not live"` 跳过。
"""

from __future__ import annotations

import pytest

from company_report_kit.search_tools.deepseek_search import DeepSeekSearcher


@pytest.mark.live
def test_deepseek_search_returns_answer_and_sources(
    skip_without_deepseek_key: None,
) -> None:
    """真实 search 返回非空答案与来源(服务端 web_search 工具)。"""
    r = DeepSeekSearcher().search("宁德时代 2024 年动力电池装机量")
    assert r.answer is not None and len(r.answer) > 0
    assert len(r.sources) > 0
    assert r.sources[0].url
