"""网页搜索工具集合,提供统一接口与各具体实现。

对外暴露:
  web_search       — 便捷函数,用默认 DeepSeekSearcher 单例搜索
  WebSearcher      — 统一接口 Protocol
  SearchResponse   — 搜索结果数据类
  Source           — 单条来源数据类
  DeepSeekSearcher — DeepSeek 实现
"""

from __future__ import annotations

from search_tools.base import SearchResponse, Source, WebSearcher
from search_tools.deepseek_search import DeepSeekSearcher

_default_searcher: DeepSeekSearcher | None = None


def web_search(query: str) -> SearchResponse:
    """用默认 DeepSeekSearcher 对 query 发起搜索。"""
    global _default_searcher
    if _default_searcher is None:
        _default_searcher = DeepSeekSearcher()
    return _default_searcher.search(query)


__all__ = [
    "web_search",
    "WebSearcher",
    "SearchResponse",
    "Source",
    "DeepSeekSearcher",
]
