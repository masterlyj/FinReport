"""网页搜索与正文提取工具集合,提供统一接口、各具体实现与 LangChain @tool。

工具既可程序化调用(SearchResponse / str 结构),也可供 agent 自主调用(@tool 返回文本)。

对外暴露:
  web_search            — 便捷函数,用默认 DeepSeekSearcher 单例搜索
  WebSearcher            — 统一接口 Protocol
  SearchResponse         — 搜索结果数据类
  Source                 — 单条来源数据类
  format_for_agent       — 把结果格式化成给 agent 阅读的文本
  DeepSeekSearcher       — DeepSeek 实现
  deepseek_web_search    — DeepSeek 的 LangChain @tool
  TavilySearcher         — Tavily 实现
  tavily_web_search      — Tavily 的 LangChain @tool
  DuckDuckGoSearcher     — DuckDuckGo 搜索实现(无需 key)
  duckduckgo_web_search  — DuckDuckGo 搜索的 LangChain @tool
  DuckDuckGoExtractor    — DuckDuckGo 正文提取(ddgs extract,无需 key)
  ddg_extract_url        — 提取 URL 原文的 LangChain @tool
"""

from __future__ import annotations

from .base import SearchResponse, Source, WebSearcher, format_for_agent
from .ddg_extract import DuckDuckGoExtractor, ddg_extract_url
from .deepseek_search import (
    DeepSeekSearcher,
    _get_default,
    deepseek_web_search,
)
from .duckduckgo_search import DuckDuckGoSearcher, duckduckgo_web_search
from .tavily_search import TavilySearcher, tavily_web_search


def web_search(query: str) -> SearchResponse:
    """用默认 DeepSeekSearcher 对 query 发起搜索。"""
    return _get_default().search(query)


__all__ = [
    "web_search",
    "WebSearcher",
    "SearchResponse",
    "Source",
    "format_for_agent",
    "DeepSeekSearcher",
    "deepseek_web_search",
    "TavilySearcher",
    "tavily_web_search",
    "DuckDuckGoSearcher",
    "duckduckgo_web_search",
    "DuckDuckGoExtractor",
    "ddg_extract_url",
]
