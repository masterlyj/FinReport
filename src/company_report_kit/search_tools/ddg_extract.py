"""DuckDuckGo 网页正文提取。

走 ddgs 的 extract(url, fmt) 提取网页正文,默认返回 markdown。
用于按需取单个 URL 的原文(供章节撰写 / 入库),无需 API key。
与搜索器不同:输入是 URL 而非 query,输出是正文文本而非 SearchResponse。

对外暴露:
  DuckDuckGoExtractor — 基于 ddgs extract 的正文提取器
  ddg_extract_url      — LangChain @tool,供 agent 按需取 URL 原文
"""

from __future__ import annotations

from ddgs import DDGS
from langchain_core.tools import tool

DEFAULT_FMT = "text_markdown"


class DuckDuckGoExtractor:
    """基于 ddgs extract 的网页正文提取器,无需 API key。

    Args:
        fmt: 输出格式,默认 text_markdown;可选 text_plain / text_rich /
            text(raw HTML)/ content(raw bytes)。
    """

    def __init__(self, fmt: str = DEFAULT_FMT) -> None:
        self._fmt = fmt

    def extract(self, url: str, fmt: str | None = None) -> str:
        """提取指定 URL 的网页正文。

        Args:
            url: 目标网页地址。
            fmt: 覆盖默认输出格式;None 用实例默认值。

        Returns:
            正文文本(markdown);提取失败返回空串。
        """
        result = DDGS().extract(url, fmt=fmt or self._fmt)
        content = result.get("content")
        # content 可能为 None(提取失败)或 bytes(content 格式),统一转 str。
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return content or ""


@tool
def ddg_extract_url(url: str) -> str:
    """提取指定 URL 的网页正文(markdown),用于按需取原文撰写章节。无需 key。"""
    return DuckDuckGoExtractor().extract(url)
