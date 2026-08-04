"""网页正文提取,自动去除导航/页脚噪声。

用 ddgs 拿 raw HTML,trafilatura 自动识别正文区域(基于文本密度/DOM 分析,
非正则),输出干净的 markdown 正文。供 researcher 按需取单个 URL 原文,
无需 API key。

对外暴露:
  DuckDuckGoExtractor — 正文提取器
  ddg_extract_url      — LangChain @tool,供 agent 按需取 URL 原文
"""

from __future__ import annotations

from ddgs import DDGS
from langchain_core.tools import tool
import trafilatura


class DuckDuckGoExtractor:
    """基于 trafilatura 的网页正文提取器,自动去除导航/页脚噪声。"""

    def extract(self, url: str) -> str:
        """提取指定 URL 的正文(自动去除导航/页脚/广告)。

        Args:
            url: 目标网页地址。

        Returns:
            清洗后的正文文本(markdown);提取失败返回空串。
        """
        result = DDGS().extract(url, fmt="text")
        html = result.get("content", "")
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        if not html:
            return ""
        # trafilatura 自动识别正文区域,去掉导航/页脚/广告/脚本。
        return trafilatura.extract(html, output_format="markdown", include_links=True) or ""


@tool
def ddg_extract_url(url: str) -> str:
    """提取指定 URL 的网页正文(markdown),自动去除导航页脚噪声。无需 key。"""
    return DuckDuckGoExtractor().extract(url)
