"""网页正文提取,自动去除导航/页脚噪声。

用 primp(与 ddgs 同源的 Rust HTTP 客户端,带浏览器 TLS 指纹)直取原始 HTML,
trafilatura 自动识别正文区域(基于文本密度/DOM 分析,非正则),输出干净的
markdown 正文。供 researcher 按需取单个 URL 原文,无需 API key。

不使用 DDGS().extract():它的 content_map 是字典字面量,会急切求值
text_markdown/text_plain/text_rich(走 primp 内置的 Rust html2text 渲染器),
无论传入 fmt 是什么都触发。遇到畸形 HTML 时 html2text 的 text_renderer 会除零
panic,直接 abort 整个 Python 进程(Rust panic 无法被 try/except 捕获)。
直取 resp.text(原始 HTML)只走 HTTP,绕开 Rust 渲染器;trafilatura 是纯 Python,
异常可被正常捕获。

对外暴露:
  DuckDuckGoExtractor — 正文提取器
  ddg_extract_url      — LangChain @tool,供 agent 按需取 URL 原文
"""

from __future__ import annotations

import primp
from langchain_core.tools import tool
import trafilatura

_DEFAULT_TIMEOUT = 10


class DuckDuckGoExtractor:
    """基于 primp + trafilatura 的网页正文提取器,自动去除导航/页脚噪声。

    Args:
        timeout: HTTP 超时秒数.
    """

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def extract(self, url: str) -> str:
        """提取指定 URL 的正文(自动去除导航/页脚/广告)。

        直取原始 HTML 交给 trafilatura,不走 primp 的 Rust html2text 渲染器
        (后者遇畸形 HTML 会除零 panic 杀进程)。

        Args:
            url: 目标网页地址。

        Returns:
            清洗后的正文文本(markdown);HTTP 200 但无正文时返回空串。

        Raises:
            RuntimeError: HTTP 状态非 200;primp 网络错误原样向上抛(供调用方
                的 return_exceptions 捕获并标记"提取失败")。
        """
        client = primp.Client(
            impersonate="random",
            impersonate_os="random",
            timeout=self._timeout,
        )
        resp = client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} 抓取失败: {url}")
        html = resp.text
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
