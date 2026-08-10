"""网页正文提取：Playwright 渲染 + trafilatura 抽正文，自动通过 JS 反爬挑战。

用 Playwright 无头 Chromium 渲染目标页面，真实执行页面 JS（WAF 挑战如
瑞数 probe.js、动态渲染站点）拿到最终 DOM，再由 trafilatura 自动识别正文
区域（基于文本密度/DOM 分析，非正则）输出干净 markdown 正文。供 researcher
按需取单个 URL 原文、审查环节核对脚注 URL 原文，无需 API key。

为何不用纯 HTTP（primp）+ trafilatura：
  部分站点（如 iyiou.com）用动态 WAF——服务端对无 JS 引擎的客户端回 202
  挑战页（仅探针 script，无正文），primp 等纯 HTTP 客户端拿不到真实页面。
  Playwright 执行完 JS 挑战后能取到最终 DOM。HTTP 状态码（202）不代表
  "失败"，以渲染后的正文为准。

进程级浏览器复用：Playwright 启动/关闭很重，extract_async 复用模块级惰性
初始化的浏览器单例，每次提取新建一个 page 用后即关，同一事件循环内并发
调用共享同一 Chromium 进程；同步 extract 每次新建独立浏览器（playwright
对象绑定事件循环，跨循环复用会报错）。

对外暴露：
  PlaywrightExtractor — 正文提取器（extract 同步 / extract_async 异步）
  extract_url         — LangChain @tool，供 agent 按需取 URL 原文
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import trafilatura
from langchain_core.tools import tool

_DEFAULT_TIMEOUT = 20_000
# 渲染后判定"已拿到正文"的最小字符数：WAF 挑战页通常只有几百字节探针，
# 正文加载后远超此阈值。
_MIN_BODY_CHARS = 800


def _needs_pw_import() -> None:
    """惰性导入 playwright，把失败推迟到首次真正提取时（而非 import 时）。

    项目依赖已声明 playwright，但浏览器二进制可能未安装（uv run playwright
    install chromium）。import 时不做硬校验，首次提取失败会抛明确错误，
    方便用户定位是缺包还是缺浏览器。
    """
    try:
        import playwright  # noqa: F401
    except ImportError as e:  # pragma: no cover - 缺包时的定位提示
        raise RuntimeError(
            "缺少 playwright 包。请先 `uv add playwright` 并 `uv run playwright install chromium`。"
        ) from e


def _ensure_stderr_available() -> None:
    """确保 sys.stderr 可用，避免 playwright driver 启动失败。

    playwright 启动 driver 子进程时读 sys.stderr.fileno() 作其 stderr（见
    _impl/_transport.py:_get_stderr_fileno）。在 stderr 被重定向/关闭的环境
    （后台任务、CI 管道、pythonw），fileno() 返回无效 fd，Windows 下
    CreateProcess 绑定它会报 WinError 5 拒绝访问，driver 起不来。
    这里在首次启动前把 sys.stderr 兜底到 devnull，保证 fileno() 始终有效。
    """
    try:
        if sys.stderr is None or sys.stderr.closed:
            import io

            sys.stderr = io.StringIO()
    except Exception:  # noqa: BLE001, S110 - 兜底失败也不阻断提取，有意静默
        pass


# ---- 进程级浏览器单例 ----------------------------------------------------
# 模块级共享：同一进程内所有 PlaywrightExtractor 调用复用同一 Chromium，
# 避免每次提取都启停浏览器（重）。惰性启动，并发安全。
_init_lock = asyncio.Lock()
_playwright: Any | None = None
_browser: Any | None = None


async def _get_browser() -> Any:
    """惰性获取浏览器单例（异步）。

    asyncio.Lock 保证首次并发调用只有一个初始化，后续直接复用。
    同步 extract 每次 asyncio.run 开新事件循环，但浏览器是绑定 driver 进程
    的独立句柄，跨事件循环可复用。

    Returns:
        Playwright Browser 实例.
    """
    global _browser, _playwright
    if _browser is not None:
        return _browser
    _ensure_stderr_available()
    async with _init_lock:
        if _browser is not None:
            return _browser
        from playwright.async_api import async_playwright

        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=True)
    return _browser


class PlaywrightExtractor:
    """基于 Playwright 渲染 + trafilatura 的网页正文提取器。

    自动执行页面 JS 通过反爬挑战，去除导航/页脚/广告噪声。每次提取新建
    一个 page 用后即关。浏览器生命周期两条路径：
      - extract_async（异步）：复用模块级浏览器单例，同一事件循环内并发安全。
      - extract（同步）：每次新建独立浏览器，避免 playwright 对象跨事件循环
        复用报 'NoneType' has no attribute 'send'。

    Args:
        timeout_ms: 页面加载超时毫秒数.
    """

    def __init__(self, timeout_ms: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout_ms = timeout_ms

    async def _extract_with_browser(self, browser: Any, url: str) -> str:
        """在给定浏览器上渲染 URL 并抽正文（共享逻辑）。

        Args:
            browser: 可用的 Playwright Browser 实例.
            url: 目标网页地址.

        Returns:
            清洗后的正文文本（markdown）；无正文时返回空串。
        """
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
            # 轮询等待正文出现：WAF 挑战执行后正文才加载，不能只看 DOMContentLoaded。
            for _ in range(12):
                body = await page.inner_text("body")
                if len(body) > _MIN_BODY_CHARS:
                    break
                await page.wait_for_timeout(500)
            html = await page.content()
            # trafilatura 自动识别正文区域，去掉导航/页脚/广告/脚本。
            return trafilatura.extract(html, output_format="markdown", include_links=True) or ""
        finally:
            await page.close()

    async def extract_async(self, url: str) -> str:
        """异步提取指定 URL 的正文（供并发调用方使用）。

        复用模块级浏览器单例，同一事件循环内 asyncio.gather 并发安全。

        Args:
            url: 目标网页地址.

        Returns:
            清洗后的正文文本（markdown）；HTTP 可达但页面无正文时返回空串。

        Raises:
            RuntimeError: 网络不可达 / 页面导航失败 / 无正文。
        """
        _needs_pw_import()
        browser = await _get_browser()
        return await self._extract_with_browser(browser, url)

    def extract(self, url: str) -> str:
        """同步提取指定 URL 的正文（供同步调用方 / @tool 使用）。

        本方法用 asyncio.run 驱动独立事件循环，每次新建独立浏览器——playwright
        对象绑定事件循环，跨循环复用单例 browser 会报 'NoneType' has no
        attribute 'send'。同步调用（如 LLM 工具调用在线程池执行）线程内自洽、
        并发安全，代价是每次启停浏览器较慢。

        Args:
            url: 目标网页地址.

        Returns:
            清洗后的正文文本（markdown）.

        Raises:
            RuntimeError: 网络不可达 / 页面导航失败 / 无正文。
        """
        _needs_pw_import()
        _ensure_stderr_available()

        async def _run() -> str:
            from playwright.async_api import async_playwright

            p = await async_playwright().start()
            try:
                browser = await p.chromium.launch(headless=True)
                try:
                    return await self._extract_with_browser(browser, url)
                finally:
                    await browser.close()
            finally:
                await p.stop()

        return asyncio.run(_run())


@tool
def extract_url(url: str) -> str:
    """提取指定 URL 的网页正文（markdown），自动去除导航页脚噪声并执行 JS 反爬挑战。无需 key。

    用无头浏览器渲染页面后提取正文，能通过动态 WAF（如瑞数）的 JS 挑战。
    对搜索结果中重要来源抓取全文时使用。
    """
    return PlaywrightExtractor().extract(url)
