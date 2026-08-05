"""日志与终端输出基础设施:logging + rich 的统一入口.

全库进度/诊断日志走标准 logging(RichHandler 渲染),避免 print 与 logging
混用导致输出顺序不可控;CLI 层面向用户的内容(报告正文/审查结果/文件路径/
交互提示)走共享 rich Console 直接输出。两条通道互不干扰:

  logger.info(...)     — 结构化进度/诊断,带时间与级别,可开关
  console.print(...)   — 面向用户的最终产物,不经日志系统

设计要点:
  - setup_logging() 幂等:重复调用不叠加 handler(防止多 CLI 入口/测试重复加载)。
  - RichHandler 默认 markup=False:报告正文里的脚注标签(如 [^1])、
    章节标题等不会被 rich 当 markdown 标签解析而报错或吞掉。
  - 第三方库(urllib3/langgraph 等)把 verbose 日志静音到 WARNING 以上,
    DEEPSEEK_LOG_LEVEL=DEBUG 可临时打开排查。
  - LogRecord 参数只按 rich 约定的 (name, levelno, message) 位置传入,
    避免 [node]/[label] 前缀被当作 set_x 语法解析。
"""

from __future__ import annotations

import logging
import os
from typing import Final

from rich.console import Console

# 供各节点共用同一个终端输出通道;markup=False 防止 [标签]/脚注被当 markdown 解析.
console: Final[Console] = Console(markup=False)

_LOG_LEVELS: Final[tuple[str, ...]] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")

# 日志级别覆盖:Python 日志名是包名或模块路径(如 company_report_kit.workflows.snapshot).
_LOGGER_NAME: Final[str] = "company_report_kit"


def _resolve_level(name: str, default: str) -> int:
    """把环境变量里的级别名解析成 logging 级别值,非法值回退默认."""
    value = os.getenv(name, default).upper()
    if value not in _LOG_LEVELS:
        return getattr(logging, default)
    return getattr(logging, value)


def setup_logging() -> None:
    """配置包级 logger:RichHandler 输出到终端(幂等,可安全重复调用).

    - 包内 logger 级别取 DEEPSEEK_LOG_LEVEL(默认 INFO);
    - 第三方库 logger 静音到 WARNING(默认),避免 urllib3/langgraph 刷屏。
    """
    root = logging.getLogger()

    # 幂等:已配置过(根 logger 已有 rich handler)则跳过.
    if any(getattr(h, "_crk_rich", False) for h in root.handlers):
        return

    from rich.logging import RichHandler

    rich_handler = RichHandler(
        console=console,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    # 标记该 handler 由本函数创建,避免二次 import 重复叠加.
    rich_handler._crk_rich = True  # type: ignore[attr-defined]
    root.addHandler(rich_handler)

    _setup_package_logger(_resolve_level("DEEPSEEK_LOG_LEVEL", "INFO"))
    _quiet_third_party(_resolve_level("DEEPSEEK_LOG_LEVEL", "INFO"))


def _setup_package_logger(level: int) -> None:
    """设置包内 logger 级别;None 时走继承."""
    logger = logging.getLogger(_LOGGER_NAME)
    if level is not None:
        logger.setLevel(level)


def _quiet_third_party(level: int) -> None:
    """第三方库日志降噪:非 DEBUG 时只透 WARNING 及以上."""
    if level <= logging.DEBUG:
        return
    for name in ("urllib3", "ddgs", "langgraph", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """取包内 logger:日志名统一挂在 company_report_kit 命名空间下.

    Args:
        name: 子模块名(如 ``graph.nodes``),自动补包前缀。

    Returns:
        带统一前缀的 logger 实例。
    """
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
