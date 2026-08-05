"""报告审查→修正闭环:提取脚注 URL 正文、LLM 结构化审查、按章节分组修正.

通用组件，供各固化 workflow 复用。审查规则与模板解耦——问题按章节号
(section) 归组，跨章节(section=0)或 action=adjudicate 的问题由主 agent
裁决，不派发章节修正。

核心入口:
  run_review       — 一次完整审查(提取 URL→LLM 结构化→渲染人读文本)
  fix_section      — 把某章节按问题清单反馈给 LLM 修正
"""

from __future__ import annotations

import asyncio
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.configuration import Configuration
from company_report_kit.graph.state import ReviewIssue, ReviewResult
from company_report_kit.logging_utils import get_logger
from company_report_kit.prompts import review_fix_prompt, review_prompt
from company_report_kit.search_tools.ddg_extract import DuckDuckGoExtractor
from company_report_kit.utils import RETRY_KWARGS, configurable_model, get_model_config

logger = get_logger("workflows.review")


def _format_issue_for_fix(issue: ReviewIssue, url_cache: dict[str, str]) -> str:
    """把一条审查问题格式化成修正 prompt 的条目.

    携带问题关联 URL 的完整原文(从 url_cache 取),供修正 LLM 对照核实。
    issue.url 可能多个 URL 逗号分隔,逐个查 cache;查不到或为空标注"(无原文)"。
    """
    urls = [u.strip() for u in issue.url.split(",") if u.strip()]
    text_lines = [f"- 类型: {issue.issue_type}\n  报告原文: {issue.report_text}"]
    for u in urls:
        text_lines.append(f"  对应URL: {u}")
        text_lines.append(f"  原文: {url_cache.get(u, '(无原文)')}")
    if not urls:
        text_lines.append(f"  对应URL: (无)\n  原文实际: {issue.evidence}")
    return "\n".join(text_lines)


async def fix_section(
    topic: str,
    section: str,
    issues: list[ReviewIssue],
    url_cache: dict[str, str],
    config: RunnableConfig,
) -> str:
    """让 LLM 按审查问题清单修正单章节;修正依据用 url_cache 里的完整原文.

    Args:
        topic: 该维度的研究主题（作为修正上下文）.
        section: 原章节文本（含脚注）.
        issues: 该章节待修正的问题列表.
        url_cache: {url: 完整正文} 映射,由 run_review 抓取后传入,避免重抓.
        config: 运行时配置.

    Returns:
        修正后的章节文本.
    """
    issues_text = "\n".join(_format_issue_for_fix(issue, url_cache) for issue in issues)
    # 收集本组问题涉及的 URL 原文,供修正 LLM 对照核实
    source_urls = [u.strip() for issue in issues for u in issue.url.split(",") if u.strip()]
    sources_text = "\n\n".join(
        f"{url}\n{url_cache.get(url, '(无原文)')}" for url in dict.fromkeys(source_urls)
    )
    prompt_content = review_fix_prompt.format(
        topic=topic[:80],
        issues=issues_text,
        sources=sources_text,
    )
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    fixer = configurable_model.with_retry(**RETRY_KWARGS).with_config(model_config)
    response = await fixer.ainvoke(
        [HumanMessage(content=prompt_content), HumanMessage(content=section)]
    )
    return str(response.content)


def _extract_issues(result: ReviewResult | None) -> list[ReviewIssue]:
    """从结构化审查结果取出问题列表;None(跳过)时返回空列表."""
    if result is None:
        return []
    return list(result.issues)


async def run_review(
    report: str, config: RunnableConfig
) -> tuple[list[ReviewIssue] | None, dict[str, str]]:
    """对完整报告执行一次审查:提取脚注 URL 正文,LLM 结构化校验.

    Args:
        report: 组装后的完整报告(含脚注定义).
        config: 运行时配置.

    Returns:
        (issues, url_cache):
          issues — 问题列表;无脚注或结构化输出失败时为 None(调用方跳过审查).
          url_cache — {url: 完整正文},供 fix_section 复用,避免重抓。
    """
    # 提取脚注定义里的 URL: [^1]: [标题](URL) 或 [^1]: URL
    footnotes = dict(re.findall(r"\[\^(\d+)\]:\s*(?:\[[^\]]*\]\()?(https?://[^\s)]+)", report))
    if not footnotes:
        return None, {}

    logger.info("审查:提取 %s 个脚注 URL 正文...", len(footnotes))
    extractor = DuckDuckGoExtractor()
    urls = list(footnotes.values())
    # extract 是同步方法,用 to_thread 包装让 return_exceptions 能捕获异常
    texts = await asyncio.gather(
        *[asyncio.to_thread(extractor.extract, u) for u in urls],
        return_exceptions=True,
    )

    # 构建 url_cache:url → 完整正文;extract 失败也进 cache(标注提取失败,供修正判断).
    url_cache: dict[str, str] = {}
    for url, t in zip(urls, texts):
        url_cache[url] = t if isinstance(t, str) else f"提取失败: {t}"

    sources_text = "\n\n".join(
        f"[^{num}] {url}\n{url_cache[url]}"
        for (num, url), _ in zip(footnotes.items(), texts)
    )

    # LLM 对照来源原文做结构化审查;失败按"跳过审查"处理,不阻断流水线
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    reviewer = (
        configurable_model
        .with_structured_output(ReviewResult, strict=False)
        .with_retry(**RETRY_KWARGS)
        .with_config(model_config)
    )
    prompt = review_prompt.format(report=report, sources=sources_text)
    try:
        result = await reviewer.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:  # noqa: BLE001 - 审查失败按跳过处理,不阻断流水线
        logger.warning("审查:结构化输出失败,跳过审查: %s", e)
        return None, url_cache
    return _extract_issues(result), url_cache


def render_review(issues: list[ReviewIssue], label_for) -> str:
    """把结构化审查问题列表渲染成人读的 markdown 文本.

    Args:
        issues: 审查问题列表.
        label_for: 章节下标 → 维度标签的回调(由模板提供,解耦具体维度名).

    Returns:
        人读的审查结果 markdown 文本.
    """
    if not issues:
        return "校验通过"
    lines = [f"共 {len(issues)} 处问题:", ""]
    for i, issue in enumerate(issues, start=1):
        if issue.section == 0:
            scope = "跨章节"
        else:
            scope = label_for(issue.section - 1)
        lines.extend([
            f"### {i}. [{issue.issue_type}] {scope}",
            f"- 报告原文: {issue.report_text}",
            f"- 对应URL: {issue.url or '(无)'}",
            f"- 原文实际: {issue.evidence}",
            "",
        ])
    return "\n".join(lines)


def group_fixable_issues(
    issues: list[ReviewIssue],
) -> tuple[dict[int, list[ReviewIssue]], list[ReviewIssue]]:
    """把审查问题按章节分组:可修正的归 fix_groups,跨章节/裁决的归 adjudicated.

    Args:
        issues: 审查问题列表.

    Returns:
        (fix_groups, adjudicated):
          fix_groups — {section: [issues]},仅 section>=1 且 action=fix 的章节问题
          adjudicated — 跨章节(section=0)或 action=adjudicate 的问题,由主 agent 裁决
    """
    fix_groups: dict[int, list[ReviewIssue]] = {}
    adjudicated: list[ReviewIssue] = []
    for issue in issues:
        if issue.section >= 1 and issue.action != "adjudicate":
            fix_groups.setdefault(issue.section, []).append(issue)
        else:
            adjudicated.append(issue)
    return fix_groups, adjudicated
