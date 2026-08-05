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
from company_report_kit.prompts import review_fix_prompt, review_prompt
from company_report_kit.search_tools.ddg_extract import DuckDuckGoExtractor
from company_report_kit.utils import RETRY_KWARGS, configurable_model, get_model_config


def _format_issue_for_fix(issue: ReviewIssue) -> str:
    """把一条审查问题格式化成修正 prompt 的条目."""
    return (
        f"- 类型: {issue.issue_type}\n"
        f"  报告原文: {issue.report_text}\n"
        f"  对应URL: {issue.url or '(无)'}\n"
        f"  原文实际: {issue.evidence}"
    )


async def fix_section(
    topic: str,
    section: str,
    issues: list[ReviewIssue],
    config: RunnableConfig,
) -> str:
    """让 LLM 按审查问题清单修正单章节;修正依据用问题自带 evidence,不重抓 URL.

    Args:
        topic: 该维度的研究主题（作为修正上下文）.
        section: 原章节文本（含脚注）.
        issues: 该章节待修正的问题列表.
        config: 运行时配置.

    Returns:
        修正后的章节文本.
    """
    prompt_content = review_fix_prompt.format(
        topic=topic[:80],
        issues="\n".join(_format_issue_for_fix(issue) for issue in issues),
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


async def run_review(report: str, config: RunnableConfig) -> list[ReviewIssue] | None:
    """对完整报告执行一次审查:提取脚注 URL 正文,LLM 结构化校验,返回问题列表.

    Args:
        report: 组装后的完整报告(含脚注定义).
        config: 运行时配置.

    Returns:
        问题列表;无脚注或结构化输出失败时返回 None(调用方跳过审查).
    """
    # 提取脚注定义里的 URL: [^1]: [标题](URL) 或 [^1]: URL
    footnotes = dict(re.findall(r"\[\^(\d+)\]:\s*(?:\[[^\]]*\]\()?(https?://[^\s)]+)", report))
    if not footnotes:
        return None

    print(f"审查:提取 {len(footnotes)} 个脚注 URL 正文...")
    extractor = DuckDuckGoExtractor()
    urls = list(footnotes.values())
    # extract 是同步方法,用 to_thread 包装让 return_exceptions 能捕获异常
    texts = await asyncio.gather(
        *[asyncio.to_thread(extractor.extract, u) for u in urls],
        return_exceptions=True,
    )

    sources_text = "\n\n".join(
        f"[^{num}] {url}\n{(t if isinstance(t, str) else f'提取失败: {t}')[:800]}"
        for (num, url), t in zip(footnotes.items(), texts)
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
        print(f"审查:结构化输出失败,跳过审查: {e}")
        return None
    return _extract_issues(result)


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
