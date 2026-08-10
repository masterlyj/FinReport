"""报告章节审查→修正闭环:提取脚注 URL 正文、LLM 结构化审查、按章节修正.

通用组件，供各固化 workflow 复用。审查按"每章一个 agent"执行——run_section_review
对单个章节文本做一次审查(提取该章脚注 URL 正文+搜索摘要→LLM 结构化→问题列表),
fix_section 按问题清单修正该章。问题不跨章节:每章独立审查,事实以该章脚注来源
(摘要+正文)为准。

核心入口:
  run_section_review — 单章节审查(提取 URL→LLM 结构化→问题列表 + url_cache)
  fix_section        — 把该章节按问题清单反馈给 LLM 修正
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.configuration import Configuration
from company_report_kit.graph.state import ReviewIssue, ReviewResult
from company_report_kit.logging_utils import get_logger
from company_report_kit.prompts import review_fix_prompt, section_review_prompt
from company_report_kit.search_tools.base import normalize_url
from company_report_kit.search_tools.web_extract import PlaywrightExtractor
from company_report_kit.utils import RETRY_KWARGS, configurable_model, get_model_config

logger = get_logger("workflows.review")


class ReviewStatus(Enum):
    """单章节审查结果状态.

    区分"审查通过/有问题/无脚注/审查失败"四态,杜绝把失败静默伪装成通过。
    早期实现用 None 同时表示"无脚注"与"审查失败",调用方无法区分导致
    审查失败被当成"校验通过"直接拼入报告——这是正确性/信任级缺陷。
    """

    PASSED = "passed"          # 审查执行,无问题
    ISSUES = "issues"          # 审查执行,发现问题(需修正)
    NO_FOOTNOTES = "no_footnotes"  # 章节无脚注 URL,无法审查(非失败)
    FAILED = "failed"          # 审查执行但失败(结构化输出/网络异常)


def _format_issue_for_fix(issue: ReviewIssue, url_cache: dict[str, str]) -> str:
    """把一条审查问题格式化成修正 prompt 的条目.

    携带问题关联 URL 的完整原文(从 url_cache 取),供修正 LLM 对照核实。
    issue.url 可能多个 URL 逗号分隔,逐个查 cache;查不到或为空标注"(无原文)"。
    """
    urls = [u.strip() for u in issue.url.split(",") if u.strip()]
    text_lines = [
        f"- 类型: {issue.issue_type} (严重度={issue.severity}, 置信={issue.confidence})\n  报告原文: {issue.report_text}"
    ]
    for u in urls:
        text_lines.append(f"  对应URL: {u}")
        text_lines.append(f"  原文: {url_cache.get(u, '(无原文)')}")
    if not urls:
        text_lines.append(f"  对应URL: (无)\n  原文实际: {issue.evidence}")
    return "\n".join(text_lines)


def _extract_snippets_from_raw_notes(raw_notes: list[str]) -> dict[str, str]:
    """从 researcher 原始笔记中解析"URL → 搜索摘要"映射.

    researcher 写报告时看到的是 format_for_agent 拼好的搜索摘要文本:
      N. 标题 — https://url
         内容片段(可多行)
    这段摘要正是章节事实的直接来源——正文爬取失败时,审查仍需拿它核实。
    正则提取"标题 — url"行后的缩进内容作为该 URL 的摘要;同 URL 多次出现
    时取内容最长的一份(信息量最大)。

    Args:
        raw_notes: 单个 researcher 的原始工具输出列表.

    Returns:
        {url: 搜索摘要片段} 映射;解析不到时为空 dict.
    """
    snippets: dict[str, str] = {}
    # 匹配 "标题 — url" 行;分隔符用 [-—] 兼容 ASCII 连字符与中文破折号,
    # 无标题(直接 url)时也可匹配。URL 归一化后作键,与脚注 URL 的微漂移
    # (协议/www/尾斜杠/查询参数/括号截断)对齐,避免摘要证据静默失配。
    entry_re = re.compile(r"(?:^|\n)\d+\.\s+(?:.*?[-—]\s*)?(https?://[^\s]+)\s*\n")
    for note in raw_notes:
        pos = 0
        for m in entry_re.finditer(note):
            url = normalize_url(m.group(1))
            pos = m.end()
            # 收集该条目标题后的缩进内容(直到下一条目/结束)
            lines: list[str] = []
            for line in note[pos:].split("\n"):
                if re.match(r"^\s*\d+\.\s", line):
                    break
                if line.strip():
                    lines.append(line.strip())
            content = " ".join(lines).strip()
            if content and len(content) > len(snippets.get(url, "")):
                snippets[url] = content
    return snippets


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
        url_cache: {url: 完整正文} 映射,由 run_section_review 抓取后传入,避免重抓.
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
        topic=topic,
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


def _recover_issues_from_raw(raw: object) -> list[ReviewIssue]:
    """从 LLM 原始输出(tool_call args 字符串)容错恢复 ReviewIssue 列表.

    结构化输出解析失败时,DeepSeek 的 tool_call args 常含未转义的裸引号/花括号,
    导致 json.loads 整体失败。此时用状态机逐对象提取,单条坏不影响其他条。
    恢复率 ~97%(极端病理 case 如 evidence 内嵌未闭合引号会丢 1 条)。

    Args:
        raw: with_structured_output(include_raw=True) 的 raw(通常为 AIMessage).
            从中取 invalid_tool_calls / tool_calls 的 args 字符串.

    Returns:
        恢复出的 ReviewIssue 列表;无法恢复时为空列表(调用方按 FAILED 处理).
    """
    import json as _json

    # 从 AIMessage 提取 args 字符串
    args_str = ""
    if hasattr(raw, "invalid_tool_calls"):
        for tc in raw.invalid_tool_calls or []:
            a = getattr(tc, "args", None)
            if isinstance(a, str):
                args_str += a
    if not args_str and hasattr(raw, "tool_calls"):
        for tc in raw.tool_calls or []:
            a = getattr(tc, "args", None)
            if isinstance(a, dict):
                try:
                    args_str += _json.dumps(a, ensure_ascii=False)
                except Exception:  # noqa: BLE001, S110 - 单条 dict 序列化失败跳过
                    pass
    args_str = args_str.strip()
    if not args_str.startswith("{"):
        return []

    # 状态机:从每个 "section" 锚点向前找对象 '{', 字符串状态识别转义, 括号配对找闭合.
    def _find_string_end(text: str, i: int) -> int:
        i += 1
        while i < len(text):
            c = text[i]
            if c == "\\":
                i += 2
                continue
            if c == '"':
                return i
            i += 1
        return -1

    def _extract_object(text: str, brace: int) -> tuple[bool, str, int]:
        depth = 0
        i = brace
        n = len(text)
        while i < n:
            c = text[i]
            if c == '"':
                end = _find_string_end(text, i)
                if end == -1:
                    return False, "", i
                i = end + 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return True, text[brace:i + 1], i
            i += 1
        return False, "", i

    recovered: list[ReviewIssue] = []
    prev_end = 0
    for m in re.finditer(r'"section":\s*\d+', args_str):
        s_start = m.start()
        brace = args_str.rfind("{", prev_end, s_start)
        if brace == -1:
            brace = args_str.rfind("{", 0, s_start)
        ok, frag, end = _extract_object(args_str, brace)
        if not ok:
            continue
        try:
            obj = _json.loads(frag)
        except Exception:  # noqa: BLE001
            try:
                import jiter
                obj = jiter.from_json(frag.encode("utf-8"))
            except Exception:  # noqa: BLE001, S112 - jiter 也失败则跳过该条
                continue
        if isinstance(obj, dict) and "issue_type" in obj and "action" in obj:
            try:
                recovered.append(ReviewIssue.model_validate(obj))
                prev_end = end
            except Exception:  # noqa: BLE001, S112 - 单条非法跳过,不影响其他条
                continue
    return recovered


def _extract_issues(result: ReviewResult | None) -> list[ReviewIssue]:
    """从结构化审查结果取出问题列表;None(跳过)时返回空列表."""
    if result is None:
        return []
    return list(result.issues)


def _strip_markdown_close(url: str) -> str:
    """剥离 markdown 脚注闭合符 `)`,保留 URL 内配对的括号.

    脚注 `[^1]: [标题](https://a.com)` 提取后 URL 可能带 markdown 的尾 `)`;
    但 Wikipedia 消歧义 `https://en.wikipedia.org/wiki/Film_(2024)` 的尾 `)`
    是 URL 的一部分。用平衡计数剥离:末尾连续 `)` 数量减去 URL 内 `(` 数量,
    多余的才是 markdown 闭合——`Film_(2024))`(1 个 `(` 2 个 `)`)剥 1 个;
    `Film_(2024)`(1 对)不剥;`a.com)`(0 个 `(` 1 个 `)`)剥 1 个。

    Args:
        url: 从脚注提取的原始 URL(可能含 markdown 尾 `)`).

    Returns:
        剥离多余 markdown 闭合后的 URL.
    """
    tail = len(url) - len(url.rstrip(")"))
    extra = max(tail - url.count("("), 0)
    if extra:
        return url[:-extra]
    return url


def _format_source_for_review(
    num: str, url: str, snippet: str, body: str
) -> str:
    """把单条来源格式化成审查 prompt 的证据块.

    审查证据 = 标题行 + URL + 搜索摘要(写作时看到的) + 爬取正文(核实用)。
    搜索摘要是 researcher 写报告的直接依据,正文是审查核实依据;两者互补,
    正文爬取失败时摘要仍是可核实来源,避免"无内容→误判无出处"。

    Args:
        num: 脚注编号.
        url: 来源 URL.
        snippet: 搜索摘要 content(可为空串).
        body: 爬取正文(提取失败时为"提取失败: ...").

    Returns:
        格式化后的证据块文本.
    """
    parts = [f"[^{num}] {url}"]
    if snippet:
        parts.append(f"[搜索摘要] {snippet}")
    parts.append(f"[页面正文] {body if body else '(提取失败,无正文)'}")
    return "\n".join(parts)


async def run_section_review(
    section_text: str,
    config: RunnableConfig,
    raw_notes: list[str] | None = None,
) -> tuple[list[ReviewIssue] | None, dict[str, str], ReviewStatus]:
    """对单个章节文本执行审查:提取该章脚注 URL 正文,LLM 结构化校验.

    per-section 审查:每次只看一个章节 + 该章脚注来源,上下文小、聚焦,避免整篇
    审查的大上下文幻觉。不检测跨章节口径冲突。

    Args:
        section_text: 单个章节的文本(含脚注定义,组装前).
        config: 运行时配置.
        raw_notes: 该章节 researcher 的原始笔记(含搜索摘要),用于给审查
            LLM 补充"researcher 写作时看到的摘要 content"——正文爬取失败
            时摘要仍是可核实依据,避免"爬取失败→无内容→误判无出处"。

    Returns:
        (issues, url_cache, status):
          issues — 问题列表;无脚注或审查失败时为 None.
          url_cache — {url: 完整正文},供 fix_section 复用,避免重抓.
          status — ReviewStatus,区分 PASSED/ISSUES/NO_FOOTNOTES/FAILED,
            调用方据此判断"通过/有问题/无脚注/审查失败",不再靠 None 猜.
    """
    # 提取脚注定义里的 URL: [^1]: [标题](URL) 或 [^1]: URL
    # [^\s]+ 允许括号 URL(如 Wikipedia 消歧义 `Film_(2024)`);markdown 闭合 `)`
    # (如 `[标题](https://a.com)`)在提取后剥离——URL 内 `(` 配对的 `)` 保留。
    footnotes = {
        num: _strip_markdown_close(url)
        for num, url in re.findall(
            r"\[\^(\d+)\]:\s*(?:\[[^\]]*\]\()?(https?://[^\s]+)", section_text
        )
    }
    if not footnotes:
        return None, {}, ReviewStatus.NO_FOOTNOTES

    logger.info("审查:提取 %s 个脚注 URL 正文...", len(footnotes))
    extractor = PlaywrightExtractor()
    urls = list(footnotes.values())
    # extract_async 是异步方法,直接并发;Playwright 单例浏览器进程级复用,
    # 多个 URL 并行共享同一 Chromium,失败由 return_exceptions 捕获。
    texts = await asyncio.gather(
        *[extractor.extract_async(u) for u in urls],
        return_exceptions=True,
    )

    # 构建 url_cache:url → 完整正文;extract 失败也进 cache(标注提取失败,供修正判断).
    url_cache: dict[str, str] = {}
    for url, t in zip(urls, texts):
        url_cache[url] = t if isinstance(t, str) else f"提取失败: {t}"

    # 从 researcher 原始笔记取搜索摘要,补充为正文爬取失败时的核实证据。
    # researcher 写报告依据的是搜索摘要 content,审查不能只给"标题+URL+爬取正文"——
    # 爬取失败/正文不含摘要信息时,凭摘要仍可判定章节事实是否有出处。
    snippets = _extract_snippets_from_raw_notes(raw_notes) if raw_notes else {}
    sources_text = "\n\n".join(
        _format_source_for_review(
            num, url, snippets.get(normalize_url(url), ""), url_cache.get(url, "")
        )
        for num, url in footnotes.items()
    )

    # LLM 对照来源原文做结构化审查;失败按"跳过审查"处理,不阻断流水线
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    reviewer = (
        configurable_model
        .with_structured_output(ReviewResult, strict=True, include_raw=True)
        .with_retry(**RETRY_KWARGS)
        .with_config(model_config)
    )
    prompt = section_review_prompt.format(section=section_text, sources=sources_text)
    try:
        raw_out = await reviewer.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:  # noqa: BLE001 - 审查失败按跳过处理,不阻断流水线
        logger.warning("审查:结构化输出失败,标记为 FAILED: %s", e)
        return None, url_cache, ReviewStatus.FAILED
    # include_raw=True 返回 {"raw": AIMessage, "parsed": obj};FakeModel 直接返回 obj
    result = raw_out.get("parsed") if isinstance(raw_out, dict) else raw_out
    # result 为 None:结构化输出解析失败(DeepSeek 底层 tool_call 解析返回 None 而非抛异常).
    # 不能靠 _extract_issues(None)→[] 当"通过",否则解析失败被静默放行。
    if result is None:
        # 文本容错:从原始 tool_call args 抢救 issues(0%→~97%)
        recovered = _recover_issues_from_raw(raw_out.get("raw") if isinstance(raw_out, dict) else None)
        if recovered:
            logger.info("审查:结构化解析失败,容错恢复 %s 条问题", len(recovered))
            return recovered, url_cache, ReviewStatus.ISSUES
        logger.warning("审查:结构化输出解析失败(返回 None),标记为 FAILED")
        return None, url_cache, ReviewStatus.FAILED
    issues = _extract_issues(cast("ReviewResult | None", result))
    status = ReviewStatus.ISSUES if issues else ReviewStatus.PASSED
    return issues, url_cache, status


def render_review(issues: list[ReviewIssue], label_for) -> str:
    """把结构化审查问题列表渲染成人读的 markdown 文本.

    Args:
        issues: 审查问题列表(每条 issue.section 已由调用方 stamp 为真实章号).
        label_for: 章节下标 → 维度标签的回调(由模板提供,解耦具体维度名).

    Returns:
        人读的审查结果 markdown 文本.
    """
    if not issues:
        return "校验通过"
    lines = [f"共 {len(issues)} 处问题:", ""]
    for i, issue in enumerate(issues, start=1):
        scope = label_for(issue.section - 1)
        lines.extend([
            f"### {i}. [{issue.issue_type}] {scope} (严重度={issue.severity}, 置信={issue.confidence})",
            f"- 报告原文: {issue.report_text}",
            f"- 对应URL: {issue.url or '(无)'}",
            f"- 原文实际: {issue.evidence}",
            "",
        ])
    return "\n".join(lines)
