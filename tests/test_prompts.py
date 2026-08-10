"""提示词模板的格式契约测试。

验证各模板的占位符完整性与关键结构元素，
确保节点调用 format 时不会因缺字段而报错。
"""

from __future__ import annotations

import string

from company_report_kit import prompts


def _format_vars(template: str) -> list[str]:
    """提取模板中的 format 占位符名称列表。"""
    return [f for _, f, _, _ in string.Formatter().parse(template) if f is not None]


# ──────────────────────────────────────────────────────────────
# 占位符完整性
# ──────────────────────────────────────────────────────────────


def test_clarify_prompt_has_messages_and_date() -> None:
    """clarify_with_user_instructions 需 {messages} 和 {date}。"""
    vars = _format_vars(prompts.clarify_with_user_instructions)
    assert "messages" in vars
    assert "date" in vars


def test_transform_topic_prompt_has_messages_and_date() -> None:
    """transform_messages_into_research_topic_prompt 需 {messages} 和 {date}。"""
    vars = _format_vars(prompts.transform_messages_into_research_topic_prompt)
    assert "messages" in vars
    assert "date" in vars


def test_lead_researcher_prompt_has_iteration_config() -> None:
    """lead_researcher_prompt 需 {date}、{max_researcher_iterations}、{max_concurrent_research_units}。"""
    vars = _format_vars(prompts.lead_researcher_prompt)
    assert "date" in vars
    assert "max_researcher_iterations" in vars
    assert "max_concurrent_research_units" in vars


def test_polish_report_prompt_has_report_var() -> None:
    """polish_report_prompt 需 {report} 变量。"""
    vars = _format_vars(prompts.polish_report_prompt)
    assert vars == ["report"]


def test_research_system_prompt_has_mcp_slot() -> None:
    """research_system_prompt 需 {date} 和 {mcp_prompt}（MCP 工具动态注入位）。"""
    vars = _format_vars(prompts.research_system_prompt)
    assert "date" in vars
    assert "mcp_prompt" in vars


def test_group_sources_prompt_has_sources_var() -> None:
    """group_sources_into_events_prompt 需 {sources} 与 {topic}(研究主题注入分组)。"""
    vars = _format_vars(prompts.group_sources_into_events_prompt)
    assert "sources" in vars
    assert "topic" in vars


# ──────────────────────────────────────────────────────────────
# 模板可格式化（不抛 KeyError）
# ──────────────────────────────────────────────────────────────


def test_clarify_prompt_format_succeeds() -> None:
    """clarify 模板传入合法变量后可成功格式化。"""
    result = prompts.clarify_with_user_instructions.format(messages="测试消息", date="2026年1月1日")
    assert "测试消息" in result
    assert "2026年1月1日" in result


def test_lead_researcher_prompt_format_succeeds() -> None:
    """lead_researcher 模板传入配置参数后可成功格式化。"""
    result = prompts.lead_researcher_prompt.format(
        date="2026年1月1日",
        max_researcher_iterations=6,
        max_concurrent_research_units=3,
    )
    assert "6" in result  # 迭代上限注入
    assert "3" in result  # 并行上限注入


def test_polish_report_prompt_format_succeeds() -> None:
    """polish_report_prompt 注入 report 后可成功格式化。"""
    result = prompts.polish_report_prompt.format(report="# 草稿\n## 1. 章节")
    assert "# 草稿" in result


def test_research_system_prompt_format_with_empty_mcp() -> None:
    """research_system_prompt 的 {mcp_prompt} 可传空串（无 MCP 工具场景）。"""
    result = prompts.research_system_prompt.format(date="2026年1月1日", mcp_prompt="")
    assert "2026年1月1日" in result


def test_group_sources_prompt_format_succeeds() -> None:
    """group_sources 模板传入来源文本与研究主题后可成功格式化。"""
    result = prompts.group_sources_into_events_prompt.format(sources="1. 来源A", topic="研究公司财务")
    assert "1. 来源A" in result
    assert "研究公司财务" in result


# ──────────────────────────────────────────────────────────────
# 关键结构元素
# ──────────────────────────────────────────────────────────────


def test_clarify_prompt_contains_json_schema_hint() -> None:
    """clarify 模板包含 JSON 输出格式说明（need_clarification/question/verification）。"""
    tpl = prompts.clarify_with_user_instructions
    assert "need_clarification" in tpl
    assert "question" in tpl
    assert "verification" in tpl


def test_lead_researcher_prompt_mentions_all_three_tools() -> None:
    """lead_researcher 模板提及三个可用工具名。"""
    tpl = prompts.lead_researcher_prompt
    assert "ConductResearch" in tpl
    assert "ResearchComplete" in tpl
    assert "think_tool" in tpl


def test_polish_report_prompt_forbids_fact_changes() -> None:
    """polish_report_prompt 严禁新增/修改事实,只润色行文。"""
    tpl = prompts.polish_report_prompt
    assert "严禁新增或修改任何事实" in tpl
    assert "纯行文润色" in tpl


def test_group_sources_prompt_has_grouping_rules() -> None:
    """group_sources 模板包含分组判定规则（同一事件 / 转载 / 不同事件 / 孤立来源 / 范围过滤）。"""
    tpl = prompts.group_sources_into_events_prompt
    assert "同一事件" in tpl
    assert "转载" in tpl
    assert "不同事件" in tpl
    assert "孤立来源" in tpl
    assert "范围过滤" in tpl


def test_research_system_prompt_mentions_search_tool() -> None:
    """research_system_prompt 提及 duckduckgo_web_search 作为主要搜索工具。"""
    assert "duckduckgo_web_search" in prompts.research_system_prompt


def test_review_fix_prompt_has_topic_issues_and_sources() -> None:
    """review_fix_prompt 模板含 topic/issues/sources 变量,且可 format 成功。"""
    tpl = prompts.review_fix_prompt
    assert "{topic}" in tpl
    assert "{issues}" in tpl
    assert "{sources}" in tpl
    # 修正指令要求删除找不到出处的细节
    assert "删除" in tpl
    result = tpl.format(
        topic="融资历史",
        issues="- 类型: 无出处\n  报告原文: xxx",
        sources="https://a.com\n原文: 未提及",
    )
    assert "融资历史" in result
    assert "https://a.com" in result


def test_review_fix_prompt_has_no_orphan_placeholder() -> None:
    """review_fix_prompt 只含 topic/issues/sources 三个占位符,无孤儿占位符。

    用 Formatter.parse 而非正则:正确跳过 {{ }} 双花括号转义,
    与 _format_vars(其余占位符检测)同一套机制。
    """
    assert _format_vars(prompts.review_fix_prompt) == ["topic", "issues", "sources"]


def test_write_section_prompt_uses_h3_without_numbering() -> None:
    """write_section 模板:章节片段用 ### 标题,不带数字编号。"""
    tpl = prompts.write_section_prompt
    assert "### 作为章节标题" in tpl
    assert "不要带数字编号" in tpl
    # 精确匹配整句,避免 ### 误伤 ## 子串
    assert "使用 ## 作为章节标题" not in tpl
    # raw_notes 占位符必须存在:write_section 凭它喂原文细节给 LLM
    assert "{raw_notes}" in tpl
    # review_issues 占位符:审查修正时追加审查意见,纯写作时传空串
    assert "{review_issues}" in tpl
    # format 四占位符齐全,不报缺字段
    formatted = tpl.format(topic="t", clusters="c", raw_notes="r", review_issues="")
    assert "t" in formatted and "c" in formatted and "r" in formatted


def test_review_prompt_guides_evidence_as_excerpt() -> None:
    """section_review_prompt 引导"原文实际内容"为原文摘录(保留措辞/数字),非总结转述。"""
    tpl = prompts.section_review_prompt
    assert "摘录对应 URL 原文中的关键语句" in tpl
    assert "保留原措辞、金额、日期、投资方" in tpl
    assert "不要总结转述" in tpl


def test_review_prompt_body_is_verification_basis() -> None:
    """规则 3:页面正文是核实基准,摘要仅作正文提取失败时的兜底——防摘要掩护过期事实。

    回归对抗审查高危#2:摘要与正文矛盾时,不能因摘要含声称就放行(摘要可能是
    过期搜索快照),正文成功时以正文为准。
    """
    tpl = prompts.section_review_prompt
    assert "[页面正文]是核实基准" in tpl
    assert "不要因[搜索摘要]含该事实而放行" in tpl
    assert "[搜索摘要]仅作正文提取失败时的兜底" in tpl


def test_review_fix_prompt_relies_on_full_text() -> None:
    """review_fix_prompt 引导依据<来源原文>完整正文核实修改,而非凭总结猜。"""
    tpl = prompts.review_fix_prompt
    assert "<来源原文>" in tpl
    assert "完整正文" in tpl
    assert "提取失败" in tpl
