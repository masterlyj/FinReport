"""非上市公司研究快照流水线:固定模板并行派发 researcher,各自写章节,per-section
审查+修正,最后单次组装。

不走 clarify → write_brief → supervisor 动态拆分,直接按标准大纲模板
(UNLISTED_TEMPLATE) 并行派发 5 个 researcher 子图(投融资/竞品/团队/业务/财务),
每个 researcher 各自写本维度报告章节(markdown 含脚注引用);然后每章一个审查
agent 并行校验脚注事实(上下文小、聚焦,避免整篇审查的大上下文幻觉),有问题的
章节反馈修正(最多 1 轮);最后代码组装(拼接+重编编号+孤儿/悬空脚注清理)。

审查是 per-section:每章只核本章脚注来源(搜索摘要+正文),不做跨章口径比较——
各章事实各自有据可依即可,跨章重复/口径差异属报告自然现象。
"""

from __future__ import annotations

import asyncio
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.researcher import (
    researcher_subgraph,
    write_section_with_feedback,
)
from company_report_kit.graph.state import (
    ResearcherState,
    ReviewIssue,
    SourceGrouping,
)
from company_report_kit.logging_utils import get_logger
from company_report_kit.outlines import UNLISTED_TEMPLATE
from company_report_kit.workflows.assembly import assemble_sections
from company_report_kit.workflows.review import (
    ReviewStatus,
    _format_issue_for_fix,
    render_review,
    run_section_review,
)

logger = get_logger("workflows.snapshot")


async def run_snapshot(company: str, config: RunnableConfig) -> tuple[str, str]:
    """按标准大纲模板并行派发 researcher,per-section 审查+修正,单次组装.

    流程：并行研究各写章节 → 每章并行审查(一个 agent/章) → 有问题则修正该章
    (最多 1 轮) → 组装(拼接+重编号+孤儿/悬空脚注清理) → 渲染审查结果。

    审查作用在组装前的 section_text 上(脚注为章内本地编号),修正后只组装一次,
    不再有"组装→审查→修正→再组装→重审"的双组装流程。

    Args:
        company: 公司名称。
        config: 运行时配置(含 api_key/thread_id)。

    Returns:
        (报告文本, 审查结果) 二元组。
    """
    topics = UNLISTED_TEMPLATE.topics_for(company)
    label_for = UNLISTED_TEMPLATE.label_for

    logger.info("研究目标: %s", company)
    logger.info("派发 %s 个 researcher 并行研究...", len(topics))

    # 并行派发 researcher 子图(各自搜索→分组→写章节)
    tasks = [
        researcher_subgraph.ainvoke(
            cast("ResearcherState", {
                "researcher_messages": [HumanMessage(content=t)],
                "research_topic": t,
            }),
            config,
        )
        for t in topics
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 收集各 researcher 的章节与原始笔记(原始笔记含搜索摘要,供审查核实正文爬取失败的事实)
    sections: list[str] = []
    raw_notes_list: list[list[str]] = []
    clusters_list: list[list[SourceGrouping]] = []
    for i, r in enumerate(results):
        label = label_for(i)
        if isinstance(r, Exception):
            logger.warning("  [%s] 失败: %s", label, r)
            sections.append(f"## {label}\n\n({label}研究失败: {r})")
            raw_notes_list.append([])
            clusters_list.append([])
        else:
            # gather(return_exceptions=True) 使类型为 dict | BaseException;
            # isinstance 已排除 Exception，cast 收窄到 dict 才能 .get。
            result: dict = cast(dict, r)
            section = result.get("section_text", "")
            logger.info("  [%s] 完成, %s 字符", label, len(section))
            sections.append(section)
            raw_notes_list.append(list(result.get("raw_notes", []) or []))
            clusters_list.append(list(result.get("clusters", []) or []))

    # per-section 审查+修正(并行):每章一个审查 agent,上下文小、聚焦。
    # 审查作用在组装前的 section_text 上,修正后单次组装。
    logger.info("并行审查 %s 个章节...", len(sections))

    # 收集每章审查状态,失败/无脚注时不再静默当"通过"
    review_statuses: list[ReviewStatus] = []

    async def _review_and_fix(i: int) -> tuple[str, list[ReviewIssue]]:
        section = sections[i]
        issues, url_cache, status = await run_section_review(
            section, config, raw_notes=raw_notes_list[i]
        )
        review_statuses.append(status)
        if status is ReviewStatus.FAILED:
            logger.warning("  [%s] 审查失败,标记为审查未执行", label_for(i))
        if status is not ReviewStatus.ISSUES:  # 通过/无脚注/失败均不修正
            return section, []
        # status==ISSUES 时 issues 必非 None(run_section_review 契约),
        # assert 让 mypy 收窄,且运行时违约时快速失败。
        assert issues is not None
        # stamp 真实章号供 render_review 显示维度标签
        for issue in issues:
            issue.section = i + 1
        # 护栏:过滤低置信(可能误报)问题,避免臆测改坏报告;高置信才触发修正
        fixable = [iss for iss in issues if iss.action == "fix" and iss.confidence != "low"]
        if not fixable:
            logger.info("  [%s] 审查 %s 条问题但均为低置信/裁决,不触发修正", label_for(i), len(issues))
            return section, issues
        # 修正走 researcher 的 write_section 核心:基于同一批 clusters+raw_notes,
        # 把审查意见作为修订指令追加,让 LLM 按意见修订而非独立重写。
        # 这就是"审查意见回喂 researcher"——researcher 保留搜索/分组上下文,
        # 只在原章节基础上改被指出的问题。
        review_feedback = "\n".join(
            _format_issue_for_fix(iss, url_cache) for iss in fixable
        )
        try:
            fixed = await write_section_with_feedback(
                topic=topics[i],
                clusters=clusters_list[i],
                raw_notes="\n\n".join(raw_notes_list[i]),
                review_issues=review_feedback,
                config=config,
            )
            logger.info("  [%s] 修正完成, %s 字符", label_for(i), len(fixed))
            return fixed, issues
        except Exception as e:  # noqa: BLE001 - 修正失败保留原章节,不阻断流水线
            logger.warning("  [%s] 修正失败: %s", label_for(i), e)
            return section, issues

    rf_results = await asyncio.gather(*[_review_and_fix(i) for i in range(len(sections))])
    sections = [r[0] for r in rf_results]
    all_issues = [iss for _, isss in rf_results for iss in isss]

    # 单次组装(含孤儿定义剪除 + 悬空引用剥离)
    logger.info("组装报告...")
    report = assemble_sections(company, sections)
    logger.info("组装完成, %s 字符", len(report))

    review_text = render_review(all_issues, label_for)
    # 失败可见性:对审查失败/无脚注的章节追加显式提示,绝不伪装成"校验通过"
    skipped = [
        f"### [{label_for(i)}] 审查{'未执行(失败)' if s is ReviewStatus.FAILED else '跳过(无脚注)'}"
        for i, s in enumerate(review_statuses)
        if s is not ReviewStatus.PASSED and s is not ReviewStatus.ISSUES
    ]
    if skipped:
        review_text = "校验通过\n" + "\n\n".join(skipped) if not all_issues else review_text + "\n\n" + "\n\n".join(skipped)
    logger.info("审查完成: %s", review_text[:200])
    return report, review_text
