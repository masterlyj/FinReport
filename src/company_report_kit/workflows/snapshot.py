"""非上市公司研究快照流水线:固定模板并行派发 researcher,各自写章节,组装+审查.

不走 clarify → write_brief → supervisor 动态拆分,直接按标准大纲模板
(UNLISTED_TEMPLATE) 并行派发 5 个 researcher 子图(投融资/竞品/团队/业务/财务),
每个 researcher 各自写本维度报告章节(markdown 含脚注引用),然后代码组装
(拼接+重编编号),最后主 agent extract URL 正文审查校验,有问题则把对应章节
反馈修正(最多 1 轮)。

与 run_snapshot.py 的区别:OUTLINE/LABELS 已迁入 outlines/unlisted.py 结构化
为 UNLISTED_TEMPLATE;组装/审查/修正逻辑已抽到 workflows/assembly.py 与
workflows/review.py;本文件只保留流水线编排。
"""

from __future__ import annotations

import asyncio
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.researcher import researcher_subgraph
from company_report_kit.graph.state import ResearcherState, ReviewIssue
from company_report_kit.outlines import UNLISTED_TEMPLATE
from company_report_kit.workflows.assembly import assemble_sections
from company_report_kit.workflows.review import (
    fix_section,
    group_fixable_issues,
    render_review,
    run_review,
)


async def run_snapshot(company: str, config: RunnableConfig) -> tuple[str, str]:
    """按标准大纲模板并行派发 researcher,组装+审查+修正闭环.

    流程：并行研究各写章节 → 组装 → 结构化审查 → 有问题则把对应章节
    反馈给 LLM 修正(最多 1 轮)→ 重新组装 → 重审 → 最终报告。

    Args:
        company: 公司名称。
        config: 运行时配置(含 api_key/thread_id)。

    Returns:
        (报告文本, 审查结果) 二元组。
    """
    topics = UNLISTED_TEMPLATE.topics_for(company)
    label_for = UNLISTED_TEMPLATE.label_for

    async def _review(text: str) -> tuple[list[ReviewIssue] | None, str]:
        """跑一次审查;返回(问题列表,渲染文本),无脚注/失败时问题为 None."""
        issues = await run_review(text, config)
        if issues is None:
            return None, "未找到脚注 URL 或审查失败,跳过审查。"
        return issues, render_review(issues, label_for)

    print(f"研究目标: {company}")
    print(f"派发 {len(topics)} 个 researcher 并行研究...")
    print("=" * 60)

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

    # 收集各 researcher 的章节
    sections: list[str] = []
    for i, r in enumerate(results):
        label = label_for(i)
        if isinstance(r, Exception):
            print(f"  [{label}] 失败: {r}")
            sections.append(f"## {label}\n\n({label}研究失败: {r})")
        else:
            section = r.get("section_text", "")
            print(f"  [{label}] 完成, {len(section)} 字符")
            sections.append(section)

    # 组装(拼接+重编编号)
    print("=" * 60)
    print("组装报告...")
    report = assemble_sections(company, sections)
    print(f"组装完成, {len(report)} 字符")

    # 审查校验
    print("审查校验...")
    issues, review = await _review(report)
    print(f"审查完成: {review[:200]}")
    if issues is None:
        return report, review

    # 修正闭环:只对有问题的章节反馈修正(最多 1 轮)
    # 跨章节口径冲突(section=0 或 action=adjudicate)不派发修正,主 agent 裁决
    fix_groups, adjudicated = group_fixable_issues(issues)

    if fix_groups:
        print(f"修正 {len(fix_groups)} 个章节(问题 {sum(len(v) for v in fix_groups.values())} 条,裁决 {len(adjudicated)} 条)...")
        fixed = await asyncio.gather(
            *[
                fix_section(topics[sec - 1], sections[sec - 1], group, config)
                for sec, group in fix_groups.items()
            ],
            return_exceptions=True,
        )
        for sec, new_section in zip(fix_groups.keys(), fixed):
            label = label_for(sec - 1)
            if isinstance(new_section, Exception):
                print(f"  [{label}] 修正失败: {new_section}")
                continue
            print(f"  [{label}] 修正完成, {len(new_section)} 字符")
            sections[sec - 1] = new_section

        report = assemble_sections(company, sections)
        print(f"重新组装完成, {len(report)} 字符")

        # 修正后重审,只作为最终校验,不再二次修正
        print("重审校验...")
        _, review = await _review(report)
        print(f"重审完成: {review[:200]}")
    elif adjudicated:
        print(f"存在 {len(adjudicated)} 条跨章节口径冲突,主 agent 裁决,不触发章节修正。")

    return report, review
