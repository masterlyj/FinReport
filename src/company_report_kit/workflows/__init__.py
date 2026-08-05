"""固化 workflow 层:各报告类型的组装→审查→修正流水线与共享闭环模块.

对外暴露:
  assemble_sections   — 章节组装(编号层级 + 脚注重编号),见 assembly
  run_review          — 一次完整审查(提取 URL → LLM 结构化 → 问题列表)
  fix_section         — 单章节按问题清单修正
  render_review       — 审查问题渲染成人读 markdown
  group_fixable_issues — 问题按章节分组(可修正 / 主 agent 裁决)
  run_snapshot        — 非上市公司快照流水线(模板→并行研究→组装→审查→修正)
"""

from __future__ import annotations

from company_report_kit.workflows.assembly import assemble_sections
from company_report_kit.workflows.review import (
    fix_section,
    group_fixable_issues,
    render_review,
    run_review,
)
from company_report_kit.workflows.snapshot import run_snapshot

__all__ = [
    "assemble_sections",
    "fix_section",
    "group_fixable_issues",
    "render_review",
    "run_review",
    "run_snapshot",
]
