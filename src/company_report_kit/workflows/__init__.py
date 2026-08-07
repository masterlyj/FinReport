"""固化 workflow 层:各报告类型的组装→审查→修正流水线与共享闭环模块.

对外暴露:
  assemble_sections     — 章节组装(编号层级 + 脚注重编号 + 孤儿/悬空脚注清理)
  run_section_review    — 单章节审查(提取 URL → LLM 结构化 → 问题列表 + url_cache)
  fix_section           — 单章节按问题清单修正
  render_review         — 审查问题渲染成人读 markdown
  run_snapshot          — 非上市公司快照流水线(模板→并行研究→per-section 审查→修正→组装)
"""

from __future__ import annotations

from company_report_kit.workflows.assembly import assemble_sections
from company_report_kit.workflows.review import (
    fix_section,
    render_review,
    run_section_review,
)
from company_report_kit.workflows.snapshot import run_snapshot

__all__ = [
    "assemble_sections",
    "fix_section",
    "render_review",
    "run_section_review",
    "run_snapshot",
]
