"""大纲模板层：提供各报告类型的固化标准大纲.

对外暴露:
  ResearchDimension    — 单个研究维度（章节标题 + 调研指令）
  OutlineTemplate      — 一份固化大纲（维度有序列表 + topics_for 约束入口）
  UNLISTED_TEMPLATE    — 非上市公司 5 维度标准大纲实例
"""

from __future__ import annotations

from company_report_kit.outlines.base import OutlineTemplate, ResearchDimension
from company_report_kit.outlines.unlisted import UNLISTED_TEMPLATE

__all__ = [
    "UNLISTED_TEMPLATE",
    "OutlineTemplate",
    "ResearchDimension",
]
