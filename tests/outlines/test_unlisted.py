"""outlines 模板层测试:非上市公司标准大纲的结构与约束契约.

验证:
  - UNLISTED_TEMPLATE 的维度数量/顺序/标签
  - topics_for() 生成的主题圈定在模板内、绑定公司名
  - label_for() 越界回退
  - ResearchDimension prompt 缺 {company} 时抛错(防约束失效)
  - 模板维度 label 重复时抛错
"""

from __future__ import annotations

import pytest

from company_report_kit.outlines import UNLISTED_TEMPLATE
from company_report_kit.outlines.base import OutlineTemplate, ResearchDimension


def test_unlisted_template_has_five_dimensions() -> None:
    """非上市模板固定 5 个维度,顺序即章节顺序."""
    assert len(UNLISTED_TEMPLATE.dimensions) == 5
    assert [d.label for d in UNLISTED_TEMPLATE.dimensions] == ["投融资", "竞品", "团队", "业务", "财务"]


def test_unlisted_template_metadata() -> None:
    """模板名与适用公司类型."""
    assert UNLISTED_TEMPLATE.name == "unlisted-company"
    assert UNLISTED_TEMPLATE.company_type == "unlisted"


def test_topics_for_binds_company_to_all_dimensions() -> None:
    """topics_for 生成 5 个主题,每个都绑定公司名,顺序与维度一致."""
    topics = UNLISTED_TEMPLATE.topics_for("月之暗面")
    assert len(topics) == 5
    for topic, dim in zip(topics, UNLISTED_TEMPLATE.dimensions):
        assert "月之暗面" in topic
        # 主题内容来自维度 prompt,而非模板外文本——约束点
        assert topic == dim.prompt.format(company="月之暗面")


def test_topics_for_no_orphan_placeholder() -> None:
    """topics_for 填充后不应残留 {company} 占位符."""
    for topic in UNLISTED_TEMPLATE.topics_for("某公司"):
        assert "{company}" not in topic


def test_label_for_returns_dimension_label() -> None:
    """label_for 返回对应维度标签,越界回退到占位标签."""
    assert UNLISTED_TEMPLATE.label_for(0) == "投融资"
    assert UNLISTED_TEMPLATE.label_for(4) == "财务"
    assert UNLISTED_TEMPLATE.label_for(5) == "第6部分"


def test_dimension_prompt_must_have_company_placeholder() -> None:
    """prompt 缺 {company} 时构造即抛错,防止 researcher 拿到未绑定公司的主题."""
    with pytest.raises(ValueError):
        ResearchDimension(label="X", prompt="研究这家公司")


def test_template_rejects_duplicate_labels() -> None:
    """维度 label 重复时抛错,避免组装章节编号歧义."""
    with pytest.raises(ValueError):
        OutlineTemplate(
            name="dup",
            company_type="x",
            dimensions=(
                ResearchDimension("同", "研究{company}甲"),
                ResearchDimension("同", "研究{company}乙"),
            ),
        )
