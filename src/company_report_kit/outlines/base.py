"""大纲模板框架：把固化大纲从代码分散处抽离为可复用的模板对象.

每个报告类型（非上市公司/行业/上市快报…）对应一个 OutlineTemplate，
内部是若干 ResearchDimension。workflow 层只认 OutlineTemplate 接口，
不关心具体是哪个模板——新增报告类型只需新增一个模板文件。

约束 researcher 的机制：
  researcher 的 research_topic 只能来自 template.topics_for(company)，
  即模板维度 prompt 填充公司名后的文本。子 Researcher 看不到模板之外的
  调研范围，从而实现"只能围绕大纲主题调研"。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchDimension:
    """单个研究维度：一个章节对应一个 researcher 调研主题.

    Attributes:
        label: 章节标题（如 "投融资"），组装报告时用于 ## N. 编号.
        prompt: 调研指令模板，必须含 {company} 占位符，researcher 据此调研.
    """

    label: str
    prompt: str

    def __post_init__(self) -> None:
        """校验 prompt 必须可填充公司名，防止 researcher 拿到未绑定公司的主题."""
        if "{company}" not in self.prompt:
            raise ValueError(
                f"dimension prompt 必须含 {{company}} 占位符: {self.label}"
            )


@dataclass(frozen=True)
class OutlineTemplate:
    """一份固化的标准大纲.

    Attributes:
        name: 模板唯一名（如 "unlisted-company"）.
        company_type: 适用公司类型（"unlisted" / 未来 "listed" / "industry"）.
        dimensions: 有序维度列表，顺序即报告章节顺序.
    """

    name: str
    company_type: str
    dimensions: tuple[ResearchDimension, ...]

    def topics_for(self, company: str) -> list[str]:
        """为指定公司生成全部 researcher 调研主题.

        子 Researcher 只能拿到这里的主题——模板约束的入口.

        Args:
            company: 公司名称.

        Returns:
            与 dimensions 等长、顺序一致的调研主题列表.
        """
        return [d.prompt.format(company=company) for d in self.dimensions]

    def label_for(self, index: int) -> str:
        """章节下标 → 维度标签，越界回退到占位标签."""
        if 0 <= index < len(self.dimensions):
            return self.dimensions[index].label
        return f"第{index + 1}部分"

    def __post_init__(self) -> None:
        """校验维度 label 唯一，避免组装时章节编号歧义."""
        labels = [d.label for d in self.dimensions]
        if len(labels) != len(set(labels)):
            raise ValueError(f"维度 label 必须唯一: {labels}")
