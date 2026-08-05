"""workflows 审查→修正闭环组件测试。

覆盖纯逻辑：审查结果渲染 / 问题条目格式化 / 结构化结果归一化 /
问题分组，以及 fix_section / run_review 的 LLM 交互(FakeModel 注入 configurable_model)。
"""

from __future__ import annotations

import re
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from company_report_kit.graph.state import ReviewIssue, ReviewResult
from company_report_kit.outlines import UNLISTED_TEMPLATE
from company_report_kit.workflows import assembly, review
from tests.conftest import FakeModel

LABEL_FOR = UNLISTED_TEMPLATE.label_for


def _config() -> RunnableConfig:
    """空 configurable → Configuration 全默认."""
    return cast(RunnableConfig, {"configurable": {}})


def _issue(**overrides: object) -> ReviewIssue:
    """构造一条默认 ReviewIssue,允许按需覆盖字段."""
    defaults: dict[str, object] = {
        "section": 1,
        "issue_type": "引用错配",
        "report_text": "2026年2月完成超7亿美元融资",
        "url": "https://example.com/a",
        "evidence": "原文未提及该融资",
        "action": "fix",
    }
    defaults.update(overrides)
    return ReviewIssue(**defaults)


# ---------- 纯逻辑:审查渲染 / 条目格式化 ----------


def test_render_review_empty_passes() -> None:
    """空问题列表渲染为"校验通过"."""
    assert review.render_review([], LABEL_FOR) == "校验通过"


def test_render_review_lists_issues_with_scope() -> None:
    """渲染含问题:按章节显示维度标签,跨章节显示"跨章节"."""
    issues = [
        _issue(section=1, issue_type="引用错配", report_text="R1"),
        _issue(section=3, issue_type="无出处", report_text="R2", action="fix"),
        _issue(section=0, issue_type="口径冲突", report_text="R3", action="adjudicate"),
    ]
    text = review.render_review(issues, LABEL_FOR)
    assert "共 3 处问题" in text
    assert "[引用错配] 投融资" in text
    assert "[无出处] 团队" in text
    assert "[口径冲突] 跨章节" in text
    assert "R1" in text and "R3" in text


def test_render_review_nonempty_not_pass() -> None:
    """有问题列表时渲染不为"校验通过"."""
    assert review.render_review([_issue()], LABEL_FOR) != "校验通过"


def test_format_issue_for_fix() -> None:
    """问题条目格式化:含类型/原文/URL/原文,URL 有缓存时带完整原文,无 URL 时兜底."""
    url_cache = {"https://example.com/a": "原文:本轮融资3.2亿美元"}
    text = review._format_issue_for_fix(_issue(), url_cache)
    assert "类型: 引用错配" in text
    assert "报告原文: 2026年2月完成超7亿美元融资" in text
    assert "对应URL: https://example.com/a" in text
    assert "原文: 原文:本轮融资3.2亿美元" in text
    # URL 无缓存原文时标注"(无原文)"
    text_missing = review._format_issue_for_fix(_issue(url="https://example.com/other"), url_cache)
    assert "(无原文)" in text_missing
    # 无 URL 时兜底用 evidence
    text_blank = review._format_issue_for_fix(_issue(url=""), url_cache)
    assert "对应URL: (无)" in text_blank
    assert "原文实际" in text_blank


# ---------- 纯逻辑:报告编号层级组装 ----------


def test_assemble_sections_numbered_hierarchy() -> None:
    """组装:加报告级标题,章节 ## N.,小节 ### N.M,脚注重编号全局唯一."""
    s1 = """### 融资历史全景
### 一、创始期与天使轮
2023年完成天使轮。[^1]
### 二、A轮
2024年完成A轮。[^2]
[^1]: [来源A](https://a.com)
[^2]: [来源B](https://b.com)"""
    s2 = """### 竞品格局
### 直接竞品识别
竞品A领先。[^1]
[^1]: [来源C](https://c.com)"""

    out = assembly.assemble_sections("月之暗面", [s1, s2])

    # 报告级标题
    assert out.startswith("# 月之暗面研究报告")
    # 章节编号
    assert "## 1. 融资历史全景" in out
    assert "## 2. 竞品格局" in out
    # 小节继承编号 + 剥离中文序数前缀
    assert "### 1.1 创始期与天使轮" in out
    assert "### 1.2 A轮" in out
    assert "### 2.1 直接竞品识别" in out
    # 脚注重编号全局唯一:第二节的 [^1] → [^3]
    assert "[^3]: [来源C](https://c.com)" in out
    assert "竞品A领先。[^3]" in out
    assert "2023年完成天使轮。[^1]" in out
    # 中文序数前缀已从标题剥离(标题中不再有"一、")
    assert "1.1 一、创始期" not in out
    assert "1.2 二、A轮" not in out


def test_assemble_sections_strips_chinese_prefix() -> None:
    """中文序数前缀(一、十一、)从小节标题剥离."""
    s = """### 融资历史
### 十一、融资历史时间线汇总
2026年完成融资。[^1]
[^1]: [来源A](https://a.com)"""
    out = assembly.assemble_sections("公司X", [s])
    assert "### 1.1 融资历史时间线汇总" in out
    assert "十一、" not in out


def test_assemble_sections_empty_section_skipped() -> None:
    """空章节片段被跳过(不产生内容),编号仍按原始维度位置(第2个→## 2.)."""
    out = assembly.assemble_sections("公司X", ["", "### 业务\n内容[^1]\n[^1]: [来源](https://a.com)"])
    assert "## 2. 业务" in out  # 空片段占第1位,业务是第2维度
    assert not re.search(r"^## 1\. ", out, re.MULTILINE)  # 无第1章节标题


def test_extract_issues_normalizes_shapes() -> None:
    """归一化:ReviewResult 取 issues;None 返回空列表."""
    issues = [_issue()]
    assert review._extract_issues(ReviewResult(issues=issues)) == issues
    assert review._extract_issues(None) == []


# ---------- LLM 交互:fix_section / _review_issues ----------


@pytest.mark.anyio
async def test_fix_section_invokes_llm_with_topic_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    """修正节点:以 topic+问题(含URL原文)+原章节为消息,返回修正后的章节文本."""
    fake = FakeModel(responses=[AIMessage(content="修正后的章节")])
    monkeypatch.setattr(review, "configurable_model", fake)
    url_cache = {"https://example.com/a": "原文:本轮融资3.2亿美元"}

    out = await review.fix_section(
        "研究X的融资历史", "## 章节\n原内容[^1]\n[^1]: [标题](https://a.com)",
        [_issue()], url_cache, _config(),
    )
    assert out == "修正后的章节"
    # 两条消息:① 修正指令(含 topic/问题/URL原文) ② 原章节文本
    assert len(fake.invocations) == 1
    msgs = fake.invocations[0]
    assert isinstance(msgs[0], HumanMessage)
    assert "研究X的融资历史" in msgs[0].content
    assert "引用错配" in msgs[0].content
    # 问题关联 URL 的完整原文已进入修正指令
    assert "本轮融资3.2亿美元" in msgs[0].content
    # 原章节文本作为第二条消息(子串校验,避免全等脆弱)
    assert "原内容[^1]" in msgs[1].content
    assert "[^1]: [标题](https://a.com)" in msgs[1].content


@pytest.mark.anyio
async def test_review_issues_returns_structured_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """审查:结构化输出返回 ReviewResult,取出问题列表与 url_cache."""
    issues = [_issue(), _issue(section=0, issue_type="口径冲突", action="adjudicate")]
    fake = FakeModel(responses=[ReviewResult(issues=issues)])
    monkeypatch.setattr(review, "configurable_model", fake)
    monkeypatch.setattr(review.DuckDuckGoExtractor, "extract", lambda self, url: "原文正文")

    out_issues, url_cache = await review.run_review("## 章节\n[^1]: [标题](https://a.com)", _config())
    assert out_issues == issues
    # url_cache 存了脚注 URL 的完整原文,供 fix_section 复用
    assert url_cache == {"https://a.com": "原文正文"}


@pytest.mark.anyio
async def test_review_issues_no_footnotes_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """无脚注 URL 时跳过审查(返回 None 与空 cache),不触发 LLM."""
    fake = FakeModel()
    monkeypatch.setattr(review, "configurable_model", fake)

    out_issues, url_cache = await review.run_review("## 章节\n无脚注", _config())
    assert out_issues is None
    assert url_cache == {}
    assert fake.invocations == []


@pytest.mark.anyio
async def test_review_issues_structured_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结构化输出抛异常时按"跳过审查"处理(返回 None),但 url_cache 仍保留原文."""
    fake = FakeModel(responses=[ValueError("no support")])
    monkeypatch.setattr(review, "configurable_model", fake)
    monkeypatch.setattr(review.DuckDuckGoExtractor, "extract", lambda self, url: "原文正文")

    out_issues, url_cache = await review.run_review("## 章节\n[^1]: [标题](https://a.com)", _config())
    assert out_issues is None
    assert url_cache == {"https://a.com": "原文正文"}


@pytest.mark.anyio
async def test_review_issues_extract_failure_still_reviews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extract 抛异常时 return_exceptions 捕获,sources_text 含"提取失败",仍走审查."""
    issues = [_issue()]
    fake = FakeModel(responses=[ReviewResult(issues=issues)])
    monkeypatch.setattr(review, "configurable_model", fake)

    def _boom_extract(self, url):
        raise RuntimeError("网络失败")

    monkeypatch.setattr(review.DuckDuckGoExtractor, "extract", _boom_extract)

    out_issues, url_cache = await review.run_review("## 章节\n[^1]: [标题](https://a.com)", _config())
    assert out_issues == issues
    assert len(fake.invocations) == 1  # 一次 ainvoke: LLM 审查
    # 提取失败信息进入 prompt
    assert "提取失败" in fake.invocations[0][0].content
    # 提取失败也进 url_cache,标注让修正环节知道无法核实
    assert "提取失败" in url_cache["https://a.com"]


@pytest.mark.anyio
async def test_review_issues_structured_returns_empty_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结构化输出为空 ReviewResult → 无问题."""
    fake = FakeModel(responses=[ReviewResult(issues=[])])
    monkeypatch.setattr(review, "configurable_model", fake)
    monkeypatch.setattr(review.DuckDuckGoExtractor, "extract", lambda self, url: "原文正文")

    out_issues, url_cache = await review.run_review("## 章节\n[^1]: [标题](https://a.com)", _config())
    assert out_issues == []
    assert url_cache == {"https://a.com": "原文正文"}


def test_group_fixable_issues() -> None:
    """分组逻辑:section>=1 且 action=fix 归入修正;跨章节/adjudicate 归裁决。"""
    in_range = _issue(section=2, action="fix")
    cross = _issue(section=0, action="adjudicate")
    adjudicated_action = _issue(section=1, action="adjudicate")
    fix_groups, adjudicated = review.group_fixable_issues([in_range, cross, adjudicated_action])
    assert fix_groups == {2: [in_range]}
    assert adjudicated == [cross, adjudicated_action]
