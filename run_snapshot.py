"""非上市公司研究快照:固定 5 维度并行派发 researcher,各自写章节,组装+审查。

不走 clarify → write_brief → supervisor 动态拆分,直接按固定大纲派发 5 个
researcher 子图(投融资/竞品/团队/业务/财务),每个 researcher 各自写本维度
报告章节(markdown 含脚注引用),然后代码组装(拼接+重编编号),最后主 agent
extract URL 正文审查校验。

用法:
    python run_snapshot.py "月之暗面"
"""

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from company_report_kit.configuration import Configuration
from company_report_kit.graph.researcher import researcher_subgraph
from company_report_kit.prompts import review_prompt
from company_report_kit.search_tools.ddg_extract import DuckDuckGoExtractor
from company_report_kit.utils import configurable_model, get_model_config, get_today_str, RETRY_KWARGS

# 固定研究大纲:每个维度自包含,researcher 看不到其他 researcher 的工作。
OUTLINE = [
    "研究{company}的完整融资历史。梳理每一轮融资的:宣布/完成时间、融资金额(含币种)、"
    "投前/投后估值、轮次类型(天使/A/B/C/...)、领投方与跟投方(新老股东区分)。"
    "按时间线排列,标注估值变化轨迹与关键驱动因素(产品里程碑/技术突破/市场事件)。"
    "如有融资空窗期或终止的融资计划,也要记录。投资方信息尽量穿透到具体基金/机构。",

    "研究{company}的竞品格局。首先准确识别直接竞品(同赛道、同产品形态)和间接竞品"
    "(替代方案),然后逐个对比:竞品的业务规模/融资阶段/市场地位、技术路线差异"
    "(架构/参数/性能基准/开源策略)、差异化优势与劣势。不要泛泛列举行业玩家,"
    "要聚焦与{company}直接争夺同一市场或同一技术路线的对手。如有公开的性能基准"
    "对比(榜单/评测),优先引用。",

    "研究{company}的组织架构与发展历程。包括:创始人背景(教育经历、过往创业/职业经历)、"
    "核心高管团队(CTO/CFO/COO 等关键岗位)、董事会构成(投资方董事席位)、"
    "公司发展历程(成立→关键产品节点→重要战略转折→现状)。"
    "如有重大人事变动(离职/仲裁/股权纠纷),也要记录。",

    "研究{company}的业务模式与商业化进展。包括:主力产品/服务及其营收模式、"
    "重大订单/政企合作/战略合作(含金额与时间)、上下游产业链位置(核心供应商/客户结构)、"
    "关键经营指标(用户规模/DAU/MAU/ARR/付费转化率等可得数据)。"
    "优先引用有数字的公开信息,不要笼统描述市场前景广阔。",

    "研究{company}可得的财务数据。非上市公司财务披露有限,重点搜集:各阶段公开的"
    "营收/利润/现金流数字(标注时间和口径)、融资节奏与累计融资额、资金储备(账面现金)、"
    "关键单位经济模型指标(如有)。每个数字必须标注来源和时间,区分实际披露 vs 市场传闻 vs 推测。",
]

LABELS = ["投融资", "竞品", "团队", "业务", "财务"]


def _config() -> RunnableConfig:
    """构造含 api_key 的运行时配置。"""
    return cast(RunnableConfig, {
        "configurable": {
            "thread_id": "snapshot-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
        }
    })


def assemble_sections(sections: list[str]) -> str:
    """拼接各 researcher 的章节,重编脚注引用编号使其全局唯一。

    每个 researcher 独立用 [^1][^2]...,组装时按顺序重编:
    researcher A 的 [^1]→[^1],B 的 [^1]→[^N+1],脚注定义对应重编。

    Args:
        sections: 各 researcher 产出的 section_text(含脚注)。

    Returns:
        拼接后的完整报告,脚注编号全局唯一。
    """
    assembled: list[str] = []
    footnote_offset = 0
    for section in sections:
        if not section:
            continue
        # 收集本章节的脚注编号,建立旧→新映射。
        old_nums = sorted(set(int(m) for m in re.findall(r"\[\^(\d+)\]", section)))
        if old_nums:
            num_map = {old: footnote_offset + i + 1 for i, old in enumerate(old_nums)}
            # 重编正文和脚注定义中的 [^N]。
            def _renum(m: re.Match) -> str:
                return f"[^{num_map[int(m.group(1))]}]"
            section = re.sub(r"\[\^(\d+)\]", _renum, section)
            footnote_offset = max(num_map.values())
        assembled.append(section)
    return "\n\n".join(assembled)


async def review_report(report: str, config: RunnableConfig) -> str:
    """extract 脚注 URL 正文,LLM 校验报告事实声称与原文是否一致。

    Args:
        report: 组装后的完整报告(含脚注定义)。
        config: 运行时配置。

    Returns:
        审查结果文本(问题列表或"校验通过")。
    """
    # 提取脚注定义里的 URL: [^1]: [标题](URL) 或 [^1]: URL
    footnotes = dict(re.findall(r"\[\^(\d+)\]:\s*(?:\[[^\]]*\]\()?(https?://[^\s)]+)", report))
    if not footnotes:
        return "未找到脚注 URL,跳过审查。"

    print(f"审查:提取 {len(footnotes)} 个脚注 URL 正文...")
    extractor = DuckDuckGoExtractor()
    urls = list(footnotes.values())
    texts = await asyncio.gather(*[extractor.extract(u) for u in urls], return_exceptions=True)

    sources_text = "\n\n".join(
        f"[^{num}] {url}\n{(t if isinstance(t, str) else f'提取失败: {t}')[:800]}"
        for (num, url), t in zip(footnotes.items(), texts)
    )

    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    reviewer = configurable_model.with_retry(**RETRY_KWARGS).with_config(model_config)
    prompt = review_prompt.format(report=report, sources=sources_text)
    response = await reviewer.ainvoke([HumanMessage(content=prompt)])
    return str(response.content)


async def run_snapshot(company: str) -> tuple[str, str]:
    """按固定大纲并行派发 5 个 researcher,组装+审查。

    Args:
        company: 公司名称。

    Returns:
        (报告文本, 审查结果) 二元组。
    """
    config = _config()
    topics = [t.format(company=company) for t in OUTLINE]
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
        label = LABELS[i] if i < len(LABELS) else str(i)
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
    report = assemble_sections(sections)
    print(f"组装完成, {len(report)} 字符")

    # 审查校验
    print("审查校验...")
    review = await review_report(report, config)
    print(f"审查完成: {review[:200]}")

    return report, review


def main() -> None:
    """CLI 入口:解析公司名,跑快照,保存报告+审查结果。"""
    parser = argparse.ArgumentParser(description="非上市公司研究快照")
    parser.add_argument("company", help="公司名称(如 月之暗面)")
    args = parser.parse_args()

    report, review = asyncio.run(run_snapshot(args.company))

    print("=" * 60)
    print("最终报告")
    print("=" * 60)
    print(report[:2000] + ("..." if len(report) > 2000 else ""))

    print("\n" + "=" * 60)
    print("审查结果")
    print("=" * 60)
    print(review[:2000])

    os.makedirs("outputs", exist_ok=True)
    safe = args.company.replace("/", "_").replace("\\", "_")[:30]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"outputs/{safe}_snapshot_{ts}.md"
    review_file = f"outputs/{safe}_review_{ts}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    with open(review_file, "w", encoding="utf-8") as f:
        f.write(review)
    print(f"\n报告已保存: {report_file}")
    print(f"审查已保存: {review_file}")


if __name__ == "__main__":
    main()
