"""非上市公司研究快照:固定 5 维度并行派发 researcher,汇总成报告。

不走 clarify → write_brief → supervisor 动态拆分,直接按固定大纲派发 5 个
researcher 子图(投融资/竞品/团队/业务/财务),汇聚 notes 后调
final_report_generation 生成报告。

用法:
    python run_snapshot.py "月之暗面"
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from company_report_kit.graph.nodes import final_report_generation
from company_report_kit.graph.researcher import researcher_subgraph

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


def _config() -> RunnableConfig:
    """构造含 api_key 的运行时配置。"""
    return cast(RunnableConfig, {
        "configurable": {
            "thread_id": "snapshot-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
        }
    })


async def run_snapshot(company: str) -> str:
    """按固定大纲并行派发 5 个 researcher,汇聚后生成报告。

    Args:
        company: 公司名称。

    Returns:
        最终报告文本。
    """
    config = _config()
    topics = [t.format(company=company) for t in OUTLINE]
    print(f"研究目标: {company}")
    print(f"派发 {len(topics)} 个 researcher 并行研究...")
    print("=" * 60)

    # 并行派发 researcher 子图
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

    # 汇聚 notes
    notes: list[str] = []
    for i, r in enumerate(results):
        label = ["投融资", "竞品", "团队", "业务", "财务"][i]
        if isinstance(r, Exception):
            print(f"  [{label}] 失败: {r}")
            notes.append(f"({label}研究失败: {r})")
        else:
            compressed = r.get("compressed_research", "")
            print(f"  [{label}] 完成, {len(compressed)} 字符 notes")
            notes.append(compressed)

    print("=" * 60)
    print("生成报告...")
    print(f"notes 总长度: {sum(len(n) for n in notes)} 字符")

    # 调 final_report_generation 生成报告
    state = cast("AgentState", {
        "notes": notes,
        "research_brief": f"{company}非上市公司研究快照:覆盖投融资历史、竞品与技术对比、"
        f"核心团队与治理、业务与商业化、财务快照五个维度。",
        "messages": [HumanMessage(content=f"研究{company}的非上市公司情况")],
    })
    cmd = await final_report_generation(state, config)
    report = cmd.update.get("final_report", "") if cmd.update else ""
    return report


def main() -> None:
    """CLI 入口:解析公司名,跑快照,保存报告。"""
    parser = argparse.ArgumentParser(description="非上市公司研究快照")
    parser.add_argument("company", help="公司名称(如 月之暗面)")
    args = parser.parse_args()

    report = asyncio.run(run_snapshot(args.company))

    print("=" * 60)
    print("最终报告")
    print("=" * 60)
    print(report[:2000] + ("..." if len(report) > 2000 else ""))

    os.makedirs("outputs", exist_ok=True)
    safe = args.company.replace("/", "_").replace("\\", "_")[:30]
    filename = f"outputs/{safe}_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存: {filename}")


if __name__ == "__main__":
    main()
