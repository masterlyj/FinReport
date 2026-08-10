"""非上市公司研究快照 CLI 入口.

用法:
    python -m company_report_kit.cli.snapshot "月之暗面"

跑固定 5 维度大纲(投融资/竞品/团队/业务/财务)并行研究,组装+审查+修正,
把报告与审查结果保存到 outputs/。

日志走 logging + rich（进度/诊断），面向用户的报告/审查正文与文件路径走
共享 rich Console 直接输出。
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from typing import cast

from dotenv import load_dotenv

load_dotenv()

from langchain_core.runnables import RunnableConfig

from company_report_kit.logging_utils import console, setup_logging
from company_report_kit.workflows.snapshot import run_snapshot

# 默认配置:非上市快照固定并行 5 个 researcher,并发上限放宽到 5.
_DEFAULT_CONFIG = {
    "configurable": {
        "max_concurrent_research_units": 5,
    }
}


def _config() -> dict:
    """构造含 api_key 的运行时配置,默认并发 5."""
    # 显式标注元素类型 object:config 混入 int(并发数)/str(api_key/thread_id),
    # 不标注会被 mypy 按首键推断成 dict[str, int] 导致赋值报错。
    configurable: dict[str, object] = dict(_DEFAULT_CONFIG["configurable"])
    configurable["api_key"] = os.getenv("DEEPSEEK_API_KEY")
    configurable["thread_id"] = "snapshot-" + datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
    return {"configurable": configurable}


def main() -> None:
    """CLI 入口:解析公司名,跑快照,保存报告+审查结果."""
    parser = argparse.ArgumentParser(description="非上市公司研究快照")
    parser.add_argument("company", help="公司名称(如 月之暗面)")
    args = parser.parse_args()
    setup_logging()

    report, review = asyncio.run(
        run_snapshot(args.company, cast("RunnableConfig", _config()))
    )

    console.rule("最终报告")
    console.print(report[:2000] + ("..." if len(report) > 2000 else ""))

    console.rule("审查结果")
    console.print(review[:2000])

    os.makedirs("outputs", exist_ok=True)
    safe = args.company.replace("/", "_").replace("\\", "_")[:30]
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    report_file = f"outputs/{safe}_snapshot_{ts}.md"
    review_file = f"outputs/{safe}_review_{ts}.md"
    # CLI 属顶层入口,文件写 outputs/ 用同步 open 即可(非 async 性能关键路径)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    with open(review_file, "w", encoding="utf-8") as f:
        f.write(review)
    console.print()
    console.print(f"报告已保存: {report_file}", style="bold green")
    console.print(f"审查已保存: {review_file}", style="bold green")


if __name__ == "__main__":
    main()
