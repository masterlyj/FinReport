"""非上市公司研究快照 CLI 入口.

用法:
    python -m company_report_kit.cli.snapshot "月之暗面"

跑固定 5 维度大纲(投融资/竞品/团队/业务/财务)并行研究,组装+审查+修正,
把报告与审查结果保存到 outputs/。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from company_report_kit.workflows.snapshot import run_snapshot

# 默认配置:非上市快照固定并行 5 个 researcher,并发上限放宽到 5.
_DEFAULT_CONFIG = {
    "configurable": {
        "max_concurrent_research_units": 5,
    }
}


def _config() -> dict:
    """构造含 api_key 的运行时配置,默认并发 5."""
    configurable = dict(_DEFAULT_CONFIG["configurable"])
    configurable["api_key"] = os.getenv("DEEPSEEK_API_KEY")
    configurable["thread_id"] = "snapshot-" + datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
    return {"configurable": configurable}


def main() -> None:
    """CLI 入口:解析公司名,跑快照,保存报告+审查结果."""
    parser = argparse.ArgumentParser(description="非上市公司研究快照")
    parser.add_argument("company", help="公司名称(如 月之暗面)")
    args = parser.parse_args()

    report, review = asyncio.run(run_snapshot(args.company, _config()))

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
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    report_file = f"outputs/{safe}_snapshot_{ts}.md"
    review_file = f"outputs/{safe}_review_{ts}.md"
    # CLI 属顶层入口,文件写 outputs/ 用同步 open 即可(非 async 性能关键路径)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    with open(review_file, "w", encoding="utf-8") as f:
        f.write(review)
    print(f"\n报告已保存: {report_file}")
    print(f"审查已保存: {review_file}")


if __name__ == "__main__":
    sys.exit(main())
