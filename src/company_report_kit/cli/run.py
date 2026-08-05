"""Company Report Kit 交互式 CLI（通用深度报告）.

用法:
    python -m company_report_kit.cli.run "腾讯控股"
    python -m company_report_kit.cli.run "腾讯控股" --no-clarify

流程:
    1. clarify: 如需追问，终端显示问题等用户输入
    2. write_brief: 生成研究简报，显示后等用户确认
    3. supervisor + researcher: 自动搜索研究
    4. final_report: 生成报告，打印并保存到 outputs/

日志走 logging + rich（进度/诊断），面向用户的报告正文/追问/文件路径走
共享 rich Console 直接输出。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()

from company_report_kit.graph.graph import graph
from company_report_kit.logging_utils import console, get_logger, setup_logging

logger = get_logger("cli.run")


def get_config(topic, allow_clarification):
    """构造 LangGraph 运行时配置."""
    return {
        "configurable": {
            "thread_id": "report-" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "allow_clarification": allow_clarification,
        }
    }


async def run(topic, allow_clarification):
    """跑通用深度报告图,处理 clarify/write_brief 两次人工确认."""
    config = get_config(topic, allow_clarification)
    console.print(f"研究目标: {topic}", style="bold cyan")
    console.rule()

    messages = [HumanMessage(content=topic)]
    while True:
        logger.info("正在分析请求...")
        result = await graph.ainvoke({"messages": messages}, config=config)
        state = await graph.aget_state(config)

        if not state.next:
            msgs = state.values.get("messages", [])
            last_msg = str(msgs[-1].content) if msgs else ""
            if last_msg:
                console.print()
                console.print(f"系统追问: {last_msg}", style="bold yellow")
                answer = input("请回答（或直接回车跳过追问）: ").strip()
                if answer:
                    messages = [HumanMessage(content=answer)]
                else:
                    config["configurable"]["allow_clarification"] = False
                    messages = [HumanMessage(content=topic)]
                continue
            else:
                logger.error("流程意外结束")
                return

        if "write_brief" in (state.next or ()):
            # 从 interrupt 取 research_brief
            brief = ""
            tasks = getattr(state, "tasks", None)
            if tasks:
                for t in tasks:
                    if hasattr(t, "interrupts") and t.interrupts:
                        for intr in t.interrupts:
                            val = getattr(intr, "value", {})
                            if isinstance(val, dict) and "research_brief" in val:
                                brief = val["research_brief"]
            console.rule("研究简报")
            if brief:
                console.print(brief[:500] + ("..." if len(brief) > 500 else ""))
            confirm = input("确认简报？(回车确认 / 输入修改意见): ").strip()
            if confirm:
                resume_value = {"feedback": confirm, "approved": False}
            else:
                resume_value = True

            logger.info("正在开展研究（可能需要几分钟）...")
            result = await graph.ainvoke(Command(resume=resume_value), config=config)
            break

        logger.info("当前节点: %s", state.next)
        break

    console.rule("最终报告")
    final_report = result.get("final_report", "")
    if final_report:
        console.print(final_report[:2000] + ("..." if len(final_report) > 2000 else ""))
        os.makedirs("outputs", exist_ok=True)
        safe_topic = topic.replace("/", "_").replace(chr(92), "_")[:30]
        filename = "outputs/" + safe_topic + "_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + ".md"
        # CLI 属顶层入口,文件写 outputs/ 用同步 open 即可(非 async 性能关键路径)
        with open(filename, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            f.write(final_report)
        console.print()
        console.print(f"报告已保存: {filename}", style="bold green")
    else:
        logger.error("报告生成失败")
        if result.get("research_brief"):
            logger.error("研究简报: %s", str(result["research_brief"])[:200])


def main():
    """CLI 入口:解析公司名/澄清开关,跑通用深度报告."""
    parser = argparse.ArgumentParser(description="Company Report Kit - 公司深度研究报告生成")
    parser.add_argument("topic", help="公司名称或股票代码")
    parser.add_argument("--no-clarify", action="store_true", help="跳过澄清追问")
    args = parser.parse_args()
    setup_logging()
    asyncio.run(run(args.topic, not args.no_clarify))


if __name__ == "__main__":
    sys.exit(main())
