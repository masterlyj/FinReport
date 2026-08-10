# CLAUDE.md — Company Report Kit

@AGENTS.md

公司深度研究报告生成系统。入口：`cli.run`（通用深度报告）、`cli.snapshot`（非上市快照）。

## 常用命令

```bash
uv sync
cp .env.example .env                       # 填 DEEPSEEK_API_KEY（必需）、TAVILY_API_KEY（可选）

python -m company_report_kit.cli.run "腾讯控股"          # 通用深度报告（clarify+write_brief 两次人工确认）
python -m company_report_kit.cli.run "腾讯控股" --no-clarify
python -m company_report_kit.cli.snapshot "月之暗面"     # 非上市快照（固定 5 维度大纲）

pytest                                     # 日常：单元+组件（默认 deselect network/live）
pytest -m network                          # 需网络、无需 key
pytest -m live                             # 需真实 API key
pytest -o addopts=""                       # 全量含集成
ruff check .
mypy src
```

## 架构约束

- **报告用拼接模式生成**：researcher 各写章节（带 `[^1]` 脚注）→ `assemble_sections` 代码拼接重编编号 → 快照走 review 闭环、主图走 `polish_report` 润色（只润色，严禁新增/修改事实）。**不要退回"LLM 用 notes 重生成整份报告"**——信息层层丢失导致幻觉（已返工一次）。
- **researcher 调研主题只能来自大纲模板** `topics_for(company)`，不扩展到大纲外。新增报告类型只加 `outlines/*.py` 模板，`workflows/snapshot.py` 复用不改。
- **证据不进图 state**：图状态只承载编排数据（brief/topics/sections）；list 字段整体覆盖用 `override_reducer`；api_key 等配置走 `config["configurable"]` 注入。

## LLM 接入约定（utils.py）

- 统一 `init_chat_model(configurable_fields=...)` + **`deepseek:` 前缀**，依赖 `langchain-deepseek` 包（pyproject 已声明，`uv sync` 后可用）。
- **DeepSeek 思考模式不支持 strict 的强制 tool_choice**（报 400）：结构化输出节点（clarify / write_brief / compress_research）必须 `thinking=False`；仅报告生成/审查节点可开思考。
- 重试统一 `RETRY_KWARGS`（指数退避 2s→60s + 0-3s 抖动，最多 8 次），应对 DeepSeek 间歇 500/503；配置经 `get_model_config` + `.with_config()` 注入。

## 日志与测试约定

- 进度/诊断走 `logging`，面向用户的最终产物（报告正文/审查结果/路径/交互）走共享 `console`。**不要用 `print` 混入**。
- 库内取 logger 用 `get_logger(name)`；CLI 入口 `setup_logging()`（幂等）；`DEEPSEEK_LOG_LEVEL=DEBUG` 排查第三方库噪音。
- 新增测试按功能选标记：`network`（需网络无 key）/ `live`（需真实 key）。图节点链式调用用 `tests/conftest.py` 的 `FakeModel` 注入替身，断言 `invocations` 验证 prompt 拼装。
