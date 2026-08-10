# Deep Research 项目流程汇总

本文档对照梳理两个 Deep Research 项目：`D:\summer\open_deep_research`（LangChain 官方通用版）与 `D:\summer\FinSight`（金融领域专用版）。两者都采用 supervisor/researcher 分层思想，但在编排框架、Agent 模型、记忆机制与产出形态上差异显著。可作为 Company Report Kit 设计公司研究报告流程时的参考样本。

---

## 一、open_deep_research（LangChain 官方开源版）

### 定位

LangChain 团队维护的通用型 Deep Research Agent，基于 LangGraph 编排，支持多模型/多搜索 API/MCP，在 [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 排行榜上排名前列（RACE 0.4344）。

### 技术栈

- **编排**：LangGraph 状态图（声明式）
- **LLM**：LangChain `init_chat_model()`，支持 OpenAI / Anthropic / Google / Groq / DeepSeek
- **搜索**：Tavily / OpenAI / Anthropic 原生 web search / MCP servers

### 核心文件

| 文件 | 作用 |
|---|---|
| `src/open_deep_research/deep_researcher.py` | 主 LangGraph 图定义，入口节点 `deep_researcher` |
| `src/open_deep_research/configuration.py` | 可配置项：4 类模型 + 搜索 API + 并发上限 |
| `src/open_deep_research/state.py` | 三层状态：`AgentState` / `SupervisorState` / `ResearcherState` |
| `src/open_deep_research/prompts.py` | 各阶段系统提示词 |
| `src/open_deep_research/utils.py` | 工具加载、token 判定、原生搜索识别 |
| `src/legacy/graph.py` | 早期 plan-and-execute + human-in-the-loop 实现 |
| `src/legacy/multi_agent.py` | 早期 supervisor-researcher 多 agent 并行实现 |

### 主流程

```
START
  → clarify_with_user      （可选）判断是否需要向用户追问澄清
  → write_research_brief    把用户消息转成结构化 ResearchBrief，初始化 supervisor
  → research_supervisor     子图：supervisor ⇄ supervisor_tools
        ├── supervisor 调用 think_tool / ConductResearch / ResearchComplete
        └── supervisor_tools 并行派发 ≤ max_concurrent_research_units 个 researcher 子图
              每个 researcher 子图：researcher ⇄ researcher_tools → compress_research
  → final_report_generation 用 final_report_model 汇总 notes 写最终报告
END
```

### 四个可独立配置的 LLM 角色

| 角色 | 默认模型 | 职责 |
|---|---|---|
| Summarization | `openai:gpt-4.1-mini` | 搜索 API 结果摘要 |
| Research | `openai:gpt-4.1` | supervisor + researcher 主体 |
| Compression | `openai:gpt-4.1` | 把单次研究压缩为结构化笔记 |
| Final Report | `openai:gpt-4.1` | 综合所有 notes 写最终报告 |

### 关键机制

- **Supervisor 派发**：supervisor 通过 `ConductResearch` 把研究拆成多个 topic，并行发给多个 researcher 子图，受 `max_concurrent_research_units`（默认 5）限制。
- **迭代上限**：`max_researcher_iterations`（默认 6，supervisor 反思轮数）与 `max_react_tool_calls`（默认 10，单 researcher 工具调用上限）。
- **压缩回传**：每个 researcher 跑完后走 `compress_research`，输出 `compressed_research` + `raw_notes`，回传 supervisor。
- **最终报告降级**：`final_report_generation` 带 token 超限自动重试，逐步截断 findings 到 90%、再 90%…直至成功或重试上限。
- **Legacy 对比**：`legacy/graph.py` 是 plan-and-execute + human-in-the-loop，`legacy/multi_agent.py` 是 supervisor-researcher 多 agent 并行；主版本是两者的简化统一。

### 入口

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 \
  langgraph dev --allow-blocking
```

启动后 LangGraph Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

---

## 二、FinSight（金融领域专用版）

### 定位

面向真实金融研究的 multi-agent 系统：输入一个股票代码，一键产出 2 万字 + 带专业图表的研究报告。论文 [arXiv:2510.16844](https://arxiv.org/abs/2510.16844)，AFAC2025 Track 4 第一名（1/1289）。

### 技术栈

- **编排**：纯 Python asyncio（无 LangGraph），优先级分组调度
- **Agent 模型**：Code Agent（CAVM 架构，LLM 写 Python 操纵统一变量空间）
- **数据源**：AkShare / efinance / yfinance（A 股 / 港股 / 美股）
- **爬虫**：Crawl4AI
- **制图**：matplotlib + VLM 反馈式迭代

### 核心文件

| 文件 | 作用 |
|---|---|
| `run_report.py` | 主入口，按优先级分批调度 agent |
| `src/agents/base_agent.py` | 自研 BaseAgent，含 checkpoint/resume、AsyncCodeExecutor |
| `src/memory/variable_memory.py` | 统一变量记忆 `Memory`，pickle 持久化 + 语义检索 |
| `src/agents/data_collector/` | 数据采集 agent |
| `src/agents/data_analyzer/` | 分析 + 制图 agent |
| `src/agents/report_generator/` | 两阶段报告生成 agent |
| `src/agents/search_agent/` | 可选深度搜索子 agent |
| `src/tools/` | financial / industry / macro / web 四类工具，自动注册 |
| `my_config.yaml` | 公司研究报告配置 |
| `my_config_industry.yaml` | 行业研究报告配置 |

### 主流程

```
run_report(resume=True, max_concurrent=3)

1. 读 my_config.yaml：target_name / stock_code /
   custom_collect_tasks / custom_analysis_tasks

2. Memory.load()  从 pickle checkpoint 恢复（断点续跑）

3. LLM 生成补充任务
     - generate_collect_tasks()  补充数据采集任务（max_num=5）
     - generate_analyze_tasks() 补充分析任务（max_num=5）
   与 custom_* 合并去重

4. 为每个任务通过 memory.get_or_create_agent() 建 agent，
   附带 priority：
     priority 1 → DataCollector   （数据采集）
     priority 2 → DataAnalyzer    （分析 + 制图）
     priority 3 → ReportGenerator （最终报告）

5. 按 priority 分组执行
     先全部跑完 priority 1（≤ 3 并发）
     再全部跑完 priority 2
     最后跑 priority 3
   每个 agent 内部 async_run() 是自己的 ReAct 循环
   （LLM → code executor / tools → 反思 → 再调用 …）

6. memory.save()  持久化全部状态
```

### 四类 Agent 的角色

| Agent | 职责 |
|---|---|
| DataCollector | 调用 financial/macro/industry/web 工具采集原始数据，写入 Memory |
| DataAnalyzer | 基于 Memory 中数据执行 Python 代码做分析，调用 VLM 生成并迭代修正图表（Iterative Vision-Enhanced Mechanism，默认最多 3 轮反馈），产出 `AnalysisResult` |
| ReportGenerator | 两阶段写作——先列大纲再逐节生成，套用 docx 模板输出可发表级报告 |
| DeepSearchAgent | 可选深度搜索子 agent，作为 tool 被父 agent 调用 |

### 关键机制

- **Variable Memory（CAVM）**：所有 agent 共享同一 `Memory` 对象，数据、分析结果、依赖关系、embedding 全部存 pickle；支持 `retrieve_relevant_data()` 语义检索。
- **Code Agent**：agent 不只调工具，还能在统一变量空间里执行 Python 代码动态操作数据/工具/记忆，透明可复现。
- **VLM 反馈制图**：LLM 写 matplotlib 代码 → 渲染图 → VLM 审图（图例/刻度/信息密度）→ 不合格则反馈修改，循环至通过或 `max_iterations`。
- **Checkpoint/Resume**：每个 agent 有 `.cache/latest.pkl`，Memory 有 `memory.pkl`；`resume=True` 时跳过已完成 agent，中断后可续跑。
- **两阶段写作**：ReportGenerator 先生成大纲（`outline_latest.pkl`）再逐节扩展（`section_0.pkl`...），保证结构性与深度。

### 入口

```bash
python run_report.py
```

配置改 `my_config.yaml`（公司）或 `my_config_industry.yaml`（行业）。

---

## 三、两者对照

| 维度 | open_deep_research | FinSight |
|---|---|---|
| 领域 | 通用 Deep Research | 金融研究专用 |
| 编排 | LangGraph 状态图（声明式） | asyncio + 优先级调度（命令式） |
| Agent 模型 | ReAct（LLM bind_tools） | Code Agent（LLM 写 Python 操纵变量空间） |
| 记忆 | 图状态 `notes`/`raw_notes` 覆盖式传递 | 独立 `Memory` 对象，pickle 持久化 + 语义检索 |
| 并发 | supervisor 并行派发多个 researcher 子图 | priority 分组 + `Semaphore` 限流 |
| 数据源 | 通用 web search + MCP | 金融专用工具（AkShare/efinance/yfinance）+ 爬虫 |
| 制图 | 无 | VLM 反馈式迭代制图 |
| 输出 | Markdown 报告 | 2 万字 docx + 嵌入图表 |
| 断点续跑 | LangGraph checkpoint | 自研 pickle checkpoint + resume |
| 评测 | Deep Research Bench（RACE 0.43） | 自评 vs OpenAI/Gemini Deep Research，总分 8.09 |

**一句话总结**：open_deep_research 是"通用、可配置、LangGraph 范式"的 Deep Research 参考实现；FinSight 是"金融场景、Code Agent + VLM 制图、可产出可发表报告"的领域加强版。两者都用了 supervisor/researcher 分层思想，但 FinSight 用代码执行代替了纯工具调用，并加了视觉反馈闭环。

---

## 四、对 Company Report Kit 的启发

Company Report Kit 的 PRD 已明确五层架构（研究流程层 / 公司领域层 / 证据治理层 / 质量控制层 / 交付层），上述两个项目可作为流程与机制的参考样本：

- **流程编排**：可借鉴 open_deep_research 的"澄清 → Brief → 计划 → 并行研究 → 压缩 → 报告"分层，把"研究计划生成与人工确认"作为人类在环控制点；FinSight 的"按优先级分批 + 信号量限流"可直接迁移为 DataCollector / DataAnalyzer / ReportGenerator 的三批执行模型。
- **公司领域层**：FinSight 的公司级 `my_config.yaml` + 公司报告 prompts 已是现成参照，可对照 PRD 的"基本面、业务拆解、财务分析、估值、风险"拆分专题。
- **证据治理**：open_deep_research 的 `compress_research` → `raw_notes`/`notes` 是 Evidence 雏形；FinSight 的 Variable Memory + 语义检索可作为证据库与来源映射的实现参考。
- **公司建模**：FinSight 的 Code Agent + Variable Memory 思路可迁移到 CompanyGraph 构建——让 agent 在统一变量空间里执行代码动态拼装业务分部、子公司、产品线节点。
- **财务制图**：FinSight 的 VLM 反馈式制图闭环可直接借鉴用于财务可视化（营收拆解、毛利率趋势、估值对比），解决"AI 画图难看"的问题。
- **质量检查**：open_deep_research 的 token 降级、FinSight 的 VLM 反馈式迭代都体现了"产出后自动审校并修正"的闭环，可对照 PRD 中"引用检查/口径冲突/缺口识别/报告审校"。
- **断点续跑**：两者都做了 checkpoint/resume，对长流程的公司研究报告生成（多专题、多轮采集、财报解析）同样必要。
- **数据接入**：FinSight 已验证 AkShare/efinance/yfinance 的可行性，Company Report Kit 的 A 股 / 港股 / 美股覆盖可直接复用。

