# 迁移方案：FinSight → LangGraph 编排（阶段 0 决策记录）

## 背景

Company Report Kit 的目标是从 FinSight（自研 asyncio 调度）迁移到以 LangGraph 为主的编排架构，保留 FinSight 的核心资产（Code Agent、金融工具、公司 prompts、VLM 制图）作为节点实现。

## 阶段 0：决策对齐

### 已确认决策

1. **代码位置**：在 `D:\summer\IndustryChain-Report-Kit` 内从零搭建，不新开仓库。理由：该目录已是空的 PRD 项目，且 PRD/README/CHANGELOG 已就位。

2. **LangGraph 版本**：对齐 open_deep_research 的版本栈（`langgraph>=0.5.4`、`langgraph-cli[inmem]`、Python 3.11）。理由：open_deep_research 是直接参考实现，版本对齐后节点结构可 1:1 对照。

3. **LLM 接入层**：迁移到 LangChain `init_chat_model()`，弃用 FinSight 的自研 [llm.py](D:/summer/FinSight/src/utils/llm.py)。理由：与 LangGraph 一致，多模型切换现成，避免维护两套 LLM 封装。

4. **状态分层**：图状态只承载编排数据（当前 brief、阶段进度、topic 列表），证据/分析结果走证据库。两层分开，不用 LangGraph state 替代证据库。（证据库实现见决策 13，已从 FinSight Variable Memory 调整为自建 sqlite+向量库。）

5. **公司领域 prompts**：直接复用 FinSight 的 [financial_company_prompts.yaml](D:/summer/FinSight/src/agents/report_generator/prompts/financial_company_prompts.yaml) 作为初版。

6. **数据源**：复用 FinSight 已验证的 AkShare + efinance + yfinance 组合，不重新选型。

7. **Code Agent 保留**：AsyncCodeExecutor 作为节点内执行器保留，不替换成纯工具调用。

8. **VLM 反馈制图**：作为嵌套子图实现，保留 FinSight 的 draw→render→judge→feedback 闭环。

9. **Memory 注入方式**：通过 `config["configurable"]["memory"]` 注入，不挂图 state。理由：Memory 是有状态服务，挂 state 会被 reducer 干扰。

10. **证据库实现来源**：~~从 FinSight 拷贝 variable_memory.py 独立维护~~ → 改为自建 sqlite + 向量库父子分块（见决策 13）。

### 决策调整（2026-07-30）

11. **合并 research_brief 与 research_plan**：删 research_plan 节点，write_brief 产出后直接 interrupt 等待人工确认，由 brief 承担 PRD 第 5 节 "研究计划生成与人工确认"环节。理由：brief 与 plan 产出高度重叠，open_deep_research 已验证 supervisor 基于 brief 用 ConductResearch 动态拆分专题可行（RACE 0.43），单次 HIL 优于两次。**【已执行：write_brief 节点 interrupt HIL 落地】**

12. **搜索工具层自建（search_tools），非迁移 FinSight**：自建 `src/company_report_kit/search_tools/`，含 DeepSeek / Tavily / DuckDuckGo 三个搜索器 + `ddg_extract` 正文提取 + 统一接口 `WebSearcher` + `@tool`。理由：公司研报需多源搜索（含无需 key 的 DDG 备选）+ 正文提取（补 DeepSeek encrypted_content 不可解的 gap）+ 统一接口供 researcher 绑定；FinSight 工具是 Code Agent 内的 financial/macro 数据采集，属不同层，不直接迁移。

13. **证据库用 sqlite + 向量库父子分块，替代 FinSight Variable Memory（调整决策 4/10）**：sqlite 存 `Source(url/title/content/raw_content)` 持久化（URL 主键去重省重爬），向量库存 `content` 嵌入（检索 + 知识去重）。Agent 写章节时检索 content → 查 sqlite 取 raw_content。理由：可复用/可追溯/省重爬，贴合 PRD 证据治理层；FinSight Variable Memory 与自研调度耦合（`get_or_create_agent` 等），自建更干净。

### 待后续阶段确认的决策

- 节点粒度（独立节点 vs 子图）：阶段 1 已基本确认（主图 + supervisor 子图 + researcher 子图三层）
- 工具适配方式：阶段 3 确认（researcher bind search_tools 的 @tool）
- VLM 模型选择：阶段 4 确认
- 章节并行 vs 串行：阶段 5 确认
- 质检是否强制门槛：阶段 6 确认
- 试点公司与验收标准：阶段 7 确认

## 当前架构

```
LangGraph 图（编排层，已落地）
  START
    → clarify           interrupt() 做 HIL 澄清
    → write_brief       产出 brief,interrupt() 等用户确认（已合并 research_plan）
    → supervisor        Send() 并行派发 research units
         每个 researcher 子图：
           researcher ⇄ researcher_tools（bind search_tools 的 @tool）→ compress_research → write_section
    → final_report      汇总 notes 写最终报告（带 token 降级重试）

固化流水线（workflows,已落地）
  snapshot —— 非上市 5 维度大纲并行派发 → assembly 组装 → review 审查修正闭环 → 重审
    outlines/ 提供 ResearchDimension/OutlineTemplate 圈定 researcher 调研范围

证据库（待建，规划自建 sqlite + 向量库）
  - sqlite: source(url/title/content/raw_content) 持久化,URL 去重省重爬
  - 向量库: content 嵌入检索 + 知识去重

搜索工具层（search_tools,已落地）
  - DeepSeek / Tavily / DuckDuckGo 搜索 + ddg_extract 正文提取
  - 统一接口 WebSearcher + @tool,供 researcher 绑定
```

## 阶段路线图

| 阶段 | 目标 | 状态 |
|---|---|---|
| 0 | 决策对齐 | ✅ 完成 |
| 1 | LangGraph 骨架 + 头尾节点接 LLM | ✅ 完成（clarify/brief/final_report 接 LLM；supervisor 子图派发 researcher_subgraph 落地） |
| 2 | 证据库（sqlite+向量库） | 待开始（搜索工具已接入 researcher；正文提取经审查闭环复用；sqlite+向量库未实现） |
| 3 | 研究子图（researcher ⇄ tools + compress_research） | ✅ 完成（事件簇分组去重 + 全文提取；supervisor 用 Send 并行派发） |
| 4 | 分析子图 + VLM 制图 | 待开始 |
| 5 | 报告生成两阶段（outline→sections） | 部分实现（final_report 单阶段已接 LLM；快照流水线按大纲模板分章节组装） |
| 6 | 质检 + 导出 | 部分实现（审查→修正闭环落地：ReviewResult 结构化审查 + 按章节修正；HTML 导出待做） |
| 7 | 对照验证 + 调优 | 待开始 |

## 当前状态

- **主图 + supervisor 子图**（`src/company_report_kit/graph/graph.py`）：clarify → write_brief → research_supervisor（挂 supervisor 子图）→ final_report → END；Command goto 路由；MemorySaver。
- **头尾节点真接 LLM**（`graph/nodes.py`）：clarify/write_brief 用 with_structured_output + interrupt HIL；final_report 带降级重试。
- **researcher 子图**（`graph/researcher.py`）：四节点 researcher/researcher_tools/compress_research/write_section，bind `duckduckgo_web_search` + `ddg_extract_url` + `think_tool`，max_react_tool_calls 控轮数，事件簇分组去重后写章节，`section_text` 回传。
- **supervisor/supervisor_tools**：supervisor bind ConductResearch/ResearchComplete/think_tool，supervisor_tools 并行派发 researcher_subgraph（受 max_concurrent_research_units 限流）。
- **大纲模板层**（`outlines/`）：ResearchDimension/OutlineTemplate 结构化，非上市 5 维度实例（投融资/竞品/团队/业务/财务），researcher 调研范围被模板圈定。
- **固化流水线**（`workflows/`）：assembly 组装（编号层级+脚注重编号）+ review 审查修正闭环（ReviewResult 结构化审查→按章节修正→重审）+ snapshot 非上市流水线。
- **搜索工具层完整**（`search_tools/`）：3 搜索器 + ddg_extract + 统一接口 + @tool。
- **日志与终端输出**（`logging_utils.py`）：print 全面迁移至 logging + rich（RichHandler 幂等配置，markup=False 防脚注被解析）。
- **测试**（`tests/`）：118 用例（纯逻辑+mock 默认跑，network/live 标记），全部通过。
- **下一步**：阶段 2 证据库（sqlite+向量库父子分块）→ 引用精确性/跨语言去重 → HTML 导出。
