# Datus Agents 总览

本文为 Datus Agent 系统的技术速读，梳理核心目录、工作流、节点类型以及扩展方式，帮助开发者快速理解 `Datus-agent-clickzetta` 仓库的代理实现细节。

## 1. 仓库结构速览

- `datus/agent/`：核心代理运行时，包含 `agent.py` 执行器、`workflow.py` 流程管理、`plan.py` 规划生成以及丰富的节点实现。
- `datus/cli/`：终端交互入口，`main.py` 暴露 CLI，`repl.py` 构建多轮对话工作台，`sub_agent_*` 负责子代理向导。
- `datus/api/`：FastAPI 服务入口（`server.py`），支持守护进程模式部署 REST 接口。
- `datus/storage/`：向量检索与持久化层，覆盖 schema/metric/document/external knowledge 等多类存储。
- `datus/tools/`：数据库连接与工具集成（`db_tools`、MCP、内置函数工具等）。
- `datus/prompts/`：系统提示词模板及 Prompt Manager。
- `conf/agent.yml.example`：完整的代理配置样板，囊括模型、节点、工作流、命名空间、存储和反思策略。
- `docs/`：官方文档源码，包含架构图、工作流、CLI、知识库等模块说明。

## 2. 运行入口

| 场景 | 文件 | 说明 |
| --- | --- | --- |
| CLI | `datus/cli/main.py` | 解析命令行，构建 `DatusCLI` 对话循环，可开启 Streamlit Web 前端。 |
| API | `datus/api/server.py` | 启动 FastAPI 服务，支持前台/后台进程、PID 文件和日志管理。 |
| SDK | `datus/agent/agent.py` | 面向程序化调用的主类 `Agent`，负责工作流初始化、执行与流式输出。 |

## 3. 核心工作流引擎

1. **Agent 初始化**（`datus/agent/agent.py`）  
   - 加载 `AgentConfig`，构建数据库管理器、工具与存储模块。  
   - 可根据命令行参数选择直接生成 workflow 或从 checkpoint 恢复。
2. **Workflow 生成**（`datus/agent/plan.py`）  
   - 读取内置 `workflow.yml` 或 `agent.yml` 中的自定义 plan，自动补齐起始节点。  
   - 支持串行、并行（`parallel`）、候选选择（`selection`）、子工作流（`subworkflow`）等结构。
3. **执行循环**（`Agent.run` / `run_stream`）  
   - 逐节点执行：`setup_input → start → execute → complete → update_context → evaluate_result → advance`。  
   - `evaluate_result`（`datus/agent/evaluate.py`）负责质量评估与上下文维护。  
   - 自动保存运行轨迹 (`trajectory_dir`) 与最终结果。
4. **流式动作历史**（`run_stream`）  
   - 每个节点执行会推送 `ActionHistory`，便于 CLI/Web UI 实时渲染进度。  
   - ActionRole/Status 定义见 `datus/schemas/action_history.py`。

## 4. 节点类型与职责

节点基类定义于 `datus/agent/node/node.py`，根据 `NodeType` 区分控制节点（start/reflect/parallel/selection/subworkflow 等）与行动节点。每个节点需实现 `setup_input` 与 `update_context`。常用节点包括：

| NodeType | 位置 | 功能概要 |
| --- | --- | --- |
| `schema_linking` | `node/schema_linking_node.py` | 解析任务，检索关联表/字段/样例值，写入上下文。 |
| `generate_sql` | `node/generate_sql_node.py` | 使用 LLM 生成 SQL，更新 `SQLContext` 与表元数据缓存。 |
| `execute_sql` | `node/execute_sql_node.py` | 通过 `DBManager` 执行 SQL，管理超时、错误和结构化结果。 |
| `reflect` | `node/reflect_node.py` | 反思与回合式修正，决定是否重试、走修复节点或结束。 |
| `reasoning` | `node/reason_sql_node.py` | 生成解释或多步推理链，常与 Parallel/Selection 联合使用。 |
| `doc_search` | `node/doc_search_node.py` | 基于向量检索文档上下文，支撑 SQL 生成。 |
| `generate_metrics` / `generate_semantic_model` | 对应节点 | 构建指标与语义模型草稿。 |
| `parallel` / `selection` | `node/parallel_node.py` / `node/selection_node.py` | 并发执行多个候选节点并选择最佳输出。 |
| `subworkflow` | `node/subworkflow_node.py` | 挂载自定义子流程，实现多阶段、多策略混合。 |

节点可使用 `Tool`、MCP server 及数据库连接（`datus/tools/db_tools`），输入/输出模型统一由 `datus/schemas` 定义。

## 5. Agentic Node 架构

`datus/agent/node/agentic_node.py` 引入面向对话/多轮计划的 Agentic Node：

- 不继承传统 `Node`，而是封装会话状态（SQLiteSession）、ActionHistory 列表与工具/MCP 集成。
- 支持计划模式：自动生成 TODO（`todo_write`/`todo_update`）、执行挂钩、重规划等流程。
- Prompt 通过 `prompt_manager` 动态渲染，可指定版本、多语言和工作空间路径。
- `chat_agentic_node.py`、`gen_sql_agentic_node.py`、`semantic_agentic_node.py` 等实现 CLI 中的高级对话体验。

## 6. 知识与存储层

- **Schema RAG**：`datus/storage/schema_metadata` 提供 `SchemaWithValueRAG`，结合数据库抽样、缓存与检索。
- **Metric RAG**：`datus/storage/metric` 负责指标 Embedding、成功案例聚合（`init_success_story_metrics`）。
- **文档与外部知识**：`storage/document` 与 `storage/ext_knowledge` 统一初始化，支持本地/云向量模型（详见 `storage/embedding_models.py`）。
- **SQL 历史 / 反馈**：`storage/sql_history`、`storage/feedback` 用于闭环学习。
- 所有存储路径固定在 `{agent.home}/data`，配置集中由 `AgentConfig` 加载，必要时通过 `init_*` 方法预热。

## 7. 工具与数据库连接

- `datus/tools/db_tools/db_manager.py` 封装多数据库连接池、逻辑命名空间、驱动能力（SQLite/DuckDB/Snowflake/StarRocks 等）。
- `agents` 包提供 `Tool`、`SQLiteSession`、MCP 客户端（由仓库依赖引入）。
- CLI 提供 `context`、`metadata` 等命令，调用 `datus/cli/*_commands.py` 中的工具方法。

## 8. 子代理（SubAgent）体系

- 配置模型 `SubAgentConfig` 位于 `datus/schemas/agent_models.py`，支持作用域上下文、工具白名单、规则和 Prompt 版本。
- `datus/storage/sub_agent_kb_bootstrap.py` 可按配置构建 Scoped KB（表、指标、SQL、外部知识等）。
- CLI 中 `sub_agent_commands.py` / `sub_agent_wizard.py` 提供交互式创建、导出与部署流程；Web 端以查询参数 `?subagent=` 调用。

## 9. 配置与 Prompt

- `conf/agent.yml.example` 展示完整配置项：模型分组、节点参数、工作流 plan、自定义反思策略、命名空间数据库、向量模型以及子工作流示例。
- 模型 `type` 新增 `dashscope`，可通过 `base_url=https://dashscope.aliyuncs.com/compatible-mode/v1` 与 `DASHSCOPE_API_KEY` 无缝连接阿里云 Dashscope OpenAI 兼容接口。
- Prompt 模板位于 `datus/prompts/`，由 `prompt_manager` 根据模板名 + 版本动态渲染，Agentic Node 会在运行期注入上下文变量（namespace、workspace 等）。
- 配置中可为不同 NodeType 指定模型、最大上下文、截断长度、反思策略等参数，实现细粒度控制。

## 10. CLI / API 能力

- CLI REPL（`datus/cli/repl.py`）集成历史记录、自动补全、命名空间切换、计划挂钩、流式输出。
- `action_history_display.py` & `generation_hooks.py` 将 Agentic Node 的 ActionHistory 与终端 UI 对接。
- Web 端（`datus/cli/web`）包装 CLI 使其可通过 Streamlit 对话。
- API 层（`datus/api/service.py`）提供聊天、执行、上下文管理等 REST 接口，并集成鉴权（`auth.py`）。

## 11. 典型数据流

1. 用户通过 CLI 输入自然语言任务，CLI 构造 `SqlTask`（`datus/schemas/node_models.py`）。
2. `Agent` 根据 plan 创建 `Workflow`，`BeginNode` 接收原始任务 → `SchemaLinkingNode` 检索元数据 → `GenerateSQLNode` 调用 LLM 产出 SQL。
3. `ExecuteSQLNode` 通过 `DBManager` 运行 SQL，结果写入上下文；若失败，`ReflectNode` 启动反思策略，可能触发重试或修复节点。
4. 最终 `OutputNode` 汇总结果 / 解释返回给 CLI 或 API 调用方；全程 ActionHistory 流式输出到终端或 Web。

## 12. 测试与基准

- `tests/`：包含 CLI 命令、工作流节点、配置解析等单元/集成测试。
- `benchmark/`：提供 BIRD、Spider2、semantic layer 等基准脚本与数据准备工具，`datus/utils/benchmark_utils.py` 实现指标计算、金标准生成。
- `build_scripts/` 与 `Makefile` 提供格式化、打包、发布等维护脚本。

## 13. 扩展指南

1. **添加新节点：**  
   - 在 `datus/agent/node/` 下创建实现，继承 `Node` 或 `AgenticNode`，实现 `setup_input`/`execute`/`update_context`。  
   - 在 `NodeType` 枚举中登记常量、描述与输入模型。  
   - 如涉及配置项，在 `conf/agent.yml` 中为节点指定模型或参数。
2. **自定义工作流：**  
   - 在 `conf/agent.yml` 的 `workflow` 段新增 plan；或直接修改 `datus/agent/workflow.yml`。  
   - 确认自定义节点在 `Node.new_instance` 中可被构造。
3. **集成新数据源：**  
   - 在 `datus/tools/db_tools` 添加对应 connector，配置 `namespace` 条目后即可在 CLI 中切换使用。
4. **扩展 RAG/Embedding：**  
   - `datus/storage/embedding_models.py` 支持本地模型与云模型切换。  
   - 如需新增嵌入服务，实现同名接口并在配置中调整 `storage` 段。

## 14. 相关文档与资源

- README.md、BUILD.md：工程概览、构建说明。  
- `docs/workflow/`、`docs/cli/`、`docs/knowledge_base/`：官方用户文档。  
- `requirements.txt`、`pyproject.toml`：依赖与打包信息。  
- Slack、官网、Quickstart 等链接见根目录 README。

通过以上结构，开发者可以快速定位 Agent 运行时的关键代码，理解标准 SQL 工作流与 Agentic 对话节点的实现，并依据业务需求扩展自定义能力。

## 15. 实战经验与注意事项

- **Python 版本兼容性**：Dashscope + Clickzetta 在 macOS 上使用 Python 3.12 时会遇到 pandas 编译、NumPy Accelerate 等兼容问题。实测选择 Python 3.11 并在 `datus/__init__.py` 中按需跳过 NumPy 的 macOS sanity check 可避免崩溃。
- **安全地管理凭证**：所有敏感信息（Dashscope/DeepSeek/Clickzetta）都通过 `.env` 注入，配置文件仅保留环境变量占位符；`.gitignore` 已忽略真实配置，防止意外提交。
- **Datus home 权限**：若默认的 `~/.datus` 无写权限，可在仓库内创建 `.datus_home` 并在启动命令前指定 `DATUS_HOME=$(pwd)/.datus_home`。
- **网络受限时的依赖安装**：国内或受限网络情况下，`pip` 会长时间回溯甚至失败。提前收集 wheel 到本地目录，再执行 `pip install --no-index --find-links=<dir>` 是更稳妥的方案。
- **Clickzetta schema 差异**：`information_schema.tables` 没有 `is_view`、`is_materialized_view`，需要改用 `table_type` 字段判断；列元数据同样没有 `ordinal_position`，改用 `column_name` 排序。
- **Python 3.11 语法兼容**：修复了多个 f-string 中的换行拼接写法，避免“expression part cannot include a backslash”；同时对 `typing.override` 引入 `typing_extensions` 兜底，保证 3.11 环境可用。
