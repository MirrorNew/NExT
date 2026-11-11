# Rebuttal 统一实验入口

在 `NExTORAgent2026` 目录执行 `python -m rebuttal_runner`。当前协议为 `nedtree-integrated-v3`；准备、执行、汇总、重评分共用一个入口。已完成最小样例和真实十题的工程预检；没有运行正式数据集补实验。v3 修复预检暴露的表示/输出契约问题，旧 v2 产物保留且不可混用目录。

## 实验层次必须显式指定

| `--level` | 输入及执行边界 | 重复次数含义 |
| --- | --- | --- |
| `math` | 作者构造并检查的数学实例；固定 `math_input` 后比较编码路径 | 重复编码，不能解释成端到端重复 |
| `coding` | 数据集题面及已冻结的抽取/数学输入；可复用逐题对齐后的历史缓存 | 重复编码，上游输入保持不变 |
| `end_to_end` | 从题面自动开始；忽略清单中的历史数学缓存 | 每次重新执行整个自动工作流 |

`math` 与 `coding` 共用执行器，但在协议、记录和汇总中分开。`ours` 的端到端 Native/NED 两臂在同一题同一重复内共享上游抽取和建模；下一重复重新生成。跨层次、不同输入 hash 或不同数据版本不会组成有效配对。

## 方法与 NED 的边界

| `--framework` | 端到端入口 | 固定输入的编码入口 |
| --- | --- | --- |
| `ours` | 原 Parameter/Factor 抽取器、ModelingAgent、CodingAgent、RepairAgent | 原 CodingAgent；失败后原 RepairAgent |
| `standard` | 题面直接生成程序 | 固定数学模型直接生成程序；按执行反馈修复 |
| `coe` | 本地 CoE 的 conductor→experts→reducer | 原 ProgrammingExpert，以冻结数学输入作为已给定建模意见 |
| `optimus` | 本地 OptiMUS 的抽取、建模、代码片段和程序组装 | 冻结输入映射到其代码阶段；不能无损映射时明确失败 |
| `or_llm_agent` | 本地 OR-LLM-Agent 的建模、编码及修复流程 | 将原建模返回点替换为冻结输入，保留原编码及修复流程 |

`ours` 的上游是 **5 次真实 completion**：参数分析/格式化、要素分析/格式化、ModelingAgent。NED 保留 **AuxiliaryModelAgent 原有的一次 completion**，在同一次响应内增加结构字段，再组织为树的辅助定义、依赖、外层关系和定义域；没有第二次 LLM 分解，也不再调用独立确定性分解器重新生成方案。首次编码一次；ours 每次修复包含原 RepairAgent 的分析和代码两次 completion。

Auxiliary 的这一次请求单独设置 `response_format={"type":"json_object"}`；其他阶段不继承该调用级设置。适配层保留原 Agent 的建模要求，将旧输出格式替换为唯一的合并 JSON 格式。有效请求参数写入调用账本，协议的 `stage_request_parameters` 记录阶段参数约定（复用缓存时不代表本次实际发出该请求）。JSON 模式不替代结构、来源、关系和依赖校验；服务端拒绝参数、截断、无效 JSON 或无效结构均保留失败，不增加一次格式修复或重新分解调用。

同一次输入的 `Context.source_expressions` 列出来源 ID 与完整原式，供 Auxiliary 复制 `source_expression`，包括目标的方向、名称及赋值前缀；只有 `linearized_form` 去掉目标标签。校验只允许既定的排版/符号归一化（如空白、`^` 与 `**`），不猜测一般代数等价；树中最终保存冻结源原串。参数列表的下标仅在源已明确完整的零基或一基集合、且与数值列表长度一致时允许；标量名不能放行任意下标。该过程不重新分解。

现有参数/要素提取调用共同要求 ASCII 声明、显式乘法、有限索引展开和每行一个原子关系，参数值与域条件分开保存；要素提取复用本题已经提取的参数名称。Auxiliary 支持单参数 `log2`/`log10` 表达，并提示用自然对数换底进行编码。不会因格式要求增加提取或分解调用，也不会把不支持的多关系或未知依赖直接放行。

Native 不接收辅助意见，允许合法顶层结果变量以及自行引入辅助变量。NED 接收该一次 Auxiliary 输出。两臂共享题面、数学输入、输出对象、合法 NLExpr/resultant API 指导、模型参数和求解限制。格式不合法、来源表达式或依赖检查失败的 Auxiliary 输出留作审计，编码退回原数学输入；失败题不删除。结构验证不等于数学等价性证明。

固定的结构化 `math_input` 是本入口的数学输入依据，ModelingAgent 的原始 `math_model` 文本作为解释另行保存；提示不允许它或辅助意见覆盖既定条件。这是当前实现的明确接口约定，不声称两份表示已自动证明一致。

外部基线端到端仅允许 `--arms native`，保留各自上游流程。它们的 Native/NED 迁移对照使用 `coding` 或 `math` 层次，不把我们的上游套在基线前面并称作原基线端到端。不同方法的原生调用次数可以不同；统一限制和真实成本都记录。

## 模型、基线来源与统一限制

- GPT-5 配置：`rebuttal_runner/examples/model.example.json`，密钥变量 `OPENAI_API_KEY`。
- DeepSeek Flash 配置：`rebuttal_runner/examples/model.deepseek-flash.json`，官方 ID `deepseek-flash`，Base URL `https://api.deepseek.com`，密钥变量 `DEEPSEEK_API_KEY`。ID 根据[官方接口文档](https://api-docs.deepseek.com/api/list-models/)核对，并已在线返回同名模型。
- Kimi 不在本轮计划中。密钥不写入配置，不自动更换模型，不隐藏重试；允许的返回模型 ID 明确列在配置内。
- GPT-5 示例保留 16384 输出 token 上限、300 秒请求超时。DeepSeek 的复杂题预检出现截断，示例上限改为 65536、超时 600 秒，保留 `reasoning_effort=high`；同一模型的所有框架使用同一配置。[官方参数文档](https://api-docs.deepseek.com/api/create-chat-completion/)列明 thinking/high 默认输出预算为 64K。上限不是实际消耗，也不保证复杂题足够；正式运行前仍须冻结预算。截断保留为 `MODEL_OUTPUT_TRUNCATED`，不隐藏重试；usage 缺失记为未知，不估造费用。
- 请求失败保留原 error，同时记录脱敏的 error_type、error_code 与有限长度的 cause_chain，区分连接、超时、HTTP、截断等；SDK 重试仍关闭，每次 Auxiliary 最多一次请求。

外部基线使用 `--baseline-root D:\LLMProject`，分别读取 `Chain-of-Experts-main`、`OptiMUS-main`、`or_llm_agent-main`。适配器仅加载实际所需函数、类和提示，不执行含旧密钥/客户端的模块顶层代码，不改原始基线项目。来源按实际 Python 文件 SHA256 固定；没有 Git 元数据的目录不捏造 commit。

适配的公共改动：调用均经过同一模型配置和日志；生成程序均经过同一执行器；使用目标值输出规范和合法非线性接口提示，完整程序必须实际调用自己定义的入口函数。CoE 保留当前源快照的 3 轮专家协作、单次 trial、关闭 reflection 的有效行为；其内部旧 `max_tokens`/temperature 不覆盖统一模型配置。OptiMUS 使用源代码现有无检索分支和 `main.py` 的错误检查设置，保留代码片段/组装协议及 `=====` 修复响应格式；OR-LLM-Agent 的重试受统一上限约束。详细执行阶段、路径和失败原因保存在 `baseline_trace`。

公共契约及 OptiMUS 修复提示明确保留符号、关系方向、自由变量域和原题目标单位；目标变换后须按原表达式重算答案，不能用未经等价验证的 PWL 或 McCormick 松弛替代精确模型。所有流程只解析唯一 `FinalAnswer` JSON，不再从普通 solver 日志猜测目标值。`output_failure` 区分执行失败、未验证 Gurobi、无解、缺少/非法标记和答案结构错误；数学数值评分规则不变。

OptiMUS 的原组装器确定性输出 `model.ObjVal`。适配层仅将其唯一已知输出尾段改为统一的标量 `FinalAnswer`，保持模型代码不变；相应代码提示要求优化目标自身保留原题数值与单位。未知尾段或非标量目标答案明确报不支持，不浪费一次 LLM 修复来改打印格式。自由文本日志仍不可作为评分答案。

OptiMUS 固定输入适配支持明确的标量 binary/integer/continuous 类型、数值上下界和明确无界域，兼容旧 `Range ` 字段；只在 coding/math 的原变量生成器中显式传入 lb/ub，避免原默认下界 0 改变数学模型。数组形状、有限开边界、复合或无法确认的范围返回 `unsupported_coding_input`，保留在分母，且在 Auxiliary 之前拒绝，不自动补造条件。端到端使用其原建模结构。

需要 Python 3.10+、OpenAI SDK、SymPy、NumPy、gurobipy；CoE 还需其 LangChain 依赖，已在 `rebuttal_runner/requirements-baselines.txt` 单独列出本机验证版本，没有改现有根依赖文件。本机验证解释器为 `E:\my_evns\py312_torch28\python.exe`。正式求解仍需合适的 Gurobi 许可证和版本。

## 数据准备与离线检查

当前工作副本在 `D:\LLM-CodexProject\NEDTree-for-ICLR\rebuttal\20261110\dataset_repair_v1`：

| 数据集 | 入口 | 可运行题数 |
| --- | --- | ---: |
| NEXTOR | `nextor/manifest.json` | 38 |
| OptiBench | `optibench/manifest.json` | 183 |
| OptMATH | `optmath/manifest_runnable.json` | 34，明确排除 137 |
| ABC | `observation/manifest.json` | 300，包含 ABC110 统一输入说明 |

原始数据不覆盖。`approved` 表示按作者确认策略完成入口准备，不意味着全部来源答案都被独立证明。原始审计清单和排除理由保留；只有 `excluded_*` 数据状态离开计划分母，方法不支持、生成/求解失败仍计入。

```powershell
python -m rebuttal_runner inspect --manifest <清单.json>
python -m rebuttal_runner plan --manifest <清单.json> --model-config rebuttal_runner/examples/model.deepseek-flash.json --level end_to_end --framework ours --output <新的计划.json>
python -m rebuttal_runner plan --manifest <清单.json> --model-config rebuttal_runner/examples/model.example.json --level end_to_end --framework coe --baseline-root D:\LLMProject --output <新的计划.json>
```

`plan` 不调用模型或求解器，检查数据、层次、源码、依赖、密钥变量是否存在以及调用硬上限；这不是线上可用性保证。`hard_call_ceiling` 是停止上限，不是预计费用。`--max-calls` 限制每臂全部请求，包括 NED 的 Auxiliary 和修复；`--max-upstream-calls` 单独限制 ours 的共享上游。失败请求也占用预算。

因此 `max-calls` 是总调用预算，NED 的一次 Auxiliary 也占额度；不是两臂拥有相同数量的纯编码调用。主要公平性控制还包括相同初始生成次数和 `repairs` 上限；正式实验应给足预算，若预算触顶必须单列报告。

模型可见字段仅为题面、数学输入/建模文本和 `answer_spec`，参考答案、评分和审计字段留在执行之后。隔离字段不能替代对输入内容的语义检查。

如需重用缓存，先将历史记录映射为清单，再做严格对齐：

```powershell
python -m rebuttal_runner attach-inputs --manifest <当前清单.json> --cache-manifest <历史输入清单.json> --output <新的合并清单.json>
```

仅同数据集、唯一 source_id、完全相同题面的合法数学输入可复用；不复制参考答案、评分和放行状态，旧 Auxiliary 必须匹配输入和题面 hash。历史模型的抽取仍标为历史缓存，不能成为新模型端到端成绩。

需要新生成输入时，使用 `extract`，它运行原上游 5 次调用和一次 Auxiliary；本轮未执行该命令：

```powershell
python -m rebuttal_runner extract --manifest <清单.json> --model-config <配置.json> --output <新的冻结输入清单.json>
```

上游输出立即保存，并逐题更新 `.partial.json`。中断后先检查已落盘记录，通过 `attach-inputs` 复用完成输入；不自动重新收费。不支持旧 `--no-refine` 省略 ModelingAgent 的模式。

## 执行、恢复与记录

一个独立于正式数据集的端到端连通样例保存在 `examples/end_to_end_smoke.json`，其解析最优值为 2。设置模型配置指定的密钥环境变量后，可用下列命令检查完整方法。原始基线换为对应 `--framework`，加 `--arms native --baseline-root D:\LLMProject`；每次使用新的输出目录。单题成功仅表示接口和流程可执行。

```powershell
python -m rebuttal_runner run --manifest rebuttal_runner/examples/end_to_end_smoke.json --model-config rebuttal_runner/examples/model.example.json --level end_to_end --framework ours --arms native ned --output <新的smoke目录> --repairs 1 --max-calls 32 --max-upstream-calls 8 --solver-time-limit 30 --process-timeout 60 --solver-threads 1
```

验证非空树可改用 `examples/nonempty_tree_smoke.json`：最小化 `exp(x^2)+exp(y^2)`，`x,y∈[0.5,2]`、`x*y>=1`，解析目标为 `2e`。清单不含数学输入或 Auxiliary 缓存；沿用 `--level end_to_end`，每次从题面执行全流程。验收应同时要求本次 Auxiliary 调用恰为一次、树非空且校验通过、生成代码实际使用定义并保留乘积约束，不能仅凭目标值正确判定树路径成功。

以下为后续正式实验的命令模板，本轮没有执行：

```powershell
python -m rebuttal_runner run --manifest <冻结输入清单.json> --model-config <配置.json> --level coding --framework ours --output <新的输出目录> --repairs 0 --repeats 1 --concurrency 1
python -m rebuttal_runner run --manifest <数据清单.json> --model-config <配置.json> --level end_to_end --framework or_llm_agent --baseline-root D:\LLMProject --output <新的输出目录> --repairs 1 --max-calls 64
```

`--repairs N` 最多允许初始执行之后 N 次修复；只根据程序执行和输出解析失败触发，不根据参考答案决定是否重试。原方法无对应修复环节时不额外伪造。直接 `optimize()` 调用注入 TimeLimit/Threads，另外限制进程时间；读取答案前要求求解限制成功注入、`SolCount > 0`。代码和求解状态保存，答案匹配不等于建模语义正确。

续跑使用完全相同命令加 `--resume`。已完成的成功/失败均保留；协议、数据、代码或配置变化拒绝复用目录。开始但未完整落盘的单元停止并要求检查已有记录，不自动重新请求。每次调用保存提示、响应/错误、模型 ID、usage 和耗时；每次执行保存原代码、执行代码、stdout/stderr。汇总只读取 Native/NED 结果，不把 Auxiliary 日志算为实验题。

输出目录同时仅允许一个进程持有 `.run.lock`。正常退出释放锁；硬中断遗留锁不自动抢占，先核查 PID 和账本再恢复，避免并行 resume 重复请求。

## 汇总及 1% / 0.1% / 0.01% 重评分

```powershell
python -m rebuttal_runner summary --run-dir <输出目录> --output <新的汇总.json>
python -m rebuttal_runner rescore --run-dir <输出目录> --thresholds 0.01,0.001,0.0001 --output <新的重评分.jsonl>
```

重新评分保留：同一批固定预测，不调用模型或求解器，不改原结果。本轮 `objective_relative` 仅比较标量目标值，非零参考用 `abs(pred-ref)/abs(ref) <= tolerance`；零参考使用 `1e-5` 绝对容差。内部完整精度参与评分，显示舍入不参与。NEXTOR:11 单独保留作者确认的 `nextor_spring_legacy`（目标值或圈数命中），不推广至其他列表题。历史其他协议行为不被静默替换。

汇总列出计划/完成/缺失、失败及准确率，按源题聚类 bootstrap。共享上游成本可分别归因给两臂，同时单列唯一调用量；执行耗时包含所有修复尝试。`current_run_ledger` 另外统计本次账本中尚未落盘为实验结果的调用，避免漏报已记录费用。缺失 usage 标为未知；调用时延之和不是并发批次墙钟时间。

求解时间单独从每次 `optimize()` 返回后的 Gurobi `Runtime` 采集，保存于 `solver_status.optimize_calls`，并累计已知值和缺失次数；程序执行的 `elapsed_seconds` 保持原含义。汇总中的 `execution.solver_runtime` 与程序执行时间分开。求解异常不沿用先前 Runtime；进程被强制终止、旧日志没有 Runtime 或历史成功记录缺执行证据时，完整求解时间为未知，不填零。明确未进入求解的上游失败保留零次求解。

## 离线验证

```powershell
python -m unittest discover -s tests -p 'test_rebuttal_*.py' -v
```

离线测试使用假模型回复，覆盖真实 agent 提示和本地基线代码路径、单次 Auxiliary、字段隔离、调用/修复预算、异常和恢复、配对分母、阈值及执行器；不能替代后续真实 API、生成质量和目标求解规模检查。
