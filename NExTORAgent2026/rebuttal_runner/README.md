# Rebuttal 实验入口

从 `NExTORAgent2026` 目录运行 `python -m rebuttal_runner`。使用同一个入口准备数据、生成共享数学输入、运行 Native/NED 成对实验、汇总和重评分。

## 适用范围

这是新的受控编码实验协议 `nedtree-paired-v1`，用于检验相同数学输入下 NED 结构信息的额外作用。它不自动复现投稿时的端到端配置，也不把历史输出改写为新实验。

- 两臂共享题面、数学输入、定义域、合法 API 提示、模型参数、求解时间和修复上限。Native 可合法使用顶层结果变量，也可自行分解。
- NED 使用已有确定性转换器；不支持或转换失败的题保留，并记录回退。不得事后只统计支持或成功的题。
- `--framework ours` 在编码失败后增加一次错误分析调用；`standard` 直接根据执行反馈修复。这里的名称是本入口的两种修复路径，不能直接作为原论文同名方法的复现。`--repairs 0` 时两者首次编码提示相同，不应重复运行并当成两种独立方法。
- `extract` 可记录一次抽取和一次建模检查；随后两臂复用该输入。`--repeats` 只重复冻结输入后的编码过程，不代表整条 pipeline 的独立重跑。
- CoE、OptiMUS、外部 OR-LLM-Agent 尚未接入。不能用自有管线重命名代替这些对照。

## 环境与模型

需要 Python 3.10+、OpenAI Python SDK、SymPy，以及执行端的 Gurobi Python 接口与可用许可证。当前本机验证环境为 Python 3.12.9、OpenAI SDK 1.78.0、Gurobi 12.0.2。API 密钥只从模型配置指定的环境变量读取，不写入配置。

`examples/model.example.json` 说明模型 ID、Base URL、超时及生成参数。接口必须显式指定；不自动更换模型、不做隐藏重试。返回模型 ID 必须在允许清单内，失败、空响应、token 和耗时都进入 `calls.jsonl`。供应商返回的模型名不是对上游身份的独立认证。记录供应商实际返回的 token，不伪造美元成本。

本轮模型安排：GPT-5 已通过最小接口检查；Kimi 已移出计划；DeepSeek 等待单独渠道，暂不填替代模型。

## 数据准备

```powershell
python -m rebuttal_runner prepare --source <原始文件.json> --dataset NEXTOR --format nora --scoring nextor_relative --output <候选清单.json>
python -m rebuttal_runner prepare --source <OptiBench.json> --dataset OptiBench --format optibench --scoring optibench_absolute --nonlinear-only --output <候选清单.json>
python -m rebuttal_runner inspect --manifest <候选清单.json>
```

准备过程保留原始题面、source_id、参考答案结构和来源信息，生成版本号；不会覆盖源文件。OptiBench 可依据源数据标签列出非线性候选，但这不是人工核验完成的名单。NORA 的模型预测类型不用于静默筛题。

正式运行前逐题确认：题面版本；所问变量及单位；答案是否为标量、列表或具名多输出；数学输入是否忠实；参考值精度；评分协议。核对后将 `review_status` 设为 `verified`，并在外部核对表保留依据。`ready/approved` 也是显式放行状态，不能由准备脚本自动赋予。

`--allow-unreviewed` 仅用于探索，结果带 `exploratory=true`。题面疑似含外部答案材料时，即使探索模式也先停止该题的模型调用；人工核实题内数据后再处理标记，不能直接删除真实题面信息。

清单的 `math_input` 只保留五个字段：`Parameters_List`、`Variables_List`、`Constraint_Table`、`Objective`、`Problem_Type`。`reference/scoring` 留在评分侧；模型只看到 `question/math_input/answer_spec`。字段隔离不能代替人工检查数学输入内是否混入参考解。

缺少数学输入时可运行：

```powershell
python -m rebuttal_runner extract --manifest <候选清单.json> --model-config <模型配置.json> --output <带数学输入清单.json>
```

默认抽取后再检查一次，`--no-refine` 只抽取。上游失败保留为失败题，不从分母删除。人工修正后的数学输入只能用于明确标注的编码诊断，不应冒充自动端到端结果。

## 成对运行与续跑

先用附带的一题解析数学例子验证接口、求解器与落盘：

```powershell
python -m rebuttal_runner run --manifest rebuttal_runner/examples/math_smoke.json --model-config rebuttal_runner/examples/model.example.json --output <新的输出目录> --repairs 0 --concurrency 1
```

正式实验显式固定参数：

```powershell
python -m rebuttal_runner run --manifest <已核对清单.json> --model-config <模型配置.json> --output <新的输出目录> --framework ours --repeats 1 --repairs 0 --concurrency 2 --solver-time-limit 60 --process-timeout 180 --solver-threads 1
```

`--repairs N` 最多执行 N+1 次生成程序。只有程序执行或答案解析失败才触发修复；得到可解析的答案就停止，不用参考答案是否正确来决定重试。错误分析与额外生成均单独计费和记录。

生成程序在独立子进程运行；直接 `optimize()` 调用会注入 TimeLimit/Threads 并记录求解状态。程序级超时另行控制。答案匹配分数不等价于模型语义正确性；无解、跳过求解或错误建模等仍需结合 `solver_status`、代码及人工检查分析。

续跑使用完全相同的命令并添加 `--resume`。已落盘的成功和失败均保留。协议、清单或参数改变时拒绝复用目录。已开始但未完整落盘的单元不会自动再次收费，需先核查后另建输出目录。

## 汇总与阈值敏感性

```powershell
python -m rebuttal_runner summary --run-dir <输出目录> --output <新汇总.json>
python -m rebuttal_runner rescore --run-dir <输出目录> --thresholds 0.01,0.001,0.0001 --output <新重评分.jsonl>
```

重评分只读取固定预测，不调用模型或求解器，不改变原始结果。本轮确认后的工作副本统一使用 `objective_relative`：只比较标量目标值，非零参考采用 `abs(pred-ref)/abs(ref) <= tolerance`，仅零参考采用绝对容差 `1e-5`。三档为 `0.01/0.001/0.0001`；内部保持完整数值精度，展示舍入不参与评分。标量 `answer_spec.fields` 设为 `["objective_value"]` 时，公共编码提示明确要求返回优化目标，不改变原题数学条件。

NEXTOR 弹簧仅使用 `nextor_spring_legacy`，限定 `case_id == "NEXTOR:11"`、参考结构 `[目标值, 9.0]`；目标值或圈数任一命中均可通过，并在 `matched_references` 记录命中属性。它不推广到其他列表题。重评分保留清单中每题的协议及特例。

历史 `nextor_relative`（零值精确匹配）、`optibench_absolute`、`optmath_normalized` 与完整结构比较仍保留原行为；旧清单不会被静默转换成新协议。历史代码中的末位绝对容差或任意项匹配也不等于本轮的三档相对误差。

`--exclude-case <case_id>` 生成明确标记的排除分析；原结果不删除。原全集和排除后结果应同时报告。参考值舍入或单位有疑问时，先解决对应问题，再解释严格阈值结果。

汇总区分计划数、完成数、缺失数、失败与正确答案；配对必须满足相同题、重复编号、输入 hash 和版本。按 source_id 做聚类 bootstrap，题目极少时区间不宜作为稳健性证据。费用分为实际唯一 API 调用和各方法归因成本；缺少 usage 时标为未知。历史输出需先核对来源并映射到当前记录格式，不能直接拼接进统计。

每次运行保留 `protocol.json`、`manifest.json`、`calls.jsonl`、逐臂 `result.json`、输入、提示、模型响应、原始/执行代码、stdout/stderr，以及可重建的 `summary.json`。日志可能包含研究题面，按项目的数据权限保存。

## 验证

```powershell
python -m unittest discover -s tests -p 'test_rebuttal_*.py' -v
```

离线测试覆盖评分边界、输出结构、数据隔离、模型失败记账、真实子进程行为、成对输入、修复上限和续跑。小样例运行只证明工具链连通，不是补实验结论。
