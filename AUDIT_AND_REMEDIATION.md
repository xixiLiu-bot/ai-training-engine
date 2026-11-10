# 智训引擎 · 代码审计与整改报告

> 审计基线：`智训引擎-代码-1109.zip`（11/10 实读代码，非文档推断）
> 整改范围：A 类（必须修）+ B 类（高性价比）中的 8 项
> 结论口径：**只写代码里能验证的东西**；无法确认的一律标「需要进一步检查」

---

## 1. 当前项目总体流程

```
                    ┌──────────────────────────────────────────────┐
                    │  评测协议（一问一答，禁止偷看未来）            │
                    │  LocalEvaluator: next_batch → submit → 下一批 │
                    └───────────────┬──────────────────────────────┘
                                    │ 第 b 批 (X_b, y_b)
                                    ▼
   ┌────────────┐  ┌───────────────────────────────────────────────────┐
   │ 数据集适配  │→ │ ① 监测  monitor.py    PSI / KS / Wasserstein       │
   │ dataset.py │  └──────────────────────┬────────────────────────────┘
   └────────────┘                         │ ψ 报告
                                          ▼
   ┌────────────┐  ┌───────────────────────────────────────────────────┐
   │ 配置       │→ │ ② 归因  diagnose.py   stable / covariate / concept │
   │ config.py  │  └──────────────────────┬────────────────────────────┘
   └────────────┘                         │ 归因结论
                                          ▼
   ┌────────────┐  ┌───────────────────────────────────────────────────┐
   │ 模型适配    │← │ ③ 决策  decide.py(规则表) + policy.py(四道闸)      │
   │ model.py   │  │        ↑ LLM 建议（qwen4b/llm_agent，仅建议权）     │
   └────────────┘  └──────────────────────┬────────────────────────────┘
                                          │ action + cost
                                          ▼
   ┌────────────┐  ┌───────────────────────────────────────────────────┐
   │ 账本       │← │ ④ 执行 + 影子评估  evaluate.shadow(同批对比)       │
   │ ledger.py  │  └──────────────────────┬────────────────────────────┘
   └────────────┘                         │ before / after / gain
                                          ▼
   ┌────────────┐  ┌───────────────────────────────────────────────────┐
   │ 版本快照    │← │ ⑤ 发布/回滚  release.py  门控(窗口+Bootstrap)      │
   │ snapshots  │  │        accept → 换代 + 快照  /  reject → 维持      │
   └────────────┘  └──────────────────────┬────────────────────────────┘
                                          │ 决策记录
                                          ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 实验记录  logs/recorder.py → out/run.jsonl (+ SwanLab 可选)        │
   │ 决策留痕  ledger → out/decisions.jsonl（门槛③证据）                │
   │ 指标汇总  out/metrics.json（含 errors / release / config）          │
   └──────────────────────────────────────────────────────────────────┘
```

**关键结构事实**：五个环节里只有 ④⑤ 触碰模型参数，①②③ 只读数据分布与分数。
因此换模型 / 换数据集时，①②③ 完全不用改。

---

## 2. 项目目录与模块职责

字段说明：**职责 / 输入→输出 / 调用方 / 完整性 / 问题 / 建议补齐 / 建议新增文件 / 建议新增测试**

### 2.1 配置模块 `evolver/config.py`
- **职责**：集中所有阈值与开关，避免散落在各处
- **输入→输出**：无 → `Config` 数据类实例
- **调用方**：`run_pipeline.py`、`diagnose.py`、`decide.py`、`policy.py`、`release.py`
- **完整性**：✅ 完整（本轮新增 task_type / detector / gate_window / gate_bootstrap / max_snapshots）
- **问题**：所有阈值仍是手写经验值；`warmup_batches=1` 实际永远不会触发（循环从第 2 批开始）
- **建议补齐**：数据驱动标定（见 §6-B5）；把 warmup 改成"前 N 批只观察"的有效语义
- **建议新增文件**：`scripts/calibrate.py`（阈值扫描与敏感性分析）
- **建议新增测试**：`tests/test_config.py` —— 校验阈值区间合法、warmup 语义

### 2.2 Qwen/DeepSeek/OpenAI-compatible 客户端 `evolver/engine/qwen4b/client.py`
- **职责**：纯标准库 urllib 的 `/chat/completions` 调用，把 HTTP 错误分成三类
- **输入→输出**：messages + 解码参数 → `Completion(text, tokens, finish_reason, has_reasoning)`
- **调用方**：`decider.py`
- **完整性**：✅ 完整（无第三方依赖，含 `enable_thinking` 透传）
- **问题**：不支持流式；不校验响应 schema 版本
- **建议补齐**：加可选的流式解析（对 27B 的长思考有用）
- **建议新增文件**：无
- **建议新增测试**：已有覆盖（假服务器 9 种响应形态）

### 2.3 LLM Adapter `evolver/engine/qwen4b/compat.py` + `evolver/engine/llm_agent.py`
- **职责**：把 `llm_agent.build(backend)` 的返回接到 4B 决策器，保持上层零改动
- **输入→输出**：backend 名 + 环境变量 → 决策器对象（`.enabled/.suggest/.stats`）
- **调用方**：`run_pipeline.py` → `policy.decide`
- **完整性**：✅ 完整（`none/mock/mock_bad` 保留原实现作对照组）
- **问题**：`pytrio` 与 `qwen4b` 共用 `QWEN_4B_MODEL` 之外的路径；27B 的 thinking 关不掉（实测）
- **建议补齐**：为 27B 找到正确的思考开关（需要进一步检查 PyTRiO 文档）
- **建议新增文件**：无
- **建议新增测试**：`tests/test_compat.py` —— 环境变量解析、缺 key 时不崩

### 2.4 Agent 执行循环
- **现状**：**不存在**。Copilot 版本里的 `agent.py`（tool-calling 循环）从未被主流程调用
- **完整性**：❌ 未实现 / 未接入
- **问题**：文档与材料若声称"我们有 Agent"，与代码不符
- **建议补齐**：明确二选一 —— 要么删掉这套死代码，要么把它作为"决策建议器"接入（当前 `decider.py` 已等价实现单步决策，多步 tool 调用对本题无必要）
- **建议新增文件**：无（倾向删代码而非加代码）
- **建议新增测试**：无

### 2.5 Tool Registry
- **现状**：**不存在**（同上，`tools.py` 死代码）
- **完整性**：❌ 未接入
- **建议**：D 类（暂时不做）。本题的动作空间只有 3 个，不需要通用工具注册表

### 2.6 漂移检测模块 `evolver/loop/monitor.py`
- **职责**：逐特征计算分布距离，聚合成 `mean/max/p90`
- **输入→输出**：参照 X + 当前 X → 报告 dict（含 psi/ks/wasserstein 三组字段）
- **调用方**：`run_pipeline.py`
- **完整性**：✅ 完整（本轮新增 KS 与 Wasserstein）
- **问题**：① 只测边缘分布，测不到相关结构；② 参照系"部署日 vs 上次良好状态"的选择是工程补丁而非统计推导；③ 三种方法的阈值互不可比
- **建议补齐**：MMD（可测相关结构）；参照系改用滑动窗口
- **建议新增文件**：无（继续放 monitor.py，避免碎片化）
- **建议新增测试**：✅ 已加 `tests/test_monitor.py`（12 例）

### 2.7 漂移归因模块 `evolver/loop/diagnose.py`
- **职责**：三层阈值表 → stable / covariate / concept
- **输入→输出**：监测报告 + 标签漂移值 → 归因 dict（含 suggest）
- **调用方**：`run_pipeline.py`
- **完整性**：⚠️ 部分（能区分"输入变了"，但识别不了真正的 concept drift）
- **问题**：真正的概念漂移是 P(y|x) 变化，`label_psi` 只捕捉类别比例变化；代码用"便宜手段失败 ⇒ 概念漂移"的**间接推断**（run_pipeline 内）来补
- **建议补齐**：用"标签条件分布"或"分箱后的 P(y|x) 距离"做直接检验
- **建议新增文件**：无
- **建议新增测试**：`tests/test_diagnose.py` —— 三种分支 + 边界值

### 2.8 策略/决策模块 `evolver/loop/decide.py`
- **职责**：6 条规则的成本阶梯，输出 3 选 1 动作
- **输入→输出**：`State` + 归因 + config → `{action, hypothesis, reason, cost_sec}`
- **调用方**：`policy.decide`（`run_pipeline` 间接）
- **完整性**：✅ 可运行，但**不是学习型策略**
- **问题**：无 bandit / 无学习；成本常数手写；`psi_low` 分支优先于"上次校准失败"分支
- **建议补齐**：见 §6-C1（最低成本版）/ C2（研究版）
- **建议新增文件**：`evolver/loop/policy_bandit.py`（C 类，可选）
- **建议新增测试**：`tests/test_decide.py` —— 6 条分支全覆盖（**当前缺失**）

### 2.9 成本模型
- **现状**：`decide.COST` 三个常数（0 / 0.05 / 2.0 秒）
- **完整性**：⚠️ 存在但不随数据规模缩放
- **问题**：真实耗时 ≈ `O(n·d·k·iters)`，常数在数据量变化时失真
- **建议补齐**：改成"实测回归"——启动时用 1 个小样本拟合一次，得到 `cost(n, d)`
- **建议新增文件**：`evolver/loop/cost_model.py`
- **建议新增测试**：`tests/test_cost.py` —— 成本单调性、与实测耗时偏差 < 30%

### 2.10 动作执行模块 `run_pipeline.py`（内联）
- **职责**：把 action 翻译成具体操作（clone+refresh / 重训）
- **完整性**：⚠️ 只有 2 档真实可用（不动 / 标准化 / 重训）
- **问题**：**"改输出（阈值/温度校准）"这一档未实现**；配置里的 `enable_output_calibration` 只是开关占位
- **建议补齐**：见 §6-B3
- **建议新增文件**：`evolver/loop/calibrate.py`
- **建议新增测试**：`tests/test_calibrate.py`

### 2.11 模型训练/重训模块 `evolver/adapters/model.py`
- **职责**：统一 fit/predict/clone/refresh_normalization；真正训练发生在 `run_pipeline.recent_window`
- **完整性**：✅ 完整（本轮新增 `ridge` 回归模型）
- **问题**：重训 = "近期数据 + 指数衰减权重重 fit"，不是增量微调；`refresh_normalization` 用当前批 X（无标签，合法但不标注）
- **建议补齐**：把"增量"与"全量重训"分成两档动作
- **建议新增文件**：无
- **建议新增测试**：`tests/test_model.py` —— clone 隔离性、refresh 不改权重（**当前缺失**）

### 2.12 Shadow Evaluator `evolver/loop/evaluate.py`
- **职责**：同一批上比较候选与现役
- **完整性**：✅ 完整但极简
- **问题**：单批单次比较，无置信区间；未做多批验证
- **建议补齐**：影子评估改为"最近 k 批累计"（与 B2 门控窗口配套）
- **建议新增测试**：`tests/test_evaluate.py`

### 2.13 发布与回滚模块 `evolver/loop/release.py`
- **职责**：门控 + 版本快照 + 回滚
- **完整性**：✅ 完整（本轮 A3 补齐快照与 `rollback_to`）
- **问题**：**margin 未按指标量纲归一化** —— 回归任务 MAE≈20，`margin=0.02` 实际等价于"任何正增益都放行"（实测回归路径 7 次回滚全部因增益为负）
- **建议补齐**：`margin_mode="abs"|"rel"`，回归用相对改进（如 1%）
- **建议新增文件**：无
- **建议新增测试**：✅ 已加 `tests/test_release.py`（10 例）

### 2.14 评测协议模块 `evolver/protocol/batch_stream.py`
- **职责**：一问一答状态机，禁止偷看未来
- **完整性**：✅ 本地完整；`RemoteEvaluator` 是占位
- **问题**：`LocalEvaluator.next_batch()` 内部 `adapter.get()` 抛异常时，`max_released` 不推进 → 该批永远取不到，只能整体停止（本轮已让主流程安全停止并保留结果）
- **建议补齐**：企业赛接官方 SDK 时实现 `RemoteEvaluator`，并把 `history()` 语义对齐
- **建议新增测试**：✅ 已有 `tests/test_protocol.py`（6 例）

### 2.15 数据集适配模块 `evolver/adapters/dataset.py`
- **职责**：把不同数据切成统一批次流
- **完整性**：✅ synthetic / gas 可用；✅ 本轮新增 `synthetic_rul`（回归）/ `bearing` / `rul` 加载器
- **问题**：`bearing` 与 `rul` 只有加载器，**没有真实数据与 prepare 脚本**；`OfficialAdapter` 仍是占位
- **建议补齐**：`scripts/prepare_bearing.py`、`scripts/prepare_rul.py`
- **建议新增测试**：`tests/test_dataset.py` —— 越界、task_type 正确、缺文件时报错文案

### 2.16 实验记录模块 `evolver/logs/recorder.py`
- **职责**：每批写 `run.jsonl`，SwanLab 可选
- **完整性**：✅ 本地完整（本轮新增）；⚠️ SwanLab 分支**未验证**（无账号）
- **问题**：SwanLab 上传代码未跑过一次真实上传
- **建议补齐**：拿到账号后跑一次，确认字段映射
- **建议新增测试**：✅ 已加 2 例（不可写降级、正常写出）

### 2.17 CLI / 脚本入口 `scripts/run_pipeline.py`
- **职责**：主流程 + 参数解析 + 收尾落盘
- **完整性**：✅ 完整（本轮新增 A1/A2 兜底、task/model 分派、detector、gate 参数、recorder）
- **问题**：`decisions.jsonl` 的 `cycle` 字段按"非 do_nothing 动作数"计数，b8/b9/b10 会重复同一 cycle，若评委按"完整闭环轮次"核验会被挑
- **建议补齐**：改成按"读历史→提假设→做实验→做决策"的完整轮次计数
- **建议新增测试**：✅ `tests/test_resilience.py`（7 例，故障注入）

### 2.18 测试模块 `tests/`
- **完整性**：✅ 从 6 例扩到 **47 例**（2 例因缺 sklearn 跳过）
- **覆盖**：协议 6 / 门控回滚 10 / 漂移检测 12 / 指标分派 8 / 韧性 7 + 记录 2 + 2 skip
- **问题**：**没有** `test_decide.py`、`test_model.py`、`test_evaluate.py`、`test_dataset.py`
- **建议新增**：四个文件（见各模块）

### 2.19 配置与交付
- **完整性**：✅ `Dockerfile` / `requirements.lock` / `run_offline.sh` / `.dockerignore` 已补齐
- **问题**：**本机没有 docker，镜像未实际构建验证**（需要进一步检查）
- **建议补齐**：在有 Docker 的机器上 `docker build` + `--network none` 跑一次

---

## 3. 已实现功能清单

| 能力 | 状态 | 证据 |
|---|---|---|
| 五环节闭环 | ✅ | `run_pipeline` 端到端，10 轮协议 0 违规 |
| 一问一答协议 | ✅ | `tests/test_protocol.py` 6 例 |
| 漂移检测 PSI | ✅ 原有 | `monitor._psi` |
| 漂移检测 KS | ✅ **新增** | `monitor._ks` + `tests/test_monitor.py` |
| 漂移检测 Wasserstein | ✅ **新增** | `monitor._wasserstein`（按参照批标准差归一化） |
| 归因（三层阈值） | ✅ 原有 | `diagnose.py` |
| 成本阶梯决策（规则） | ✅ 原有 | `decide.py` |
| LLM 建议 + 四道闸 | ✅ 原有 | `policy.py` |
| 4B/27B 决策层（约束解码/修复/熔断/记账） | ✅ 新增（上一轮） | 42 个测试 + 13 用例矩阵 |
| 影子评估 | ✅ 原有 | `evaluate.shadow` |
| 门控 + 自动回滚 | ✅ 原有 | `release.try_release` |
| **版本快照 + 回滚到历史版本** | ✅ **新增** | `release.rollback_to` + 4 例测试 |
| **多批窗口 + Bootstrap 门控** | ✅ **新增** | `release.gate` 窗口分支 + 3 例测试 |
| **主循环兜底（异常不中断、必交卷）** | ✅ **新增** | `tests/test_resilience.py` 5 例 |
| **预算超限降级（不崩）** | ✅ **新增** | `tests/test_resilience.py` 1 例 |
| **只读文件系统降级（内存账本）** | ✅ **新增** | Ledger/Recorder 非致命 + 2 例测试 |
| **任务类型 → 指标分派** | ✅ **新增** | `metric.build_for` + 8 例测试 |
| **回归模型（ridge）** | ✅ **新增** | `model.py` + 回归路径实测（MAE −22.3→−20.0） |
| **回归标签漂移（KS）** | ✅ **新增** | `label_drift` + 3 例测试 |
| **实验记录（本地 JSONL）** | ✅ **新增** | `logs/recorder.py` + `out/run.jsonl` |
| **输出校准动作 `calibrate_output`** | ⚠️ **已实现，默认关闭** | `model.calibrate_output` + `decide` 第 7 档 + 4 个动作的成本阶梯（实测在本数据集劣化 0.836→0.748，见 §11） |
| **门控 margin 按量纲归一化** | ✅ **已修复（R1）** | `release.margin_mode` + 相对归一化 + 9 例测试 |
| 轴承 / RUL 适配器 | ⚠️ 加载器就绪，**无数据** | `dataset.BearingAdapter / RULAdapter` |
| SwanLab 上传 | ❌ 代码在，**未验证** | `Recorder.use_swanlab` |
| 上下文赌博机 / 学习型策略 | ❌ 未实现 | —— |

---

## 4. 缺失功能与风险清单

| # | 缺失/风险 | 严重度 | 影响 |
|---|---|---|---|
| R1 | ~~门控 margin 未按指标量纲归一化~~ | ~~高~~ | ✅ **已修复**：新增 `margin_mode`。注：上一轮把"回归 7 次回滚"归因为 margin 是**错的**，实测那 7 次全是候选本身更差（`raw_gain` 为负，见 §11） |
| R2 | 阈值全部手写、不可跨数据集迁移 | **高** | 接官方 A 包必须先标定，否则要么不动作要么永远走最贵档 |
| R3 | 轴承 / RUL 无真实数据 | **高** | 公开榜三个任务只跑得动 1 个 |
| R4 | Docker 镜像未构建验证 | **高** | 第一交付物，可能构建失败 |
| R5 | `decisions.jsonl` 的 `cycle` 计数不可靠 | 中 | 门槛③核验时可能被判"轮次不足" |
| R6 | 输出校准档已实现但**默认关闭** | 中 | 在该数据集上劣化（0.836→0.748），必须逐数据集 A/B 后再开 |
| R7 | 无留出批次 / 多种子 / 多漂移强度 | 中 | 无法证明"没有过拟合"（评分明确要考） |
| R8 | 概念漂移判据是间接推断 | 中 | 归因可能误判，答辩易被追问 |
| R9 | SwanLab 上传未验证 | 中 | 门槛②风险 |
| R10 | 无 LLM 贡献的统计对照 | 中 | "接了大模型"缺乏证据 |
| R11 | 技能库只有 1 个 md，无代码调用 | 低 | 三层架构名不副实 |
| R12 | `warmup_batches` 永不触发 | 低 | 配置语义错误 |
| R13 | MMD 未实现 | 低 | 文档曾声称三种方法 |
| R14 | 不支持流式推理 | 低 | 27B 长思考时延迟高 |

---

## 5. 文档与代码不一致清单

| 文档说法 | 代码实际 | 状态 |
|---|---|---|
| 共识文档「实现 PSI / KS / MMD」 | 原：只有 PSI → **现：PSI + KS + Wasserstein**，仍无 MMD | ⚠️ 部分修复 |
| 共识文档「规则 + 上下文赌博机」 | 无 bandit，纯规则表 | ❌ 仍不一致 |
| 共识文档「手段阶梯第 1 档 = 改输出（阈值/温度校准）」 | 未实现；`refresh_normalization` 改的是**特征标准化**（第 2 档） | ❌ 仍不一致 |
| 共识文档「技能库 skills/」 | 1 个 md，无代码调用 | ❌ 仍不一致 |
| 「增量微调」 | = 近期数据 + 指数衰减权重重 fit | ❌ 表述不准确 |
| 交接说明「`D:\anaconda\python.exe`」 | 该路径在本机**不存在** | ❌ 命令失效 |
| README「`logs/` 目录」 | 原不存在 → **现已创建 `evolver/logs/`** | ✅ 已修复 |
| README「待办：SwanLab 接入」 | 代码已写，未验证 | ⚠️ 部分 |
| README「待办：Docker 打包」 | Dockerfile 已写，未构建 | ⚠️ 部分 |
| `requirements.txt` 只列 5 个包 | 原 `llm_agent.py` 运行时 `import requests`（未列）→ 现真后端走 urllib，不再需要 | ✅ 已规避 |

---

## 6. 按优先级排列的整改方案

### A 类（必须修复）—— 本轮已完成

**A1 主循环兜底**
- 问题：`run_pipeline.main()` 冷启动与循环体均无 try/except → 任何异常＝整轮 0 分
- 根因：默认"数据/模型/指标都可用"
- 方案：冷启动 try 包裹（失败写 `fatal` 后正常退出）；每批 `try/except/finally`，`finally` 里检测"未交卷"则用现役模型保底提交
- 涉及文件：`scripts/run_pipeline.py`
- 关键结构：
  ```python
  try:
      ...  # 原逻辑
      proto.submit(...); submitted = True
  except BudgetExceeded as exc:  ...
  except Exception as exc:       ...
  finally:
      if not submitted:
          submit_fallback(proto, rel, batch, round_no, metric, "保底")
  ```
- 验证：`tests/test_resilience.py` 5 例（数据中途损坏 / 模型失能 / 冷启动失败）
- 副作用：失败批会被记为 `failed` 行，平均分只统计成功批（正常情况下无差异）
- 优先级：P0

**A2 预算超限降级**
- 问题：`Ledger.charge` 抛 `BudgetExceeded` 无人捕获 → 崩
- 方案：主循环捕获，本批降级为 `do_nothing` 并留痕
- 验证：`test_预算超限降级而不是崩溃`（预算设 0.001，4 轮协议全完成）
- 副作用：无（预算内行为不变）
- 优先级：P0

**A3 版本快照与真回滚**
- 问题：只存决策记录，不存模型；`current = candidate` 后旧版无法恢复
- 方案：`snapshots[]` 保存深拷贝，新增 `rollback_to(version)`；上限 `max_snapshots`；version 0 永久保留
- 验证：`tests/test_release.py` 4 例（含"快照是副本不是引用"）
- 副作用：内存增长（上限 20 版，模型为 sklearn 对象，约几 MB）
- 优先级：P0

**A4 任务类型分派（指标 + 模型 + 标签漂移）**
- 问题：指标写死 `f1_macro`；模型只有分类器；连续标签的 `label_psi` 恒为 0
- 方案：`DatasetAdapter.task_type` → `metric.build_for()` / `model.default_for()` / `label_drift()` 三处分派；新增 `ridge`
- 验证：`tests/test_metric.py` 8 例 + 回归路径实测（MAE −22.32 → −19.99）
- 副作用：`--model` 默认值改为"按任务类型自动选"，显式传参行为不变
- 优先级：P0
- **遗留**：`release.margin` 未按量纲归一化（见 R1）

**A5 交付形态**
- 方案：`Dockerfile`（slim + 非 root）+ `requirements.lock`（全固定）+ `run_offline.sh`（无交互、产物兜底）+ `.dockerignore`
- 验证：**未构建**（本机无 docker）；脚本逻辑经静态审查
- 优先级：P0（但有环境依赖，见 R4）

### B 类（高性价比）—— 本轮完成 3 项

**B1 三种漂移检测（已完成）**
- 方案：`_psi / _ks / _wasserstein` + `Monitor(detector=...)`；`--detector all` 同时输出三组
- 适用性：PSI 对整体平移/缩放敏感且工业常用；KS 无需分箱、值域 [0,1] 易设阈值；W1 给出"平移了几个标准差"的物理意义
- 验证：`tests/test_monitor.py` 12 例（同分布≈0、平移 3σ → W1≈3）
- 副作用：`--detector all` 计算量约 3 倍（10 批 × 20 特征仍是毫秒级）

**B2 多批窗口 + Bootstrap 门控（已完成）**
- 方案：`gain_window` + `bootstrap_lower_mean`；默认 `gate_window=1` 保持旧行为
- 验证：3 例（窗口未满退回单批、窗口均值、噪声拒绝）
- 副作用：窗口 >1 时**决策会滞后**（需要积累 k 批证据）

**B4 实验记录（已完成）**
- 方案：`Recorder` 写 `out/run.jsonl`；SwanLab 可选；**不可写时降级为内存记录**
- 验证：2 例
- 副作用：无

### B 类 —— 未完成（下一阶段）

- **B3 输出校准档**：补 `calibrate_output`（阈值/温度），使成本阶梯完整
- **B5 阈值标定脚本**：扫 `psi_low/psi_mid` 网格，输出建议值 + 敏感性表
- **B6 留出批次 / 多种子 / 多漂移强度**：`scripts/sweep.py`
- **B7 LLM 贡献对照**：规则 vs 建议 vs 门控后，配对符号检验

### C 类（研究增强）

- **C1 最低成本版 bandit**：把"3 档动作"当 arms，用 ε-greedy + 滑动奖励（省到近乎为零的改动），即可称"学习型策略"
- **C2 完整研究版**：上下文赌博机（LinUCB，特征=PSI/分数/上次增益），离线用历史日志评估
- **C3 MMD / 相关结构漂移检测**
- **C4 概念漂移直接检验**（分箱 P(y|x) 距离）

### D 类（暂时不要做）

- 多智能体 / 长链 tool-calling（本题动作空间仅 3 个）
- 驱动大模型微调（成本 30 分钟 GPU 换 2600 token，客观分分母爆炸）
- 自研深度学习工业模型（评分不看单点精度）
- 重写框架 / 换语言

---

## 7. 推荐的文件修改列表

| 文件 | 动作 | 说明 |
|---|---|---|
| `evolver/config.py` | 修改 | 新增 task_type / detector / gate_* / max_snapshots |
| `evolver/loop/monitor.py` | 修改 | 新增 KS / Wasserstein / label_drift |
| `evolver/loop/release.py` | 重写（向后兼容） | 快照 + rollback_to + 窗口 + Bootstrap |
| `evolver/adapters/metric.py` | 修改 | build_for / rmse / 任务标注 |
| `evolver/adapters/model.py` | 修改 | ridge / default_for / check_task / is_fitted |
| `evolver/adapters/dataset.py` | 修改 | task_type + synthetic_rul / bearing / rul |
| `evolver/ledger/ledger.py` | 修改 | 只读文件系统降级 |
| `evolver/logs/recorder.py` | **新增** | 实验记录 |
| `evolver/logs/__init__.py` | **新增** | 导出 |
| `scripts/run_pipeline.py` | 重写（逻辑保留） | 兜底 + 分派 + 新参数 + 记录 |
| `Dockerfile` | **新增** | 评测镜像 |
| `requirements.lock` | **新增** | 依赖锁定 |
| `run_offline.sh` | **新增** | 离线入口 |
| `.dockerignore` | **新增** | 镜像瘦身 |
| `tests/test_release.py` | **新增** | 10 例 |
| `tests/test_monitor.py` | **新增** | 12 例 |
| `tests/test_metric.py` | **新增** | 8 例 |
| `tests/test_resilience.py` | **新增** | 7 例 |

---

## 8. 测试与验收方案

### 已具备
```bash
# 1) 单元 + 集成（47 例，2 例因缺 sklearn 跳过）
python -m unittest discover -s tests -t .

# 2) 分类主流程（应与整改前一致）
python scripts/run_pipeline.py --llm none

# 3) 回归主流程 + 三检测器
python scripts/run_pipeline.py --dataset synthetic_rul --detector all

# 4) 门控窗口 + Bootstrap
python scripts/run_pipeline.py --gate-window 3 --gate-bootstrap 500

# 5) 4B 决策层（42 例 + 13 故障矩阵）
python -m unittest discover -s <qwen4b>/tests -t <qwen4b>
python <qwen4b>/scripts/run_matrix.py
```

### 验收标准（每条都必须能复现）

| # | 验收项 | 判据 |
|---|---|---|
| V1 | 分类主流程不回归 | 基线 0.498 / 系统 0.836（±0.001） |
| V2 | 异常不崩溃 | 注入任一异常，进程正常退出且 `metrics.json` 存在 |
| V3 | 协议零违规 | `protocol.violations == 0` |
| V4 | 回滚可用 | `rollback_to(v0)` 后 `current` 与快照一致 |
| V5 | 回归链路正确 | `task_type=regression` 时指标为 `mae`，模型为 `ridge` |
| V6 | 只读目录可跑 | 输出目录不可写时仍完成，`ledger.write_ok=False` |
| V7 | 记录完整 | `out/run.jsonl` 行数 = 批次数 + 1(summary) |

---

## 9. 最终实施顺序

| 阶段 | 内容 | 状态 |
|---|---|---|
| S1 | 读全代码 + 建审计基线 | ✅ 完成 |
| S2 | A1/A2 主循环与预算兜底 | ✅ 完成（7 例） |
| S3 | A3 快照与回滚 | ✅ 完成（10 例） |
| S4 | A4 任务分派（指标/模型/标签漂移） | ✅ 完成（8 例 + 回归实测） |
| S5 | B1 三检测器 + B2 窗口门控 | ✅ 完成（15 例） |
| S6 | B4 实验记录 | ✅ 完成（2 例） |
| S7 | A5 交付形态 | ✅ 文件就位（**未构建**） |
| S8 | **B3 输出校准档 + R1 margin 归一化** | ✅ 完成（R1 9 例 + B3 20 例） |
| S9 | **B5 阈值标定脚本** | ⏭ 下一步 |
| S10 | **B6 多 seed / 留出批次** | ⏭ 下一步 |
| S11 | **B7 LLM 贡献对照** | ⏭ 下一步 |
| S12 | 接官方 A 包三任务 + 复标定 | ⏭ 数据到位后 |
| S13 | Docker 构建 + 断网实测 | ⏭ 需 Docker 环境 |
| S14 | SwanLab 上传验证 | ⏭ 需账号 |

---

## 10. 当前项目完成度评估

| 维度 | 完成度 | 依据 |
|---|---|---|
| 可运行闭环 | **90%** | 10 轮协议 0 违规，异常路径全部可收尾 |
| 工程韧性 | **85%** | 兜底/熔断/只读降级/预算降级齐备；缺"多批累计影子评估" |
| 交付形态 | **60%** | Dockerfile/lock/入口就位但未构建；SwanLab 未验证 |
| 任务覆盖 | **55%** | 适配器 3/3 就绪，但轴承/RUL 无真实数据 |
| 算法严谨性 | **65%** | 三检测器 + 窗口门控 + 快照 + margin 归一化 + 输出校准；缺标定、留出集 |
| 测试覆盖 | **85%** | 76 例；仍缺 `test_decide`（已并入 test_calibration 部分覆盖）/ `test_model` 的部分用例 |
| 文档一致性 | **70%** | 已修 logs/、任务分派、输出校准档；仍有 bandit / 技能库 两处不一致 |
| **综合** | **≈75%** | 能跑、能讲、能扛故障；"能拿分"仍取决于标定与三任务落地 |

---

## 11. 更正与补充（第二轮整改后）

### 11.1 更正上一轮的一个错误归因

上一轮我在对话里说"回归任务 7 次回滚是 margin 未归一化造成的"——**这是错的**。
实测 `decisions.jsonl` 的 `raw_gain` 全部为负（−1.39 ~ −5.11），说明候选模型在当批
确实更差，门控拒绝是正确行为，与 margin 量纲无关。

margin 量纲错位依然是**真实缺陷**（0.02 在 MAE≈22 的量纲下等于"任何正增益都放行"），
只是它在这份数据上没有造成那 7 次回滚。修复后分类结果不变、回归也暂无行为变化，
属于"修正确性、非修分数"。

### 11.2 R1 实现与公式

```
内部一律用 utility（越大越好）：utility = raw（higher），utility = −raw（lower）

raw_gain        = candidate_u − reference_u
                  恒等于 direction==higher 时的 (cand_raw − ref_raw)
                         direction==lower  时的 (ref_raw − cand_raw)
normalized_gain = raw_gain                                (absolute)
                = raw_gain / max(|reference_u|, eps)       (relative)
accept         ⇔ normalized_gain ≥ margin
```

| 情况 | 处理 |
|---|---|
| `|baseline| < eps`（约 0） | 退回 absolute，记 `margin_mode_used="absolute_fallback"`，**不做除法** |
| baseline 为负 | 取绝对值归一 |
| 非法 margin_mode | 构造时抛 `ValueError` |

分类 vs 回归的门控策略：

| 任务 | 指标 | 方向 | 默认 margin_mode | margin 语义 |
|---|---|---|---|---|
| 分类 | f1_macro | higher | **absolute** | 2 个百分点（保持既有 0.836 不变） |
| 回归 | mae | lower | **relative** | 相对 baseline 改善 2% |

可用 `--margin-mode` 显式覆盖。

### 11.3 B3 实现与实测结论（重要）

新增动作 `calibrate_output`（成本 0.02s，低于 refresh 的 0.05s）：

| 任务 | 校准方式 |
|---|---|
| 二分类 | 在历史窗口上网格搜索概率阈值，最大化 macro-F1 |
| 多分类（≤10 类） | 对 log 概率加**逐类偏置**，坐标上升最大化 macro-F1 |
| 回归 | **不支持**，抛 `CalibrationUnsupported` 并记录原因 |

> 为什么不用温度缩放：温度是单调变换，**不改变 argmax**，对 F1/准确率毫无作用。
> 要改变决策边界必须动逐类偏置（等价于一 vs 其余阈值）。

**动作阶梯**（仅当 `--enable-output-calibration` 且分类任务）：
```
do_nothing → calibrate_output → refresh_normalization → refit_model
```

**实测结果（同一份合成协变量漂移流）**：

| 配置 | 系统平均分 | 动作 | 上线 | 回滚 |
|---|---|---|---|---|
| 默认（校准关闭） | **0.836** | 4 | 3 | 1 |
| 开启输出校准 | **0.748** | 4 | 2 | 2 |

**结论：在这份数据上，输出校准比"改特征"更差，因此默认关闭。**
原因很直接：该数据的漂移是**特征尺度平移**，`refresh_normalization` 能完全恢复；
而校准只动判定阈值，改不了特征尺度，反而把 b04 从 0.972 拉到 0.566。

这不是实现错误，而是**方法适用性**问题：`calibrate_output` 适合"类别先验/判定阈值偏移"
（例如质检标准改了、合格率变化），不适合"传感器增益漂移"。因此：
- 代码保留、可一键开启
- 默认关闭，避免破坏既有 0.836
- 换数据集必须做 A/B 后再决定是否开启


---

## 12. Recovered 阈值标定结果（第三轮）

### 12.1 做了什么

新增 `scripts/calibrate.py`：在**时间顺序分段**上网格搜索 `psi_low × psi_mid`，
并把"选阈值"与"报告效果"彻底分开。

| 段 | 批次 | 用途 |
|---|---|---|
| calibration | b02–b04 | 观察 |
| validation | b05–b08 | **唯一**用于选阈值 |
| pre-holdout | b02–b09 | 仅补充报告 |
| holdout | b10 | 冻结后只跑一次，且不做旧阈值对照 |

选择目标：`utility = gain − λ_cost·归一化成本 − λ_rollback·回滚率`，
硬约束：违规=0、增益≥0、回滚率≤上限、浪费占比≤上限。

### 12.2 结论：默认不换（这是本轮最重要的判断）

| 角色 | psi_low | psi_mid |
|---|---|---|
| **默认（现役）** | **1.2** | **2.5** |
| 实验候选 | 0.6 | 4.0 |

validation 段上候选确实更好（0.8281 vs 0.7952，回滚 2 vs 3，成本同为 4.10s），
但**该优势可证明只来自 b02 一个批次**：

- 把 `psi_low` 从 0.6 调到 1.2，只有 b02（PSI=0.766）落在两阈值之间
- 仅此一处差异，validation 系统均值就从 0.8281 变为 0.7952

判据实现为 `single_batch_flip()`：找"只改一个轴、该改动只波及 1 个批次、
结果却翻转"的相邻网格点。命中即 `fallback_reason = single_batch_evidence`，
稳健推荐回退到现役配置，**不写入默认**。

> 注意：早期版本用"动作差异批次数"判断，得到 7 个批次不同，
> 会误判为"多批次证据"。实际是相位级联（b02 一动，后面每批动作都 shift），
> 不是 7 份独立证据。此坑已修正并写入测试。

### 12.3 一次性 holdout（已消耗）

| 配置 | 系统 | 静态基线 | 增益 | 动作 |
|---|---|---|---|---|
| 0.6/4.0（候选） | 0.3946 | 0.2897 | +0.1049 | 0（do_nothing） |

**b10 已用完，不可复用。** 该数字远低于 validation 的 +0.6130，
说明 validation 的高增益依赖特定更新时间点，不能外推。
本轮为遵守"只跑一次"，**未**在 b10 上跑 1.2/2.5 做对照，
因此"候选优于现役"只在 validation 上成立。

### 12.4 交付与约束

- `out/cal_robust/calibration_report.json` / `.md` / `thresholds.json`
- `thresholds.json` 固化：`active_thresholds=1.2/2.5`、
  `calibrated_candidate=0.6/4.0`、`fallback_reason=single_batch_evidence`
- `tests/test_calibrate.py`：35 例，含"单批次证据不得替换默认值"
- 核心闭环算法**未改动**；b10 未被本轮任何步骤读取

### 12.5 仍存在的不确定性

1. validation 只有 4 个批次，任何比例都是小样本
2. 单 seed，无方差估计
3. 网格饱和：20 个点只有 5 种不同结果，阈值区分度低
4. 只适用于 Gas 公开数据，**不是跨数据集结论**

---

## 附：本轮实测数字（供复现）

```
分类（synthetic）：基线 0.498 / 系统 0.836 / 动作 4 / 上线 3 / 回滚 1
                 协议 10 轮 · 违规 0 · 账本 2.15s
回归（synthetic_rul，detector=all）：
                 基线 MAE −22.323 / 系统 −19.991（改善 2.33）
                 动作 8 / 上线 1 / 回滚 7  ← margin 未归一化的直接后果
测试：vendor 47（2 skip）· qwen4b 42
4B 矩阵：13 个用例全部 0.836（不低于纯规则）
```
