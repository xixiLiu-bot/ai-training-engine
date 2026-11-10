# 智训引擎 · 工业模型自进化系统

> 第二屆北京市大学生「人工智能+」创新大赛 · 命题赛题
> 一句话：**做一个能让模型一直好用的系统，而不是一个更准的模型。**

## 现在是什么状态

已经能跑通完整闭环（用合成漂移数据）：

```
监测漂移 → 归因诊断 → 决策 → 执行 + 影子评估 → 上线 / 回滚
```

## 跑一遍

**最简单：双击项目根目录的 `run.bat`** —— 自动跑完，最后会弹出曲线图。

或者用命令行（在项目根目录）：

```bash
python scripts/make_data.py        # 造一条会漂移的数据流（10 批）
python scripts/run_pipeline.py     # 静态基线 vs 自进化闭环
```

> `python` 必须指向 Anaconda（带 sklearn 的那个）。
> 用 `where python` 确认，应该是 `D:\anaconda\python.exe`。
> 如果报 `ModuleNotFoundError: sklearn`，说明用错了 Python，改成
> `D:\anaconda\python.exe scripts\run_pipeline.py`。

产出在 `out/`：

| 文件 | 是什么 |
|---|---|
| `metrics.json` | 每批分数，用于画曲线 |
| `decisions.jsonl` | 每一次决策的完整记录（**门槛③的证据**） |
| `curve.png` | 静态基线 vs 我们，两条线 |

## 环境

用 Anaconda 的 Python（已自带 sklearn / matplotlib）：

```bash
D:\anaconda\python.exe -V     # 3.13.9
```

换机器时装依赖：

```bash
pip install -r requirements.txt
```

## 目录

```
evolver/
├── adapters/          三个适配器——闭环只认接口，不认实现
│   ├── dataset.py     数据：synthetic / gas / official(占位)
│   ├── model.py       模型：logreg / gbdt / rf，含“零成本校准”
│   └── metric.py      指标：f1_macro / accuracy / mae
├── loop/              闭环五环节
│   ├── monitor.py     ① 漂移监测（PSI）
│   ├── diagnose.py    ② 归因诊断（协变量 vs 概念）
│   ├── decide.py      ③ 决策器（规则版，成本阶梯）
│   ├── evaluate.py    ④ 影子评估
│   └── release.py     ⑤ 门控 + 回滚
├── engine/            决策大脑（可选层）
│   ├── llm_agent.py   大模型建议器：可切换后端，出错就降级
│   └── policy.py      规则 + 大模型 → 过四道闸 → 最终动作
├── protocol/          评测协议（一问一答）
│   └── batch_stream.py  LocalEvaluator / RemoteEvaluator(占位) / Submission
├── ledger/            预算账本（算力 / 标注 / Token 三重上限）
└── config.py          所有阈值集中在这里

scripts/
├── make_data.py       合成漂移流
└── run_pipeline.py    主流程 + 画图

tests/                 回归测试（协议约束）
skills/                技能库（可复用的处置经验）
data/                  数据（不进仓库）
out/                   产物（不进仓库）
```

## 评测协议：一问一答（很重要）

依据 11/6 说明会答疑：

> B 包数据按批次下发，**必须先输出当前批次模型或不上线的判断，才能获取下一批数据**。

所以评测不是「把数据流灌给你」，而是**一问一答**：

```
要第 1 批 → 拿到 → 处理 → 【交卷】 → 才给第 2 批 → ...
```

这条约束的真正作用是**从结构上堵死"偷看未来"**——你看不到后面的数据，
就没法为了测试流调参。这正是评分里"是否过拟合"要防的事。

代码里用 `LocalEvaluator` 强制执行，**违反会直接抛异常**：

| 违规动作 | 结果 |
|---|---|
| 没交卷就要下一批 | `ProtocolViolation` |
| 想看还没下发的批次 | `ProtocolViolation` |
| decision 不是 deploy / keep | `ProtocolViolation` |

```bash
D:\anaconda\python.exe -m unittest tests.test_protocol -v
# 6 个测试全过，其中一条专门验证"只加载真正下发过的批次"
```

跑完会打印协议结果：

```
协议    10 轮 | 部署 4 | 维持 6 | 违规 0
```

企业赛时把 `LocalEvaluator` 换成 `RemoteEvaluator`（等官方 SDK），
上层闭环代码一行都不用动。

## 三个关键概念（别搞混）

| | 是什么 | 会变吗 |
|---|---|---|
| **模型**（权重） | 被训练、被更新的那堆参数 | 变，但不可跨数据集迁移 |
| **闭环**（系统） | 你写的代码：监测→归因→决策→评估→回滚 | 结构固定，不训练，只调试 |
| **技能**（经验） | “遇到这种漂移，按这几步做” | 会越攒越多，跨模型通用 |

## 更新手段的性价比阶梯（这套代码的核心）

| 动作 | 成本 | 实现位置 |
|---|---|---|
| `do_nothing` | 0 | —— |
| `refresh_normalization` | ≈0.05s | `model.refresh_normalization()` |
| `refit_model` | ≈2s | 近期加权重训 |

**省钱就省在这里**：能靠“重算标准化”解决的，绝不重训。

## 一次决策记录长这样

```json
{
  "cycle": 1, "batch": 4,
  "history_read": "近几批 PSI 0.02 → 0.07；当前分数 0.988",
  "hypothesis": "诊断=covariate；特征整体平移，重算标准化应可恢复",
  "experiment": { "action": "refresh_normalization", "cost_sec": 0.05 },
  "result": { "f1_before": 0.4752, "f1_after": 0.9762 },
  "decision": "accept",
  "reason": "同批对比：候选 0.9762 vs 当前 0.4752，增益 +0.5011 ≥ 门控 0.02"
}
```

门槛③要求「至少 3 次 读历史 → 提假设 → 做实验 → 做决策」，
**每一轮就跑出一条这样的记录**。这是留痕，不是补文档。

## 当前实测结果（合成数据，10 批）

```
平均分   基线 0.499   系统 0.837
决策     动作 4 次 | 上线 3 次 | 回滚 1 次
成本     2.15 秒 / 2000 条标注
```

故事线：前两批不动（正确决策）→ 协变量漂移用零成本校准救回来
→ 第 6 批判定标准改变，校准失效 → 升级重训 → 之后一直稳定。

## 换官方数据只改一个文件

写一个 `OfficialAdapter`（`evolver/adapters/dataset.py` 里已有占位），
实现 `n_batches` / `get(batch_id)` 两个东西即可，`loop/` 一行都不用动。

## 决策层：接不接大模型

```bash
python scripts/run_pipeline.py --llm none      # 只用规则（默认，保底）
python scripts/run_pipeline.py --llm mock      # 假模型，验证链路
python scripts/run_pipeline.py --llm pytrio    # 真模型（等额度）
python scripts/run_pipeline.py --llm ollama    # 本地 ollama（开发替代）
```

三种模式实测结果**完全一致**（平均分 0.837）：

| 模式 | 结果 | 说明 |
|---|---|---|
| `none` | 0.837 | 只有规则 |
| `mock` | 0.837 | 建议被采纳或拦下，与规则一致 |
| `mock_bad` | 0.837 | 9 次调用全失败 → 全部降级到规则 |

**这就是设计目标：接了大模型，最坏情况也不低于纯规则。**

### 四道闸（`engine/policy.py`）

大模型的建议必须全过才算数，任何一条不过就退回规则：

1. 能解析成合法 JSON
2. 动作在白名单内
3. **不许重复已被证明无效的便宜动作**（实测出来的：没这道闸，系统从 0.84 掉到 0.56）
4. 已接近历史最好时不接受“重训”建议

### 拿到 PyTRiO 额度后改哪两行

`evolver/engine/llm_agent.py` 里的 `PRESETS["pytrio"]`：

```python
"base_url": "http://<赛题组给的地址>/v1",
"model":    "Qwen3.8-27B",
```

其余代码不用动——协议是 OpenAI 兼容的。

## 待办

- [ ] 拿到官方数据 → 实现 `OfficialAdapter`
- [ ] SwanLab 接入（`logs/`，见官方《实验记录规范》）
- [ ] Docker 打包，断网自测
- [ ] 接 Qwen3.8-27B 作为决策层（企业赛阶段）
- [ ] 补 pytest：门控/回滚必须有回归测试

## 两个踩过的坑（别再踩）

1. **重训不能用当前批的标签**。用第 b 批训练、又在第 b 批验收 = 数据泄漏，分数虚高。
   本仓库用 `b-4 .. b-1` 训练，再到 `b` 上检验。
2. **概念漂移必须做成“可学习的规则改变”**。把标签随机翻掉会让标签不可学，
   重训反而更差——那不是概念漂移，那是数据被污染。

## 程序结构

![程序结构](assets/fig3_design.png)

五个环节按顺序走一遍就是一次决策。右边标了四个「以后要改就改这里」的位置——
**加神经网络只需要动 dapters/model.py 一个文件**，五个环节、协议层、门控回滚都不用改。

## L2 经验表：策略层自改进

evolver/memory/experience.py 把「哪类漂移用了什么动作、结果如何」记成一张表：

    漂移类型 → 动作 → 次数 / 累计增益 / 累计成本 / 单位成本增益

决策顺序变成三层，后面的可以覆盖前面的，但都能退回：

    规则引擎（保底）→ 经验记忆（免费，有证据才改判）→ 大模型（过四道闸）

**两个防坑设计**（都是踩过才知道的）：

1. **失败惩罚**：最近连续两次回滚的动作，不再推荐
2. **20% 探索率**：不能因为一次成功就锁死，否则会陷在过时的经验里

> 实测教训：没有这两条时，经验表让分数从 0.722 掉到 0.709 —— 
> 第 3 批校准成功后，它以为校准永远有效，可后面早就失效了。

开关：--no-memory 可关闭，用于 A/B 对照。

## 续训（warm start）：把模型当经验用


efit_model 现在是**从当前模型接着训**，不是重建。实测三种做法：

| 做法 | 第4批 耗时 / F1 |
|---|---|
| 冷启动重训 | 128.5 ms / 0.7851 |
| 续训（保留旧标准化） | 253.6 ms / 0.7946 |
| **续训 + 刷新标准化** | **75.0 ms / 0.8049** |

**必须同时刷新标准化**——否则权重配着旧尺度，反而被拽住，又慢又差。

## 决策大脑：接了 Pydantic AI 框架

evolver/engine/llm_agent.py 现在有两条路径：

| 路径 | 说明 |
|---|---|
| 手写 HTTP + 正则解析 | 兼容性最好，保留 |
| **Pydantic AI 框架** | 结构化输出，白名单从“事后检查”变成“结构约束” |

`python
class Decision(BaseModel):
    action: Literal[do_nothing, refresh_normalization, refit_model]
    reason: str
`

模型吐出不存在的动作，框架直接拒绝并重试；重试还不行就抛异常，被我们降级到规则。

### 后端一览

`ash
--llm none            # 纯规则（默认，保底）
--llm pydantic-test   # 框架 + TestModel（离线验证链路）
--llm pydantic-sim    # 框架 + 模拟模型（能给出合理决策）
--llm pydantic        # 框架 + 真模型（改 base_url 即可接 Qwen）
--llm pytrio          # 手写 HTTP 直连
--llm ollama          # 本地 ollama
`

### 实测：三种决策层的对比

| 决策层 | 系统 F1 | 说明了什么 |
|---|---|---|
| 纯规则 | 0.740 | 能用 |
| 框架 + TestModel | **0.320（=基线）** | **假模型不会思考 → 系统退化成静态基线** |
| 框架 + 模拟模型 | 0.752 | 框架路径能跑出好结果 |

第二行值得记住：**接入一个大模型不等于变强**。TestModel 永远返回“不动”，
系统就一次也不更新，分数掉回基线。**关键在于模型的判断力，不在于有没有接。**

## 动作清单：让成本阶梯跟模型走

以前所有模型共用一张成本表（校准 0.05s / 重训 2.0s）。但不同模型的重训
代价差好几倍：逻辑回归 0.5s，GBDT 2.2s。

现在每个模型自己声明：

    logreg  {不动:0, 校准:0.05, 重训:0.5}
    gbdt    {不动:0, 校准:0.05, 重训:2.2}
    rf      {不动:0, 校准:0.05, 重训:1.5}
    mlp     {不动:0, 校准:0.05, 重训:1.2}

启动时会打印出来，换模型不用改决策代码。

### 一个踩过的坑（记下来）

曾经想「把对树模型有害的校准直接删掉」，实测反而更差：
**带校准 0.478 → 删掉 0.437**。

因为它不只是修，还是探——先花 0.05 秒试一下，不行下一批再升级。
删掉它，系统就变得急躁，每次都直接上最贵的重训。

**所以清单只管成本，效用交给经验表学。** 两者别混。

## 四个模型都能跑

    --model logreg / gbdt / rf / mlp

mlp 是最简单的神经网络（sklearn 的 MLPClassifier，64-32 两层），
零额外依赖——**不用装 PyTorch**（torch 占 0.48 GB，镜像限额 2 GB）。

实测（气体数据，10 批）：

| 模型 | 基线 | 系统 | 说明 |
|---|---|---|---|
| logreg | 0.320 | **0.762** | 稳 |
| mlp | 0.367 | 0.641 | 神经网络能用 |
| rf | 0.426 | 0.546 | 基线本身最高 |
| gbdt | 0.358 | 0.437 | 重训贵，闭环里吃亏 |

**注意**：只有 10 个批次，这些差异不足以断定哪个模型更好。
