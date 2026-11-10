# 数据接入审查报告（Data Intake Review）

> 状态：**仅审查，未改代码**
> 核实方式：UCI 官方页、XJTU-SY 官方仓库 README、data.gov 数据集页 + 本地 `data/gas` 实测
> 结论口径：能核实的事实写明来源；不能核实的标「待下载确认」

---

## 0. 三个必须先说清的前提

**前提 1：这三个是公开替代数据，不是官方 A/B 包。**
赛题宣讲点名的三个任务是"气体传感器漂移 / 轴承退化识别 / 发动机 RUL"，与下面三个公开数据集一一对应，但官方数据包本身尚未下发。**用它们做本地验证可以，不能拿它们的分数对外声称是官方评测结果。**

**前提 2：你给的 Gas 数据其实已经接好了。**
本地 `data/gas/batch_01..10.npz` 实测：**13,910 样本 / 128 特征 / 6 类**，与 UCI 官方页的 13,910 / 6 gases / 128 维特征完全一致，批次大小 445·1244·1586·161·197·2300·3613·294·470·3600 合计正好 13,910。
→ Gas 这一条**不是"待接入"，是"已接入但未按新协议规范化"**。

**前提 3：你给的目录 `qwen4b/data/` 位置不对。**
实际代码包根目录是 `qwen4b/vendor/智训引擎-代码/`，数据应放在它的 `data/` 下（`.gitignore` 已经忽略 `data/`）。`qwen4b/` 是我的验证工具目录，不是项目仓库。

---

## 1. 三个数据集的结构与任务类型

| 数据集 | 规模 | 结构 | 任务类型 | 关键指标 | 许可证 |
|---|---|---|---|---|---|
| **UCI Gas Sensor Array Drift** | 13,910 样本 × 128 维 × 6 类 | 10 个批次 = 36 个月内 10 个时间段 | **分类**（6 种气体） | macro-F1 | ⚠️ **仅限研究用途，明确排除商业用途** |
| **XJTU-SY Bearing** | 15 个轴承，全寿命跑至失效 | 每个轴承 = 一条振动时序（2 通道） | **退化阶段分类** 或 **RUL 回归**（需转换） | macro-F1 / MAE / RMSE | README 未声明许可证，仅要求引用论文 |
| **NASA C-MAPSS** | FD001–FD004，100~260 台发动机 | 每台一条多变量时序，26 列 | **回归**（RUL） | MAE / RMSE（官方另有 PHM08 score） | 美国政府公开数据（PCoE） |

### 1.1 Gas Sensor Array Drift（已本地核实）

- **标签**：6 种气体（Ammonia / Acetaldehyde / Acetone / Ethylene / Ethanol / Toluene），浓度 5–1000 ppmv
- **特征**：16 个传感器 × 8 个特征（ΔR、|ΔR|、6 个 EMA）= **128 维**
- **批次**：`batch1.dat … batch10.dat`，libsvm 稀疏格式，第一列是类别（1–6）
- **漂移性质**：**真实的传感器老化漂移**（2007-01 → 2011-02，36 个月），这是三个数据集里**唯一天然的"时间漂移"**，最贴合赛题叙事
- **⚠️ 实测发现的坑**：batch_03 / 04 / 05 只有 **5 个类别**（缺 Toluene）。宏平均 F1 在这些批次上会因缺类而系统性偏低，且若模型误报该类，precision=0 直接拉低均值。**这不是数据错误，是原始测量就是这样。**
- **下载**：`https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip`（已验证可访问，返回 200，但服务端未给 Content-Length）

### 1.2 XJTU-SY Bearing

- **规模**：15 个滚动轴承，加速退化实验跑到失效；3 种工况（35Hz/12kN、37.5Hz/11kN、40Hz/10kN），每种 5 个轴承
- **结构**：每轴承一个目录（`Bearing1_1 … Bearing3_5`），每个采样点一个 CSV（**待下载确认**：普遍记录为 32768 点 × 2 通道，25.6 kHz，每次采样 1.28 s，每分钟采一次）
- **标签**：**没有逐点标签**，只有"跑到失效"这一事实 → 需自行构造：退化阶段（人工分段）或 RUL（总寿命 − 当前采样序号）
- **⚠️ 致命问题**：README 只给了 **Google Drive / Dropbox / MediaFire / MEGA / 百度网盘 / 个人主页**，**没有稳定的直链**。这意味着下载脚本无法做到"一键可复现"，必须人工下载后校验。体积预计 GB 级（需下载后确认）
- **许可证**：README 未声明，仅要求引用 Wang et al. 2020 (IEEE TR)。对外提交时需按"作者未声明许可证"处理，标注来源与引用

### 1.3 NASA C-MAPSS

- **结构**：`train_FDxxx.txt` / `test_FDxxx.txt` / `RUL_FDxxx.txt`，空格分隔，**26 列** = 单元号 + cycle + 3 个工况设置 + 21 个传感器测量
- **四个子集**（官方页原话）：

| 子集 | 训练轨迹 | 测试轨迹 | 工况数 | 故障模式 |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1（海平面） | 1（HPC 退化） |
| FD002 | 260 | 259 | 6 | 1（HPC 退化） |
| FD003 | 100 | 100 | 1 | 2（HPC + Fan） |
| FD004 | 248 | 249 | 6 | 2 |

- **标签**：测试集只给每条轨迹**末尾**的 RUL（`RUL_FDxxx.txt`），不是逐 cycle 的 RUL
- **⚠️ 关键认知**：FD001/FD003 是**单工况**，其分布变化主要来自"退化进程"而非"外部环境漂移"。若要用它体现协变量漂移，应选 **FD002/FD004（6 工况）**，工况切换本身就是输入分布漂移
- **下载**：data.gov 提供 `CMAPSSData.zip`；但 data.gov 的 API id 查询返回 404（页面 slug 与 API id 不一致），**真实直链需从页面资源按钮抓取**。另有 NASA PCoE 与 Zenodo 镜像（Zenodo API 本次返回 403，未核实）

### 1.4 一个必须指出的错位

赛题要求的是"**同一条数据流随时间漂移**"，而三个数据集的时间结构完全不同：

| 数据集 | 天然的"时间"是什么 | 是否天然存在"漂移" |
|---|---|---|
| Gas | 10 个真实月份段 | ✅ 有（传感器老化） |
| C-MAPSS | 每台发动机各自的 cycle | ⚠️ 弱（FD001/3 只有退化；FD002/4 靠工况切换） |
| XJTU-SY | 每个轴承各自的寿命 | ⚠️ 弱（是退化进程 + 轴承个体差异） |

**结论：C-MAPSS 与 XJTU-SY 需要人为构造"批次流"语义**，否则跑出来的 PSI 曲线反映的是"退化程度"而不是"分布漂移"，答辩时会被问穿。

---

## 2. 推荐的目录结构（按实际路径修正）

```
qwen4b/vendor/智训引擎-代码/          ← 实际项目根（不是 qwen4b/）
├── data/                             ← .gitignore 已忽略，不进仓库
│   ├── gas/
│   │   ├── raw/                      ← Dataset.zip 解压后的 batch1..10.dat
│   │   ├── processed/                ← batch_01..10.npz（已有）
│   │   └── README_DATA.md
│   ├── xjtu_sy/
│   │   ├── raw/                      ← Bearing1_1/ … Bearing3_5/
│   │   ├── processed/                ← batches.npz + index.json
│   │   └── README_DATA.md
│   ├── cmapss/
│   │   ├── raw/                      ← train/test/RUL_FD001..004.txt
│   │   ├── processed/                ← batches.npz + index.json
│   │   └── README_DATA.md
│   └── _cache/                       ← 下载的原始压缩包（可删）
├── scripts/
│   ├── download_gas.py
│   ├── download_xjtu_sy.py
│   ├── download_cmapss.py
│   ├── prepare_gas.py                ← 已存在，需扩展校验
│   ├── prepare_xjtu_sy.py            ← 新增
│   └── prepare_cmapss.py             ← 新增
├── checksums.sha256                  ← 下载物与 processed 产物的哈希清单
└── evolver/adapters/
    ├── dataset.py                    ← 现有 DatasetAdapter（闭环用）
    ├── sources.py                    ← 新增：统一数据源接口 + 三个实现
    └── bridge.py                     ← 新增：把 source 桥接成现有 get(batch_id) 协议
```

**为什么加 `sources.py` + `bridge.py` 而不是直接改 `dataset.py`**：
你要求的接口（`load/split/iter_batches/get_features/get_labels/get_task_type/get_metric/get_metadata`）与现有闭环协议（`name/task_type/n_batches/get(batch_id)`）是两套东西。硬塞进一个类会让 `run_pipeline` 和协议层都要改。桥接层能让**闭环零改动**，只在最外面换数据源。

---

## 3. 训练 / 验证 / 测试划分

> 总原则：**按"时间/实体"切分，绝不按行随机切分。** 官方 A/B 包到位后以官方划分为准，本方案仅用于本地验证。

### 3.1 Gas（10 批次）

| 用途 | 批次 | 理由 |
|---|---|---|
| train（冷启动） | batch_01 | 只有它，模拟"部署那天" |
| 在线流（评估） | batch_02 … batch_09 | 主评估段 |
| 留出（holdout） | batch_10 | **只看一次，不参与任何调参** |

- ❌ 禁止：把 batch_10 用于调 `psi_low/psi_mid`、门控 margin、动作阈值
- ❌ 禁止：把同一 batch 内部随机 7:3 划分当 train/test（同一次测量，几乎重复样本）

### 3.2 C-MAPSS（以 FD002 为主，FD001 作对照）

| 用途 | 划分方式 |
|---|---|
| train | 官方 `train_FDxxx.txt` 的**前 70% 发动机**（按 unit 号，整台切） |
| validation | 官方 train 的**后 30% 发动机**（整台切，绝不跨台） |
| test | 官方 `test_FDxxx.txt` + `RUL_FDxxx.txt`（官方定义，不重划分） |

- **批次流定义（二选一，建议 A）**：
  - **A. 寿命百分比分箱**：把每台发动机的运行归一化到 [0,1]，按 10 段切分 → batch_k = 所有发动机在第 k 个寿命段的样本。每批样本量充足，时间顺序清晰，RUL 标签随段递减。
  - B. 按 cycle 绝对值分箱：简单，但不同发动机寿命差异大（FD001 最短 128、最长 362 周期），批内分布不均。
- ❌ 禁止：把整张表按行 shuffle 后 8:2 划分 —— 同一台发动机的相邻 cycle 会同时进 train 和 test，**这是 C-MAPSS 最经典的泄漏，准确率能虚高到接近作弊**

### 3.3 XJTU-SY

| 用途 | 划分方式 |
|---|---|
| train | 工况 1 的 5 个轴承（Bearing1_1…1_5） |
| validation | 工况 2 的 5 个轴承（Bearing2_1…2_5） |
| test | 工况 3 的 5 个轴承（Bearing3_1…3_5） |

- **理由**：任务本质是"跨工况/跨轴承泛化"。按轴承整条切分，才能测出真实泛化能力
- **批次流定义**：对每个轴承，按寿命百分比切 10 段 → batch_k = 所有（当前可用的）轴承在第 k 段的窗口特征
- ❌ 禁止：同一轴承的相邻窗口分到 train/test（窗口重叠 → 泄漏）
- ❌ 禁止：用 test 轴承的失效点反推分段边界

---

## 4. 当前项目哪些模块可以复用

| 模块 | 能否复用 | 说明 |
|---|---|---|
| `loop/monitor.py`（PSI/KS/W1） | ✅ **直接复用** | 对 128 维气体、21 维传感器特征都适用，无需改 |
| `loop/diagnose.py` | ⚠️ 复用但需重标定阈值 | 气体数据 PSI 量级是 1~4，不是合成的 0.1~0.3 |
| `loop/decide.py` | ✅ 复用 | 4 档动作阶梯与数据集无关 |
| `loop/evaluate.py` | ✅ 复用 | 同批对照，与任务类型无关 |
| `loop/release.py` | ✅ **复用（已支持回归）** | `margin_mode` 已按方向归一化，MAE 场景可直接用 |
| `adapters/metric.py` | ✅ 复用 | `f1_macro / mae / rmse` 齐备，`direction` 已标注 |
| `adapters/model.py` | ✅ 分类复用；⚠️ 回归只有 `ridge` | C-MAPSS 可能需要 GBDT 回归，目前只有线性 |
| `adapters/dataset.py` | ⚠️ 部分 | `GasSensorAdapter/BearingAdapter/RULAdapter` 加载器已在，但**缺 load/split/iter_batches 全套** |
| `protocol/batch_stream.py` | ✅ 复用 | 一问一答，与数据源解耦 |
| `ledger/` `logs/` | ✅ 复用 | 预算与实验记录与数据无关 |
| `engine/policy.py` + `qwen4b/` | ✅ 复用 | 决策层与数据无关 |
| `scripts/run_pipeline.py` | ✅ 基本不用改 | 已有 `--dataset/--task-type/--detector/--margin-mode` |

**结论：闭环与决策层几乎零改动，工作量集中在"数据源适配 + 预处理 + 标定"。**

---

## 5. 哪些模块需要新增或修改

| 模块 | 动作 | 原因 |
|---|---|---|
| `adapters/sources.py` | **新增** | 实现你要的 8 个方法：`load/split/iter_batches/get_features/get_labels/get_task_type/get_metric/get_metadata` |
| `adapters/bridge.py` | **新增** | 把 source 桥接成 `get(batch_id)` 给闭环用（零侵入） |
| `adapters/dataset.py` | 修改 | 让 `build()` 能识别 `gas_drift / xjtu_sy / cmapss`，内部走 bridge |
| `adapters/model.py` | 修改 | 补回归模型（GBDT 回归 / `HistGradientBoostingRegressor`），当前只有 `ridge` |
| `scripts/download_*.py` | **新增** ×3 | 下载 + 校验 + 断点续传 |
| `scripts/prepare_xjtu_sy.py` | **新增** | 原始 CSV → 窗口特征 → 批次 |
| `scripts/prepare_cmapss.py` | **新增** | 26 列 → 特征/标签/寿命分段 → 批次 |
| `scripts/prepare_gas.py` | 修改 | 加 SHA256 校验与 `checksums.sha256` 写入 |
| `config.py` | 修改 | 每个数据集需要独立的 `psi_low/psi_mid` 预设（用 `--psi-low/--psi-mid` 亦可） |
| `scripts/calibrate.py` | **新增（原计划 B5）** | 阈值标定，接官方数据前必须有 |
| `data/*/README_DATA.md` | **新增** ×3 | 数据说明（你要求的第 5 条） |
| `checksums.sha256` | **新增** | 版本可复现 |
| `tests/test_sources.py` | **新增** | 适配层单测 |

---

## 6. 可能的数据泄漏风险（按严重度）

| # | 风险 | 场景 | 防法 |
|---|---|---|---|
| L1 | **同一实体的相邻样本跨 train/test** | C-MAPSS 按行随机划分；XJTU-SY 按窗口随机划分 | 按 unit / bearing 整条切分 |
| L2 | **用 test 标签调阈值/margin** | 拿 holdout 批次反复调 `psi_low` | 阈值只在 validation 上调；holdout 只看一次 |
| L3 | **当批训练 + 当批验收** | 重训窗口包含当前批 | 现有代码已用 `b-4..b-1`，**必须保持** |
| L4 | **标签延迟假设不成立** | 假设当批标签可用（现有代码如此），但官方协议可能延迟 | 加 `labels_available` 开关；延迟时用置信度分布代理（`label_drift_from_proba`） |
| L5 | **标准化泄漏** | 用全量数据拟合 scaler | 只用 train 批拟合；`refresh_normalization` 用当批 X 属"转导式"，需在 README_DATA 写明并说明理由 |
| L6 | **窗口重叠泄漏** | XJTU-SY 滑窗步长 < 窗长 | 步长 ≥ 窗长，或重叠窗只落在同一 split |
| L7 | **目标泄漏** | C-MAPSS 把 cycle 或 unit 号当特征 | 特征只用 21 个传感器 + 3 个工况设置 |
| L8 | **RUL 截断口径不一致** | 训练截到 125，评估不截断 | 训练/评估统一口径，与官方脚本对齐 |

---

## 7. 每个数据集的预处理方案

### 7.1 Gas

```
batch1..10.dat (libsvm) → 稠密 128 维 → 标签 1..6 → 0..5 → batch_XX.npz
```
- **已完成**（`prepare_gas.py` + `data/gas/batch_01..10.npz`）
- 待补：原始 `.dat` 的 SHA256、`README_DATA.md`、缺失类别的说明
- **不做标准化**（现有模型内部有 StandardScaler，且标准化本身是"更新手段"之一，不能提前做掉）
- **不做缺失值填补**（UCI 声明无缺失）

### 7.2 C-MAPSS

```
train_FDxxx.txt (26 列)
  ├─ 丢弃：unit(1)、cycle(2) 不进特征（cycle 可保留为时间索引，不进模型）
  ├─ 工况设置(3-5)：进特征；FD002/FD004 需按工况分组或做 one-hot
  ├─ 传感器(6-26)：21 维
  ├─ 常量列剔除：FD001 中若干传感器方差≈0（已知 s1,s5,s10,s16,s18,s19 近乎常量），需按数据实测决定
  └─ RUL 标签：train 中 RUL = (该 unit 最大 cycle − 当前 cycle)，**截断到 125**
```
- 按寿命百分比分成 10 段 → batch_01..10.npz（X: 特征, y: RUL）
- 记录每批的 unit 列表，供排查泄漏

### 7.3 XJTU-SY

```
BearingX_Y/Z.csv (32768×2)
  ├─ 时域特征：RMS、峰峰值、峭度、偏度、峰值因子、裕度因子（每通道 6 个 × 2 通道 = 12 维）
  ├─ 每 W 个采样点（如 10 个）聚合成一个窗口 → 一个样本
  └─ 标签二选一：
       阶段分类：按寿命百分比分 4 段 → 0/1/2/3（推荐先做这个，复用现有分类链路）
       RUL 回归：剩余采样数 / 总采样数
```
- ⚠️ 原始 CSV 体积大，**必须流式处理**（逐个文件读→算特征→丢），不能全部读进内存
- 窗口步长 = 窗长（不重叠），避免 L6

---

## 8. 指标与评测协议

| 数据集 | 任务 | 主指标 | 辅助 | 说明 |
|---|---|---|---|---|
| Gas | 6 分类 | **macro-F1** | PSI/KS 漂移曲线 | 缺类批次需单独标注，避免误读 |
| XJTU-SY（阶段分类） | 4 分类 | **macro-F1** | 混淆矩阵 | 类别不均衡，macro 比 accuracy 更能反映问题 |
| XJTU-SY（RUL） | 回归 | MAE | RMSE | 需统一 RUL 归一化口径 |
| C-MAPSS | 回归 | **MAE** | RMSE、PHM08 score | RUL 截断 125；score 函数对"预测过大"惩罚更重 |

**评测协议（沿用现有）**：
- 一问一答：`next_batch → 决策 → submit → 下一批`
- 所有指标返回 utility（越大越好），回归取负号；门控按 `margin_mode` 归一化
- 冷启动只训练第 1 批，之后逐批推进
- **不得**：用未来批训练当前批；用 test 标签调参；当批训练后当批验收

---

## 9. 下载脚本设计

### 9.1 共同约定

```
python scripts/download_<name>.py --dest data/<name>/raw
    1. 若已存在且 SHA256 匹配 → 跳过
    2. 下载到 _cache/<file>.part，支持断点续传（Range）
    3. 校验 SHA256（首次下载则记录到 checksums.sha256）
    4. 解压到 raw/（保留原始压缩包）
    5. 打印文件清单 + 大小 + 哈希
```

- **不把大数据提交进 Git**：`data/` 已在 `.gitignore`
- **不在脚本里硬编码任何凭证**
- 失败必须**明确报错并给出人工下载指引**，不能静默跳过

### 9.2 三个脚本的差异

| 脚本 | 能否全自动 | 设计 |
|---|---|---|
| `download_gas.py` | ✅ 可以 | 直链 + 校验 + 解压 |
| `download_cmapss.py` | ⚠️ 部分 | data.gov 资源直链需从页面解析；解析失败则打印候选镜像（NASA PCoE / Zenodo）要求人工下载 |
| `download_xjtu_sy.py` | ❌ **不能** | 官方只给网盘链接，**脚本只能做"校验 + 解压 + 目录规范化"**，下载必须人工。脚本要打印明确的获取步骤与预期目录结构 |

**这是必须提前说清的一件事**：XJTU-SY 无法做到"一键复现"。复现说明里要写明"需人工从官方网盘下载，下载后运行 `prepare_xjtu_sy.py` 校验"。

### 9.3 checksums 格式

```
<sha256>  data/gas/raw/Dataset.zip
<sha256>  data/gas/processed/batch_01.npz
...
```
`prepare_*.py` 完成后自动追加 processed 产物的哈希，保证"同输入 → 同产物"。

---

## 10. 预计需要修改 / 新增的文件清单

| # | 文件 | 类型 | 工作量 |
|---|---|---|---|
| 1 | `evolver/adapters/sources.py` | 新增 | 大（三个数据集类 + 统一接口） |
| 2 | `evolver/adapters/bridge.py` | 新增 | 小 |
| 3 | `evolver/adapters/dataset.py` | 修改 | 小（注册新数据集名） |
| 4 | `evolver/adapters/model.py` | 修改 | 中（补回归 GBDT） |
| 5 | `scripts/download_gas.py` | 新增 | 小 |
| 6 | `scripts/download_cmapss.py` | 新增 | 中 |
| 7 | `scripts/download_xjtu_sy.py` | 新增 | 小（只做校验/解压） |
| 8 | `scripts/prepare_cmapss.py` | 新增 | 大（寿命分段、RUL、常量列剔除） |
| 9 | `scripts/prepare_xjtu_sy.py` | 新增 | 大（流式特征提取） |
| 10 | `scripts/prepare_gas.py` | 修改 | 小（加哈希） |
| 11 | `scripts/calibrate.py` | 新增 | 中（阈值标定，原 B5） |
| 12 | `data/{gas,cmapss,xjtu_sy}/README_DATA.md` | 新增 | 中 |
| 13 | `checksums.sha256` | 新增 | 自动生成 |
| 14 | `tests/test_sources.py` | 新增 | 中 |
| 15 | `config.py` | 修改 | 小（各数据集阈值预设） |

---

## 待确认事项（我不知道，不能猜）

1. **XJTU-SY 的确切文件格式与体积**：README 未描述 CSV 结构，需下载后确认（现按"32768×2、25.6kHz"惯例处理，**未经核实**）
2. **C-MAPSS 的真实直链**：data.gov 页面有 `CMAPSSData.zip` 资源，但 API id 查询 404；Zenodo 镜像 API 返回 403。需从页面按钮抓取或人工确认
3. **XJTU-SY 的许可证**：README 未声明，仅要求引用。对外提交需谨慎
4. **官方 A/B 包与这三个公开集的对应关系**：宣讲只是口头点名，未公开映射
5. **官方评测是否给当批标签**：宣讲说"总能拿到标签"，但未写入规则 PDF。这直接决定门控是否合法（见 L4）
6. **各数据集的 PSI 阈值**：必须用 `calibrate.py` 实测标定，现在无法给数值

---

## 下一步建议（等你确认后执行）

1. 先做 **Gas 规范化**（数据已有，最快见效）：`sources.py` + `bridge.py` + `README_DATA.md` + 哈希清单
2. 再做 **C-MAPSS**（能验证回归链路，且下载可控）
3. 最后做 **XJTU-SY**（下载不可自动化，体积最大，需任务转换设计）
4. 全程用 `holdout` 批次，**不拿 test 调参**
