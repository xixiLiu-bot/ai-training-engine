# RELEASE NOTES · release-candidate-0

> 冻结时间：2026-11-14 21:12 (+08:00)
> 版本标识：`release-candidate-0`（见同目录 `VERSION`）
> 本轮**冻结当前代码，未修改算法、阈值、测试或任何源文件**。
> 校验清单：`SHA256SUMS.txt`（不含 `.env`、密钥、`data/gas` 原始数据、临时产物）

---

## 一、版本事实（四句话，请勿改写）

1. **测试：272 例通过。**
   两种数据环境都跑过：无 `data/gas` → `272 tests OK (skipped=43)`；
   有 `data/gas` → `272 tests OK (skipped=7)`。**0 error / 0 failure。**

2. **Gas 是公开替代数据，不是官方数据包。**
   用的是 UCI Gas Sensor Array Drift（dataset 224），13,910 样本 / 128 特征 / 6 类 / 10 批。
   数据身份标记为 `public_substitute`，`is_official = False`。

3. **没有任何官方成绩。**
   官方公开集尚未下发，隐藏测试流不可本地运行（赛题红线）。
   证据文件里 `provenance.official_test_result = null`。
   本仓库不含任何官方评测分数，也不含任何官方数据文件。

4. **Docker 未验证。**
   `Dockerfile` 与 `.dockerignore` 存在且内容完整，但**从未构建、从未跑过 `--network none`**
   （开发机没有 docker）。不要声称镜像已验证。

补充一条：**SwanLab 曾在本机验证（`Upload complete` + 网页端可见），但当前环境未复验**
（缺 SDK 与 API Key）。它是可选依赖，不影响闭环。

---

## 二、文件清单（113 个文件，不含 `__pycache__`）

| 目录 | 文件数 | 体积 | 说明 |
|---|---|---|---|
| `evolver/` | 37 | 218.2 KB | 核心包：loop / adapters / engine / logs / memory / protocol / ledger |
| `docs/` | 18 | 1,547.5 KB | 赛题规则、技术简报、证据（`docs/evidence/calibration/`）、汇报 PPT |
| `tests/` | 16 | 133.5 KB | 全量测试，272 例 |
| `tools/` | 12 | 161.7 KB | 文档/插图生成脚本（不参与运行） |
| `scripts/` | 11 | 155.5 KB | 运行、标定、体检、扫描、转换入口 |
| `assets/` | 3 | 569.9 KB | 文档插图 |
| `data/` | 2 | 4.2 KB | **只有两份目录契约**：`cmapss/README_DATA.md`、`xjtu_sy/README_DATA.md` |
| `skills/` | 1 | 1.0 KB | 技能说明 |
| 根目录 | 15 | 62.3 KB | 见下 |

根目录文件：

```
VERSION                     release-candidate-0（本文件）
RELEASE_NOTES.md            本文件
README.md                   主文档（含数据准备、依赖分级、证据查看）
SHA256SUMS.txt              本次冻结的完整校验清单
requirements.txt            依赖（>= 约束）
requirements.lock           依赖（版本锁定，镜像用）
requirements-dev.txt        开发/可选依赖（swanlab、pytest）
Dockerfile                  镜像定义（未构建验证）
.dockerignore
.env.example                配置模板（纯占位符，无真实凭据）
.gitignore                  只忽略具体数据目录，保留两份契约
run_offline.sh / run.bat    一键运行脚本
00-先读我.md                 交接入口
DATA_INTAKE_REVIEW.md       数据源调研（UCI / XJTU-SY / C-MAPSS）
PACKAGE_MANIFEST.txt        ⚠️ 历史产物：上一轮 zip 打包的清单，已被本文件取代，
                            保留仅为追溯，不要按它核对文件
```

---

## 三、测试结果

| 场景 | 命令 | 结果 |
|---|---|---|
| 干净 clone（无 `data/`、无 `out/`） | `python -m unittest discover -s tests -t .` | **Ran 272 tests → OK (skipped=43)** |
| 有 `data/gas` | 同上 | **Ran 272 tests → OK (skipped=7)** |

skip 全是"缺数据/缺可选依赖"，没有一条是 error：

| 数量 | 原因 |
|---|---|
| 29 | 本地没有气体数据（`data/gas`） |
| 4 | 需要 `data/synthetic`（先跑 `scripts/make_data.py`） |
| 4 | 需要气体数据 + sklearn / numpy 后端 |
| 2 | 需要 `data/gas` |
| 2 | `checksums.sha256` 不存在（策略上不入库，见下） |
| 1 | `swanlab` 未安装 |
| 1 | 需要 scikit-learn + `data/gas` |

> `checksums.sha256` 刻意不入库：它校验 `data/gas/processed/*.npz`，而数据不进仓库。
> 拿到数据后用 `python scripts/gen_checksums.py --write` 生成。**不在仓库放伪校验和。**

---

## 四、实验结果（可复现）

### 1. 合成数据（10 批，零外部数据依赖）

```
平均分   基线 0.509  系统 0.868
决策    动作 5 次 | 上线 3 次 | 回滚 2 次
协议    10 轮 | 部署 4 | 维持 6 | 违规 0
```

### 2. Gas 公开替代数据（10 批，默认阈值 1.2 / 2.5）

```
平均分   基线 0.320  系统 0.669
决策    动作 6 次 | 上线 3 次 | 回滚 3 次
协议    10 轮 | 部署 4 | 维持 6 | 违规 0
```

### 3. 阈值标定证据（`docs/evidence/calibration/`）

| 项 | 值 |
|---|---|
| 分段 | calibration b02–b04 / validation b05–b08 / pre-holdout b02–b09 / holdout b10 |
| **默认阈值（active）** | **1.2 / 2.5**（`changed_default = false`） |
| 实验候选 | 0.6 / 4.0 |
| 回退原因 | `single_batch_evidence`（优势只来自 b02 一个批次） |
| 网格候选 | 20 个 |
| b10 既有记录 | 候选配置 0.6/4.0 跑过一次：系统 0.3946 / 基线 0.2897 / 增益 +0.1049（n=1） |

**b10 纪律**：b10 是 `public_holdout`，**不是官方测试集**。从未用于选阈值、选模型、改动作或调门控；
默认配置 1.2/2.5 **在 b10 上没有数字**（守"只跑一次"，未补跑）。详见
`docs/evidence/calibration/README.md` 第七节。

---

## 五、状态标记

| 项 | 状态 | 说明 |
|---|---|---|
| 公开 Gas 数据 | ✅ **已接入（但不进仓库）** | `public_substitute`；他人需按 README 用 `prepare_gas.py` 生成 |
| 官方数据 | ❌ **未提供** | `data/official/` 不存在；无官方成绩 |
| matplotlib | 🟡 **可选依赖（缺失路径已验证）** | 只画 `curve.png`；缺失→警告+跳过，不进 `errors`。**真实出图未验证**（本机装不上） |
| SwanLab | 🟡 **可选依赖 / 当前环境未复验** | 曾在本机验证；本次无 SDK/Key |
| `checksums.sha256` | ⛔ **被数据缺失阻塞** | 随数据包生成，不提交伪校验和 |
| Docker | ❌ **未验证** | 文件存在，从未构建 |
| C-MAPSS / XJTU-SY | ❌ **未下载** | 只有目录契约，**不得声称已接入真实数据** |
| 官方测试流 | ⛔ **不可本地运行** | 赛题红线：禁止获取隐藏测试集 |

---

## 六、未完成事项

| # | 项 | 阻塞原因 |
|---|---|---|
| 1 | 官方公开集接入（`OfficialSource` 真实字段适配） | 数据未下发 |
| 2 | 官方测试流一次性评测 | 只能平台跑，本地无数据 |
| 3 | Docker 构建 + `--network none` 自测 | 开发机无 docker |
| 4 | C-MAPSS / XJTU-SY 真实数据运行 | 未下载；XJTU-SY 无许可声明 |
| 5 | 真实 matplotlib 出图 | 依赖装不上（pypi 连接失败） |
| 6 | SwanLab 上传复验 | 本环境无 SDK/Key |
| 7 | 多 seed 在真实数据上的方差 | 仅合成数据跑过 |
| 8 | 官方评测所需的重跑申请流程 | 依赖 #1 #2 |

---

## 七、本 RC 相对上一状态的变更

本轮 **0 代码变更**（纯冻结 + 元数据）。

上一阶段累计变更（已包含在本 RC 内）：

| 阶段 | 内容 |
|---|---|
| 缺陷修复 | `fit_continue` 类别集变化不再崩溃（降级+留痕）；`prepare_xjtu` 空输入非零退出；`sweep` 三类不确定性分开报告 |
| 环境/纪律 | matplotlib 改为可选（warning 不进 errors）；README 补依赖分级；证据文档补 b10 纪律 |
| 打包完整性 | `.gitignore` 改为显式数据目录，使两份目录契约可提交 |
| 测试 | 268 → 272 例（新增 4 例绘图可选性测试） |

---

## 八、从零开始（完整版见 `README.md`）

```bash
pip install -r requirements.lock          # 运行时依赖
python scripts/make_data.py               # 合成数据（无需任何外部数据）
python scripts/run_pipeline.py --dataset synthetic --llm none --out out/syn
python -m unittest discover -s tests -t . # 272 例

# 用真实 Gas（可选）
# 从 https://archive.ics.uci.edu/dataset/224/gas 下载 → data/gas/raw/Dataset/
python scripts/prepare_gas.py
python scripts/gen_checksums.py --write
python scripts/run_pipeline.py --dataset gas --psi-low 1.2 --psi-mid 2.5 --llm none
```

---

## 九、使用本版本时的对外表述红线

- ❌ 不要说"已完成官方评测"或"官方得分 X"——**没有官方数据，也没有官方成绩**
- ❌ 不要说"镜像已验证"——Docker **从未构建**
- ❌ 不要说"在留出集上达到 0.3946 说明泛化好"——那是**候选配置的一次既有记录**，且默认配置未在该批评估
- ✅ 可以说"系统在 Gas 公开替代数据上 10 轮闭环，基线 0.320 → 系统 0.669，0 违规"
- ✅ 可以说"阈值标定做了时间顺序分段，默认值仍是 1.2/2.5，0.6/4.0 仅候选"
