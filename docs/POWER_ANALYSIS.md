# 统计功效分析（toy）

> 本文所有数字来自内置 toy 环境（backend=toy，purpose=framework_smoke_test），只用来校准实验框架本身，
> **不代表 AuctionNet 上的表现，也不代表任何真实广告收益**。

## 1. 目的与判据

在评估信息交换（`cooperative` 模式）之前，需要先确认实验框架：

- **G1 正确性**：random proposer 下 single 与 independent 在分布上等价（A/A）。要求 R = 100 时
  `independent − single` 的 95% CI 包含 0、置换检验 p > 0.05，并且 R = 20 的运行恰好是 R = 100 数据池的前 20 个 replicate。
- **G2 敏感性**：框架能检出方向已知的真实效应。阳性对照是训练预算：同为 ES，预算 192 应不差于 96 和 32。
  要求 R = 100 时两个预算对照的 95% CI 都不含 0 且为正。
- **G3 功效**：给出 R = 20 / 50 / 100 下 80% 功效对应的最小可检出效应（MDE），据此确定迁移实验的 R。

G1、G2 通过才进入信息交换的实验。判据在看过 R = 20 的初步结果之后、分析 R = 100 数据池之前确定。

## 2. 数据与方法

- **数据池**（`configs/power/*.toml`，均为 R = 100，与 `toy_experiment.toml` 同 seed，因此同一 replicate 的场景完全相同，
  可以跨运行按 replicate 配对）：

  | 配置 | proposer | 训练预算 |
  |---|---|---|
  | `power_es_b192.toml` | es | 192 |
  | `power_random_b192.toml` | random | 192 |
  | `power_es_b96.toml` | es | 96 |
  | `power_es_b32.toml` | es | 32 |

- **指标**：每个 replicate 先对 16 个测试场景求 `score_expected` 均值，再取两组的配对差。
- **校准**：把 A/A 配对差去中心化（真实效应为 0），从中有放回抽取 R 个，按报告完全相同的程序计算 95% 百分位 bootstrap CI
  （10000 次、固定种子），统计"CI 不含 0"的比例；置换检验（符号翻转）用同样的方式统计 p ≤ 0.05 的比例。
- **功效**：把 ES 下的配对差去中心化后加上真实效应 δ（平移模型）。百分位 CI 对平移等变，因此每次模拟实验的 CI 只需计算一次。
  MDE 为模拟功效首次达到 0.8 的 δ；同时给出正态近似 `(z₀.₉₇₅ + z₀.₈)·σ/√R`。
- **阳性对照的"R = 20 时检出率"**：从真实（未去中心化）的配对差中有放回抽取 20 个，统计 CI 不含 0 的比例。
- **cooperative − independent**：功效取自 `toy_cooperative.toml`（R = 100）运行。它的 single / independent 分支与
  `power_es_b192` 数据池逐字一致（见下方一致性检查），cooperative 与 independent 共用随机数流，因此配对差方差更小。
- **局限**：模拟把 100 个 replicate 的经验分布当作总体；结论只适用于 toy、预算 192、ES、`score_expected` 这一设置；
  平移模型假设效应不改变配对差的分布形状。

## 3. 结果（`scripts/power_analysis.py` 原始输出）

下面是 2026-11-06 重跑后的脚本输出原文（toy 改为只对曝光收费之后；脚本里的二、三级标题在此改为四、五级标题）。运行环境 Python 3.11.18、numpy 2.4.6、4 进程，代码提交 `ea1f628`。

- A/A 池：`power_random_b192-20261106T100540Z`（proposer=random，R=100）
- ES 池：`power_es_b192-20261106T100321Z`（proposer=es，R=100）
- CI 方法与报告一致：百分位 bootstrap，10000 次重采样，固定种子 20261030
- 模拟：每个设计规模 2000 次模拟实验，从池中有放回抽取 replicate 级配对差（种子 12345）

- 一致性检查：`toy_experiment-20261106T101023Z` 的 60 个 selected 文件与 `toy_cooperative-20261106T101147Z` 中同名文件完全一致
- 一致性检查：`toy_experiment_random-20261106T101105Z` 的 60 个 selected 文件与 `toy_cooperative_random-20261106T101640Z` 中同名文件完全一致
- 一致性检查：`power_es_b192-20261106T100321Z` 的 200 个 selected 文件与 `toy_cooperative-20261106T101147Z` 中同名文件完全一致
- 一致性检查：`power_random_b192-20261106T100540Z` 的 200 个 selected 文件与 `toy_cooperative_random-20261106T101640Z` 中同名文件完全一致

#### 1. A/A 检查（random：single 与 independent 在分布上等价）

- 测试分配对差 independent − single：n=100，均值 -1.074，标准差 4.322，标准误 0.432，95% bootstrap CI [-1.949, -0.224]
- 符号翻转置换检验（均值 = 0）：p = 0.016
- 90% CI [-1.806, -0.363]：等价性（TOST 形式）在 ±1.81 分的界值内成立
- 训练集最终 best-so-far 配对差：n=100，均值 -0.077，标准差 2.429，标准误 0.243，95% bootstrap CI [-0.558, +0.392]

#### 2. 校准：真实差为 0 时的假阳性率（名义 0.05）

| R | bootstrap CI 不含 0（2000 次模拟） | 置换检验 p ≤ 0.05（500 次模拟） |
|---|---|---|
| 20 | 0.065 ± 0.005 | 0.044 ± 0.009 |
| 50 | 0.051 ± 0.005 | 0.036 ± 0.008 |
| 100 | 0.060 ± 0.005 | 0.044 ± 0.009 |

模拟实验从去中心化的 A/A 配对差中有放回抽取，因此真实效应为 0；± 为蒙特卡洛标准误。

#### 3. ES 下的功效

##### independent − single（`power_es_b192-20261106T100321Z`）

- 观测配对差：n=100，均值 +1.484，标准差 6.110，标准误 0.611，95% bootstrap CI [+0.333, +2.712]
- 符号翻转置换检验：p = 0.017
- single 选定策略平均测试分 128.884（用于把 MDE 换算成相对值）

| R | δ=0 拒绝率 | 功效 δ=1 | δ=2 | δ=3 | δ=5 | MDE（80% 功效） | 正态近似 MDE |
|---|---|---|---|---|---|---|---|
| 20 | 0.072 | 0.12 | 0.33 | 0.64 | 0.99 | 3.58（2.8%） | 3.83 |
| 50 | 0.064 | 0.23 | 0.67 | 0.96 | 1.00 | 2.29（1.8%） | 2.42 |
| 100 | 0.051 | 0.40 | 0.93 | 1.00 | 1.00 | 1.65（1.3%） | 1.71 |

##### cooperative − independent（`toy_cooperative-20261106T101147Z`）

- 观测配对差：n=100，均值 -0.593，标准差 4.685，标准误 0.469，95% bootstrap CI [-1.566, +0.268]
- 符号翻转置换检验：p = 0.217
- independent 选定策略平均测试分 130.368（用于把 MDE 换算成相对值）

| R | δ=0 拒绝率 | 功效 δ=1 | δ=2 | δ=3 | δ=5 | MDE（80% 功效） | 正态近似 MDE |
|---|---|---|---|---|---|---|---|
| 20 | 0.100 | 0.32 | 0.59 | 0.75 | 0.93 | 3.57（2.7%） | 2.94 |
| 50 | 0.081 | 0.42 | 0.78 | 0.96 | 1.00 | 2.11（1.6%） | 1.86 |
| 100 | 0.065 | 0.57 | 0.96 | 1.00 | 1.00 | 1.43（1.1%） | 1.31 |

δ 为加在配对差上的真实效应（分）；MDE 为模拟功效首次达到 0.8 的 δ（步长 0.01）。

#### 4. 阳性对照：共享场景的两组运行之间的配对效应

| 对照 | 组 | 均值差 | 差的标准差 | 95% CI（R=100） | 置换 p | R=20 时检出率 |
|---|---|---|---|---|---|---|
| ES 预算 192 vs 32 | selected:single | +5.097 | 6.479 | [+3.843, +6.383] | < 0.0001 | 0.95 |
| ES 预算 192 vs 32 | selected:independent | +5.592 | 5.528 | [+4.530, +6.707] | < 0.0001 | 1.00 |
| ES 预算 192 vs 96 | selected:single | +1.072 | 3.395 | [+0.396, +1.719] | 0.0018 | 0.40 |
| ES 预算 192 vs 96 | selected:independent | +1.703 | 3.328 | [+1.083, +2.381] | < 0.0001 | 0.78 |
| ES vs random（预算 192） | selected:single | -0.940 | 5.605 | [-2.069, +0.108] | 0.0940 | 0.14 |
| ES vs random（预算 192） | selected:independent | +1.619 | 4.171 | [+0.793, +2.421] | < 0.0001 | 0.46 |

## 4. 结论

以下是 2026-11-06 toy 改为只对曝光收费之后重跑的结论；此前（按获胜收费）的结论与数字见本文件的 git 历史，两者不能混合比较。

- **G1 正确性：名义上未通过，按偶然的假阳性处理。** random 下 `independent − single` = −1.074，95% CI [−1.949, −0.224]，
  置换 p = 0.016，不满足"CI 含 0 且 p > 0.05"的判据。两种模式在构造上等价（都是 48 个独立均匀样本、同一选择规则），
  训练集 best-so-far 的配对差为 −0.08（CI [−0.56, +0.39]），R = 20 / R = 100 的前缀一致性与 cooperative 不改变
  single / independent 的检查也全部通过；真实差为 0 时置换检验 p ≤ 0.016 的概率约 1.6%。因此把它当作偶然的假阳性，
  但它也说明：**在这个 toy 上，single 与 independent 之间约 1 ~ 1.5 分的测试分差，不能归因于搜索的组织方式。**
  （改动前同一检查为 −0.313，p = 0.382。）
- **G2 敏感性：通过。** 预算的剂量反应单调且可检出：192 vs 32 为 +5.1 ~ +5.6 分（R = 20 时检出率 95% ~ 100%），
  192 vs 96 为 +1.1 ~ +1.7 分（R = 100 时置换 p ≤ 0.002，R = 20 时检出率只有 40% ~ 78%）。
- **G3 功效：** 80% 功效下的 MDE——
  `independent − single`：R = 20 约 3.6 分（2.8%），R = 50 约 2.3 分，R = 100 约 1.7 分（1.3%）；
  `cooperative − independent`：R = 20 约 3.6 分，R = 50 约 2.1 分，**R = 100 约 1.4 分（1.1%）**。
  改为按曝光收费后，cooperative 与 independent 的配对差标准差从 3.3 升到 4.7，共用随机数流带来的方差缩减变小，
  R = 100 的 MDE 从约 1.0 分升到约 1.4 分。
- **校准：** 百分位 bootstrap CI 不含 0 的比例在 R = 20 / 50 / 100 时为 6.5% / 5.1% / 6.0%，符号翻转置换检验为
  4.4% / 3.6% / 4.4%（蒙特卡洛误差约 ±0.5 和 ±0.9 个百分点）。报告继续同时给出置换检验 p、Holm 校正，并在 R < 30 时提示。
- **ES vs random（预算 192）：** single −0.94（n.s.），independent +1.62（p < 0.0001）。方向不一致，结合 G1，
  搜索"方法"层面的差别在这个 toy 上仍然很小；搜索的"量"（训练预算）才有稳定、可检出的影响。

**决定：** 仍用 R = 100 比较迁移（`configs/toy_cooperative.toml`，`cooperative − independent` 的 MDE 约 1.4 分），
并保留 random A/A 对照（`configs/toy_cooperative_random.toml`）；解读 `independent − single` 时以 A/A 的波动幅度为参照。
R = 20 的配置只用于冒烟测试或检查大效应。结果见 [TOY_RESULTS.md](TOY_RESULTS.md)。

## 5. 复现

```bash
for c in power_es_b192 power_random_b192 power_es_b96 power_es_b32; do .venv/bin/python -m bidswarm run --config configs/power/$c.toml --out runs/power && .venv/bin/python -m bidswarm evaluate --run-dir "$(ls -d runs/power/$c-2* | tail -1)"; done
```

```bash
for c in toy_experiment toy_experiment_random toy_cooperative toy_cooperative_random; do .venv/bin/python -m bidswarm run --config configs/$c.toml && .venv/bin/python -m bidswarm evaluate --run-dir "$(ls -d runs/$c-2* | tail -1)"; done
```

```bash
E192=$(ls -d runs/power/power_es_b192-*); R192=$(ls -d runs/power/power_random_b192-*); E96=$(ls -d runs/power/power_es_b96-*); E32=$(ls -d runs/power/power_es_b32-*)
COOP=$(ls -d runs/toy_cooperative-2*); COOPR=$(ls -d runs/toy_cooperative_random-2*); TE=$(ls -d runs/toy_experiment-2*); TER=$(ls -d runs/toy_experiment_random-2*)
.venv/bin/python scripts/power_analysis.py --aa $R192 --es $E192 --power $COOP:cooperative:independent \
  --control "ES 预算 192 vs 32=$E192,$E32" --control "ES 预算 192 vs 96=$E192,$E96" --control "ES vs random（预算 192）=$E192,$R192" \
  --prefix-check $TE:$COOP --prefix-check $TER:$COOPR --prefix-check $E192:$COOP --prefix-check $R192:$COOPR
```

四个数据池在 4 进程上共约 7 分钟，两个 R = 20 实验各约 1 分钟，两个 cooperative 实验各约 4 ~ 5 分钟，分析脚本约 1.5 分钟。
结果只依赖配置与种子，与进程数无关。
