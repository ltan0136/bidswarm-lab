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

下面是脚本输出原文（脚本里的二、三级标题在此改为四、五级标题）。

- A/A 池：`power_random_b192-20261029T193217Z`（proposer=random，R=100）
- ES 池：`power_es_b192-20261029T192857Z`（proposer=es，R=100）
- CI 方法与报告一致：百分位 bootstrap，10000 次重采样，固定种子 20261030
- 模拟：每个设计规模 2000 次模拟实验，从池中有放回抽取 replicate 级配对差（种子 12345）

- 一致性检查：`toy_experiment-20261030T040850Z` 的 60 个 selected 文件与 `toy_cooperative-20261030T041032Z` 中同名文件完全一致
- 一致性检查：`toy_experiment_random-20261030T040941Z` 的 60 个 selected 文件与 `toy_cooperative_random-20261030T041436Z` 中同名文件完全一致
- 一致性检查：`power_es_b192-20261029T192857Z` 的 200 个 selected 文件与 `toy_cooperative-20261030T041032Z` 中同名文件完全一致
- 一致性检查：`power_random_b192-20261029T193217Z` 的 200 个 selected 文件与 `toy_cooperative_random-20261030T041436Z` 中同名文件完全一致

#### 1. A/A 检查（random：single 与 independent 在分布上等价）

- 测试分配对差 independent − single：n=100，均值 -0.313，标准差 3.581，标准误 0.358，95% bootstrap CI [-1.009, +0.390]
- 符号翻转置换检验（均值 = 0）：p = 0.382
- 90% CI [-0.901, +0.271]：等价性（TOST 形式）在 ±0.90 分的界值内成立
- 训练集最终 best-so-far 配对差：n=100，均值 -0.192，标准差 2.461，标准误 0.246，95% bootstrap CI [-0.669, +0.281]

#### 2. 校准：真实差为 0 时的假阳性率（名义 0.05）

| R | bootstrap CI 不含 0（2000 次模拟） | 置换检验 p ≤ 0.05（500 次模拟） |
|---|---|---|
| 20 | 0.076 ± 0.006 | 0.060 ± 0.011 |
| 50 | 0.052 ± 0.005 | 0.036 ± 0.008 |
| 100 | 0.064 ± 0.005 | 0.068 ± 0.011 |

模拟实验从去中心化的 A/A 配对差中有放回抽取，因此真实效应为 0；± 为蒙特卡洛标准误。

#### 3. ES 下的功效

##### independent − single（`power_es_b192-20261029T192857Z`）

- 观测配对差：n=100，均值 +0.669，标准差 5.779，标准误 0.578，95% bootstrap CI [-0.432, +1.822]
- 符号翻转置换检验：p = 0.250
- single 选定策略平均测试分 111.986（用于把 MDE 换算成相对值）

| R | δ=0 拒绝率 | 功效 δ=1 | δ=2 | δ=3 | δ=5 | MDE（80% 功效） | 正态近似 MDE |
|---|---|---|---|---|---|---|---|
| 20 | 0.086 | 0.15 | 0.39 | 0.68 | 0.97 | 3.51（3.1%） | 3.62 |
| 50 | 0.065 | 0.26 | 0.73 | 0.96 | 1.00 | 2.19（2.0%） | 2.29 |
| 100 | 0.052 | 0.44 | 0.94 | 1.00 | 1.00 | 1.59（1.4%） | 1.62 |

##### cooperative − independent（`toy_cooperative-20261030T041032Z`）

- 观测配对差：n=100，均值 +0.182，标准差 3.343，标准误 0.334，95% bootstrap CI [-0.496, +0.809]
- 符号翻转置换检验：p = 0.593
- independent 选定策略平均测试分 112.655（用于把 MDE 换算成相对值）

| R | δ=0 拒绝率 | 功效 δ=1 | δ=2 | δ=3 | δ=5 | MDE（80% 功效） | 正态近似 MDE |
|---|---|---|---|---|---|---|---|
| 20 | 0.090 | 0.35 | 0.71 | 0.94 | 1.00 | 2.32（2.1%） | 2.09 |
| 50 | 0.066 | 0.54 | 0.96 | 1.00 | 1.00 | 1.43（1.3%） | 1.32 |
| 100 | 0.062 | 0.81 | 1.00 | 1.00 | 1.00 | 0.99（0.9%） | 0.94 |

δ 为加在配对差上的真实效应（分）；MDE 为模拟功效首次达到 0.8 的 δ（步长 0.01）。

#### 4. 阳性对照：共享场景的两组运行之间的配对效应

| 对照 | 组 | 均值差 | 差的标准差 | 95% CI（R=100） | 置换 p | R=20 时检出率 |
|---|---|---|---|---|---|---|
| ES 预算 192 vs 32 | selected:single | +5.298 | 5.831 | [+4.185, +6.449] | < 0.0001 | 0.97 |
| ES 预算 192 vs 32 | selected:independent | +4.522 | 4.310 | [+3.714, +5.365] | < 0.0001 | 1.00 |
| ES 预算 192 vs 96 | selected:single | +1.702 | 3.316 | [+1.053, +2.373] | < 0.0001 | 0.70 |
| ES 预算 192 vs 96 | selected:independent | +1.377 | 2.747 | [+0.855, +1.911] | < 0.0001 | 0.69 |
| ES vs random（预算 192） | selected:single | -0.143 | 5.025 | [-1.184, +0.803] | 0.7803 | 0.09 |
| ES vs random（预算 192） | selected:independent | +0.839 | 4.112 | [+0.035, +1.615] | 0.0442 | 0.24 |

## 4. 结论

- **G1 正确性：通过。** random 下 `independent − single` = −0.313，95% CI [−1.009, +0.390]，置换 p = 0.382；
  90% CI 落在 ±0.90 分内，即两种模式在 ±0.9 分（约 0.8%）的界值内等价。训练集 best-so-far 同样无差异。
  R = 20 的运行与同配置 R = 100 运行的前 20 个 replicate 完全一致；加入 cooperative 不改变 single / independent 的结果。
- **G2 敏感性：通过。** 预算的剂量反应单调且可检出：192 vs 32 为 +4.5 ~ +5.3 分（R = 20 时检出率 97% ~ 100%），
  192 vs 96 为 +1.4 ~ +1.7 分（R = 100 时置换 p < 0.001，但 R = 20 时检出率只有约 70%）。
- **G3 功效：** 80% 功效下的 MDE——
  `independent − single`：R = 20 约 3.5 分（3.1%），R = 50 约 2.2 分，R = 100 约 1.6 分（1.4%）；
  `cooperative − independent`（共用随机数流，配对差标准差 3.3 而不是 5.8）：R = 20 约 2.3 分，R = 50 约 1.4 分，
  **R = 100 约 1.0 分（0.9%）**。MDE 大致按 1/√R 缩小。
- **校准注意事项：** R = 20 时报告使用的百分位 bootstrap CI 偏窄，名义 5% 的假阳性率实测约 7.6% ~ 9%；
  符号翻转置换检验约 6.0%（500 次模拟，蒙特卡洛误差约 ±1 个百分点）。R ≥ 50 时两者大致在 3.6% ~ 6.8% 之间。
  因此报告从 v0.2 起同时给出置换检验 p 值、在比较多于一个时做 Holm 校正，并在 R < 30 时提示。
- **一个需要注意的现象：** 预算 192 时 ES 与 random 的测试分几乎没有差别（single −0.14，n.s.；independent +0.84，p = 0.044）。
  在这个 8 维 toy 问题上，48 个随机点已经能找到接近最好的策略：**搜索的"量"影响明显，搜索"方法"的差别很小**。
  迁移属于"方法"层面的改动，因此在 toy 上即使有效应也可能很小，toy 上的阴性结果不能推广到 AuctionNet。

**决定：** G1、G2 通过，进入信息交换的实验。比较迁移效果使用 **R = 100**（`configs/toy_cooperative.toml`，
`cooperative − independent` 的 MDE 约 1.0 分），并配一个 random A/A 对照（`configs/toy_cooperative_random.toml`）。
R = 20 的配置只用于冒烟测试或检查大效应。结果见 [TOY_RESULTS.md](TOY_RESULTS.md)。

## 5. 复现

```bash
for c in power_es_b192 power_random_b192 power_es_b96 power_es_b32; do .venv/bin/python -m bidswarm run --config configs/power/$c.toml --out runs/power && .venv/bin/python -m bidswarm evaluate --run-dir "$(ls -d runs/power/$c-2* | tail -1)"; done
```

```bash
for c in toy_cooperative toy_cooperative_random; do .venv/bin/python -m bidswarm run --config configs/$c.toml && .venv/bin/python -m bidswarm evaluate --run-dir "$(ls -d runs/$c-2* | tail -1)"; done
```

```bash
E192=$(ls -d runs/power/power_es_b192-*); R192=$(ls -d runs/power/power_random_b192-*); E96=$(ls -d runs/power/power_es_b96-*); E32=$(ls -d runs/power/power_es_b32-*); COOP=$(ls -d runs/toy_cooperative-2*)
.venv/bin/python scripts/power_analysis.py --aa $R192 --es $E192 --power $COOP:cooperative:independent \
  --control "ES 预算 192 vs 32=$E192,$E32" --control "ES 预算 192 vs 96=$E192,$E96" --control "ES vs random（预算 192）=$E192,$R192"
```

四个数据池在 4 进程上共约 10 分钟，两个 cooperative 实验各约 4 ~ 5 分钟，分析脚本约 1 ~ 2 分钟。
结果只依赖配置与种子，与进程数无关。
