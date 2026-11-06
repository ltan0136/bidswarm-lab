# toy 实验记录

> 本文记录在内置 toy 环境上实际跑过的实验及其关键数字（backend=toy，purpose=framework_smoke_test）。
> **这些结果不代表 AuctionNet 上的表现，也不代表任何真实广告收益。** 运行目录 `runs/` 不进版本库；
> 所有数字都可以用文末命令按相同配置与种子逐字重现（与进程数无关）。

- **2026-11-06 全部重跑。** toy 与内核改为只对曝光的获胜收费（与 AuctionNet 一致，见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) 第 10 节），
  所有得分都变了（测试分从约 112 升到约 129，因为同样的预算能买到更多曝光）。此前按获胜收费的数字见本文件的 git 历史，**不能与下面的数字混合比较**。
- 运行环境：Python 3.11.18，numpy 2.4.6，4 进程（spawn）；代码提交 `ea1f628`。
- 一致性：R = 20 的运行与同配置 R = 100 运行的前 20 个 replicate 逐字相同，`power_es_b192` / `power_random_b192` 的 single / independent
  与 `toy_cooperative` / `toy_cooperative_random` 中的同名结果逐字相同（[POWER_ANALYSIS.md](POWER_ANALYSIS.md) 第 3 节的一致性检查）。
- 主指标：`score_expected`（每个 replicate 先对 16 个测试场景取均值）。表中数字取自各运行的 `report.md` 与 `scripts/compare_runs.py`。

## 1. 三模式对照，R = 20（`toy_experiment.toml` 与 `toy_experiment_random.toml`）

| proposer | 比较 | 差值均值 | 95% bootstrap CI | 置换 p | Holm 校正 p |
|---|---|---|---|---|---|
| es | independent − single | +1.571 | [−1.043, +4.226] | 0.270 | 0.270 |
| es | cooperative − independent | −2.043 | [−3.428, −0.770] | 0.006 | 0.012 |
| random | independent − single | −1.794 | [−3.932, +0.451] | 0.131 | 0.263 |
| random | cooperative − independent | 0.000 | [0.000, 0.000] | 1.000 | 1.000 |

R = 20 时 `cooperative − independent` 显著为负（Holm p = 0.012），但这 20 个 replicate 正是第 3 节 R = 100 运行的前 20 个，
R = 100 时差值为 −0.59 且不显著。这是 R = 20 只适合冒烟测试的一个具体例子：小样本下的显著结果不可靠。

## 2. 统计功效数据池（R = 100）

见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md)。要点：

- **A/A 检查名义上未通过**：random 下 `independent − single` = −1.07（95% CI [−1.95, −0.22]，p = 0.016），而两种模式在构造上等价，
  按偶然的假阳性处理；但它说明这个 toy 上 single 与 independent 之间约 1 ~ 1.5 分的差不能归因于搜索的组织方式。
- 预算阳性对照可检出（192 vs 32：+5.1 ~ +5.6 分；192 vs 96：+1.1 ~ +1.7 分）。
- R = 100 时 MDE 约 1.7 分（independent − single）和 1.4 分（cooperative − independent）。

## 3. 迁移实验（`toy_cooperative.toml`，ES，R = 100）

三种模式同为 ES、每个 (replicate, mode) 192 个训练 episode；cooperative 为环形迁移、`exchange_interval = 1`、N = 4。

**测试分（选定策略）**

| 模式 | score_expected 均值 ± 标准差 |
|---|---|
| single | 128.884 ± 8.800 |
| independent | 130.368 ± 8.503 |
| cooperative | 129.775 ± 9.422 |

**配对比较（测试分，主指标）**

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p | Holm 校正 p |
|---|---|---|---|---|
| independent − single | +1.484 | [+0.333, +2.712] | 0.017 | 0.034 |
| cooperative − independent | −0.593 | [−1.566, +0.268] | 0.217 | 0.217 |

**训练集 best-so-far（最终检查点，配对差）**

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p |
|---|---|---|---|
| independent − single | +0.064 | [−0.763, +0.944] | 0.883 |
| cooperative − independent | +0.381 | [−0.060, +0.813] | 0.090 |

**迁移统计**：共 1199 次送达（最多可能 100 × 5 轮 × 4 = 2000 次；最后一轮后不迁移，同一张卡不重复送达）。
由 `candidates.jsonl` 计算：cooperative 的 4800 个候选中有 2014 个（42%）的父代来自其他搜索者。
预算账本 used == planned，失败 0，超时 0。三种模式的"验证分 − 测试分"约为 2.1 ~ 2.6 分（2.138 / 2.489 / 2.565）。

## 4. 迁移管线的 A/A 对照（`toy_cooperative_random.toml`，random，R = 100）

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p | Holm 校正 p |
|---|---|---|---|---|
| independent − single | −1.074 | [−1.949, −0.224] | 0.016 | 0.032 |
| cooperative − independent | 0.000 | [0.000, 0.000] | 1.000 | 1.000 |

cooperative 与 independent 共用随机数流，random 又不使用迁移来的候选，因此两者的候选逐字相同：
903 次送达、外来父代 0 个，配对差恰好为 0。这比"CI 包含 0"更强地说明迁移管线没有改动预算或候选。
`independent − single` 的显著差值是第 2 节所说的 A/A 假阳性（random 下两种模式在分布上等价）。

## 5. 解读

- **迁移管线本身没有引入偏差**：random 下 cooperative 与 independent 逐字相同。
- **迁移在训练集上略好，测试集上没有可分辨的差异**：cooperative − independent 的训练 best-so-far +0.38（p = 0.090），
  测试分 −0.59，95% CI [−1.57, +0.27]。R = 100 时本设计的 MDE 约 1.4 分，因此 toy 上迁移的测试分效应即使存在，也大概率小于约 1.4 分。
- **`independent − single` 的显著结果不能当真**：ES 下 +1.48（Holm p = 0.034，报告的自动结论判为"高于"），但同样设计的 random A/A
  给出 −1.07（p = 0.016），两种模式在那里构造上等价；两者幅度相近、方向相反，说明这个量级的差在 toy 上落在噪声范围内。
- **搜索的"量"比"方法"重要**：训练预算 192 vs 32 稳定带来约 5 分，方法层面的改动（ES vs random、迁移、CMA-ES）都在约 1.6 分以内且方向不一致。
- toy 上的结论不能推广到 AuctionNet；AuctionNet 后端已接入，第一版实验设计见 `configs/auctionnet_pilot.toml`（尚未运行）。

## 6. CMA-ES 对照（`toy_cooperative_cmaes.toml`，R = 100）

与第 3 节相同的种子、场景和预算（每个 (replicate, mode) 192 个训练 episode），只把 proposer 换成 CMA-ES
（默认种群 λ = 10，σ₀ = 0.3）。两次运行按 replicate 与模式配对，CMA-ES − ES，主指标 `score_expected`。

| 模式 | CMA-ES 测试分 | ES 测试分 | 差（测试） | 95% bootstrap CI | 置换 p | Holm p（3 个比较） | 差（训练 best-so-far） | 95% CI | 置换 p |
|---|---|---|---|---|---|---|---|---|---|
| single | 129.073 | 128.884 | +0.189 | [−0.897, +1.276] | 0.739 | 0.739 | −0.469 | [−1.246, +0.319] | 0.255 |
| independent | 128.754 | 130.368 | −1.615 | [−2.537, −0.740] | 0.0006 | 0.002 | −1.259 | [−1.796, −0.723] | < 0.0001 |
| cooperative | 128.853 | 129.775 | −0.922 | [−1.940, +0.125] | 0.089 | 0.177 | −1.362 | [−1.944, −0.771] | < 0.0001 |

CMA-ES 下 cooperative − independent 的测试分差为 +0.099（95% CI [−0.529, +0.753]，p = 0.770）。

**事后敏感性检查（种群 λ = 4）**：看到默认种群的结果后补跑，把 `toy_cooperative_cmaes.toml` 的 `population` 改为 4（每代 2 轮，
independent 每个搜索者约 3 代），其余不变。它不是预先登记的比较，只用来判断上表是否只是种群过大造成的。

| 模式 | CMA-ES(λ=4) 测试分 | ES 测试分 | 差（测试） | 95% CI | 置换 p | Holm p | 差（训练 best-so-far） | 95% CI | 置换 p |
|---|---|---|---|---|---|---|---|---|---|
| single | 129.304 | 128.884 | +0.420 | [−0.663, +1.503] | 0.455 | 0.909 | +0.156 | [−0.628, +0.973] | 0.710 |
| independent | 129.399 | 130.368 | −0.970 | [−1.701, −0.242] | 0.012 | 0.036 | −0.724 | [−1.238, −0.207] | 0.006 |
| cooperative | 129.849 | 129.775 | +0.074 | [−0.796, +1.037] | 0.882 | 0.909 | −0.559 | [−1.080, −0.035] | 0.038 |

与默认 λ = 10 相比，λ = 4 的 cooperative 训练 best-so-far 高 0.802 分（95% CI [+0.362, +1.258]）：代更短，迁移来的精英更早成为均值。
λ = 4 下 cooperative − independent 的测试分差为 +0.451（95% CI [−0.100, +1.011]，p = 0.115）。

**解读**：在这个预算下，CMA-ES 不是比 (1+λ)-ES 更强的对照。independent 每个搜索者只有 12 个候选，默认 λ = 10 时约 1.2 代，
非精英的 CMA-ES 在训练和测试上都落后于精英保留的 (1+λ)-ES（测试 −1.6 分，Holm p = 0.002）；single 持平。把种群降到 4 后差距缩小，
但 independent 仍落后约 1 分（Holm p = 0.036）。这与"方法排名依赖预算"的观察一致（[arXiv:2609.19799](https://arxiv.org/abs/2609.19799)）：
经典优化器在固定低维空间里常是强基线，但前提是预算够它跑若干代。比较 LLM proposer 时应同时报告 ES 与 CMA-ES，并在多个预算点上比较。

配对比较命令：

```bash
.venv/bin/python scripts/compare_runs.py runs/toy_cooperative_cmaes-<时间戳> runs/toy_cooperative-<时间戳>
```

## 7. 复现

```bash
for c in toy_experiment toy_experiment_random toy_cooperative toy_cooperative_random toy_cooperative_cmaes; do .venv/bin/python -m bidswarm run --config configs/$c.toml && .venv/bin/python -m bidswarm evaluate --run-dir "$(ls -d runs/$c-2* | tail -1)"; done
```

λ = 4 的检查：复制 `configs/toy_cooperative_cmaes.toml`，把 `name` 改为 `toy_cooperative_cmaes_pop4`、`population` 改为 4，同样运行。

R = 20 的两个实验各约 1 分钟，R = 100 的四个各约 4 ~ 5 分钟（4 进程）。功效数据池的命令见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md) 第 5 节。
