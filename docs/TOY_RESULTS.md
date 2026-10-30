# toy 实验记录

> 本文记录在内置 toy 环境上实际跑过的实验及其关键数字（backend=toy，purpose=framework_smoke_test）。
> **这些结果不代表 AuctionNet 上的表现，也不代表任何真实广告收益。** 运行目录 `runs/` 不进版本库；
> 所有数字都可以用文末命令按相同配置与种子逐字重现（与进程数无关）。

- 运行环境：Python 3.11.17，numpy 2.4.6，4 进程（spawn）。
- 代码版本：功效数据池（`configs/power/`）在 v0.1（提交 `45b6592`）上运行，其余实验在 v0.2 上运行。
  两版代码对 single / independent 给出逐字相同的结果：`toy_cooperative` 中这两个模式的 200 个 selected 文件
  与 `power_es_b192` 完全一致（random 对照同理，见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md) 的一致性检查）。
  所有报告都用 v0.2 的报告代码重建。
- 主指标：`score_expected`（每个 replicate 先对 16 个测试场景取均值）。表中数字取自各运行的 `report.md`。

## 1. 三模式对照，R = 20（`toy_experiment.toml` 与 `toy_experiment_random.toml`）

| proposer | 比较 | 差值均值 | 95% bootstrap CI | 置换 p | Holm 校正 p |
|---|---|---|---|---|---|
| es | independent − single | +0.864 | [−1.696, +3.389] | 0.532 | 1.000 |
| es | cooperative − independent | −0.058 | [−1.390, +1.187] | 0.929 | 1.000 |
| random | independent − single | −1.117 | [−2.653, +0.401] | 0.182 | 0.364 |
| random | cooperative − independent | 0.000 | [0.000, 0.000] | 1.000 | 1.000 |

R = 20 只够检出约 3.5 分（independent − single）或 2.3 分（cooperative − independent）以上的效应，均未观察到差异。
R = 20 的两次运行恰好是下面 R = 100 运行的前 20 个 replicate。

## 2. 统计功效数据池（R = 100）

见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md)。要点：A/A 等价（±0.9 分内）；预算阳性对照可检出；
R = 100 时 MDE 约 1.6 分（independent − single）和 1.0 分（cooperative − independent）；预算 192 下 ES 与 random 几乎无差别。

## 3. 迁移实验（`toy_cooperative.toml`，ES，R = 100）

三种模式同为 ES、每个 (replicate, mode) 192 个训练 episode；cooperative 为环形迁移、`exchange_interval = 1`、N = 4。

**测试分（选定策略）**

| 模式 | score_expected 均值 ± 标准差 |
|---|---|
| single | 111.986 ± 7.793 |
| independent | 112.655 ± 7.793 |
| cooperative | 112.837 ± 8.034 |

**配对比较（测试分，主指标）**

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p | Holm 校正 p |
|---|---|---|---|---|
| independent − single | +0.669 | [−0.432, +1.822] | 0.250 | 0.500 |
| cooperative − independent | +0.182 | [−0.496, +0.809] | 0.593 | 0.593 |

**训练集 best-so-far（最终检查点，配对差）**

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p |
|---|---|---|---|
| independent − single | −0.362 | [−1.072, +0.366] | 0.327 |
| cooperative − independent | +0.458 | [+0.034, +0.869] | 0.038 |

**迁移统计**：共 1183 次送达（最多可能 100 × 5 轮 × 4 = 2000 次；最后一轮后不迁移，同一张卡不重复送达）。
由 `candidates.jsonl` 计算：cooperative 的 4800 个候选中有 2052 个（43%）的父代来自其他搜索者。
预算账本 used == planned，失败 0，超时 0。三种模式的"验证分 − 测试分"都约为 2 分（2.062 / 2.019 / 2.167）。

## 4. 迁移管线的 A/A 对照（`toy_cooperative_random.toml`，random，R = 100）

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p |
|---|---|---|---|
| independent − single | −0.313 | [−1.009, +0.390] | 0.382 |
| cooperative − independent | 0.000 | [0.000, 0.000] | 1.000 |

cooperative 与 independent 共用随机数流，random 又不使用迁移来的候选，因此两者的候选逐字相同：
904 次送达、外来父代 0 个，配对差恰好为 0。这比"CI 包含 0"更强地说明迁移管线没有改动预算或候选。

## 5. 解读

- **迁移管线本身没有引入偏差**：random 下 cooperative 与 independent 逐字相同。
- **迁移让搜索在训练集上略好**：cooperative − independent 的训练 best-so-far 高 0.46 分（p = 0.038，未做多重比较校正）。
- **测试集上没有观察到可分辨的差异**：cooperative − independent 的测试分差 +0.18，95% CI [−0.50, +0.81]。
  本设计在 R = 100 时对 1 分的真实效应约有 81% 的检出率，因此 toy 上迁移的测试分效应即使存在，也大概率小于约 1 分。
- 与功效分析一致：在这个 toy 上，搜索方法层面的改动（ES vs random、迁移 vs 不迁移）带来的测试分差都在 1 分以内，
  而训练预算的变化带来数分的差别。toy 上的结论不能推广到 AuctionNet。

## 6. 复现

```bash
for c in toy_experiment toy_experiment_random toy_cooperative toy_cooperative_random; do .venv/bin/python -m bidswarm run --config configs/$c.toml && .venv/bin/python -m bidswarm evaluate --run-dir "$(ls -d runs/$c-2* | tail -1)"; done
```

R = 20 的两个实验各约 1 分钟，R = 100 的两个各约 4 ~ 5 分钟（4 进程）。功效数据池的命令见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md) 第 5 节。
