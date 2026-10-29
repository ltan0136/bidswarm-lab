# toy 实验记录

> 本文记录在内置 toy 环境上实际跑过的实验及其关键数字（backend=toy，purpose=framework_smoke_test）。
> **这些结果不代表 AuctionNet 上的表现，也不代表任何真实广告收益。** 运行目录 `runs/` 不进版本库；
> 所有数字都可以用文末命令按相同配置与种子逐字重现（与进程数无关）。

- 运行环境：Python 3.11.17，numpy 2.4.6，4 进程（spawn）。
- 代码版本：R = 20 实验与大部分功效数据池在 v0.1（提交 `45b6592`）上运行，exchange 实验在提交 `8484954` 上运行。
  两版代码给出的 single / independent 结果逐字一致：exchange 实验中这两个模式的 200 个 selected 文件和全部候选，
  与 `power_es_b192` 数据池完全相同。之后只改了报告的渲染（Holm 校正、p 值格式）和版本号，
  不影响任何 episode、候选或选择结果；所有报告已用最终代码重建。
- 主指标：`score_expected`（每个 replicate 先对 16 个测试场景取均值）。

## 1. single vs independent（R = 20）

| 配置 | proposer | independent − single | 95% bootstrap CI | 置换 p（取自 report.md） |
|---|---|---|---|---|
| `toy_experiment.toml` | es | +0.864 | [−1.696, +3.389] | 0.532 |
| `toy_experiment_random.toml` | random | −1.117 | [−2.653, +0.401] | 0.182 |

R = 20 只够检出 ≥ 3.5 分的效应（见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md)），两者都未观察到差异。

## 2. 统计功效数据池（R = 100）

见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md)。要点：A/A 等价（±0.9 分内）；预算阳性对照可检出；
R = 100 时 MDE 约 1.6 分；预算 192 下 ES 与 random 几乎无差别。

## 3. 信息交换实验（`toy_exchange.toml`，ES，R = 100）

三种模式同为 ES、同样每个 (replicate, mode) 192 个训练 episode；exchange 为 `share_best`、拓扑 `all`、N = 4。
数字取自该运行的 `report.md`。

**测试分（选定策略）**

| 模式 | score_expected 均值 ± 标准差 |
|---|---|
| single | 111.986 ± 7.793 |
| independent | 112.655 ± 7.793 |
| exchange | 113.127 ± 7.688 |

**配对比较（测试分，主指标）**

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p | Holm 校正 p |
|---|---|---|---|---|
| independent − single | +0.669 | [−0.432, +1.822] | 0.250 | 0.500 |
| exchange − independent | +0.472 | [−0.487, +1.403] | 0.330 | 0.500 |
| exchange − single | +1.141 | [+0.059, +2.202] | 0.040 | 0.120 |

**训练集 best-so-far（最终检查点，配对差）**

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p |
|---|---|---|---|
| independent − single | −0.362 | [−1.072, +0.366] | 0.327 |
| exchange − independent | +0.806 | [+0.081, +1.483] | 0.026 |
| exchange − single | +0.444 | [−0.329, +1.189] | 0.255 |

**交换统计**：2400 条交换记录（100 replicate × 6 轮 × 4 搜索者），送达 7200 张卡片，采纳 1248 次（52%）；
exchange 模式 4800 个候选中 3000 个的父代来自其他搜索者。预算账本 used == planned，失败 0，超时 0。
三种模式的"验证分 − 测试分"都约为 2 分（赢家诅咒程度相近）。

## 4. 交换管线的 A/A 对照（`toy_exchange_random.toml`，random，R = 100）

random 忽略证据卡，因此三种模式在分布上等价。

| 比较 | 差值均值 | 95% bootstrap CI | 置换 p |
|---|---|---|---|
| independent − single | −0.313 | [−1.009, +0.390] | 0.382 |
| exchange − independent | −0.102 | [−0.853, +0.665] | 0.794 |
| exchange − single | −0.415 | [−1.192, +0.359] | 0.289 |

送达 7200 张卡片、采纳 0 次、外来父代 0 个；single / independent 两个模式与 `power_random_b192` 数据池逐字一致。

## 5. 解读

- **交换管线本身没有引入偏差**：random 下 exchange − independent 的 CI 紧贴 0（±0.8 分），送达了卡片但没有任何候选受其影响。
- **交换让搜索在训练集上更有效**：exchange − independent 的训练 best-so-far 高 0.81 分（p = 0.026）。
- **但这个优势在测试集上没有被确认**：exchange − independent 的测试分差 +0.47，CI 包含 0；
  exchange − single 的未校正 p = 0.040，但三个比较做 Holm 校正后为 0.12，**不能确认**。
  测试分差 +0.47 小于 R = 100 的 MDE（约 1.6 分），所以这是"效应可能存在但小于本设计的分辨率"，不是"证明无效"。
- 与功效分析一致：在这个 toy 上，搜索方法层面的改动（ES vs random、交换 vs 不交换）带来的测试分差都在 1 分以内，
  而训练预算的变化带来数分的差别。toy 上的结论不能推广到 AuctionNet。

## 6. 复现

```bash
.venv/bin/python -m bidswarm run --config configs/toy_exchange.toml
```

```bash
.venv/bin/python -m bidswarm evaluate --run-dir runs/toy_exchange-<UTC 时间戳>
```

`toy_exchange_random.toml`、`toy_experiment.toml`、`toy_experiment_random.toml` 同理。每个 R = 100 的 exchange 实验在 4 进程上约 5 分钟。
功效数据池的命令见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md) 第 5 节。
