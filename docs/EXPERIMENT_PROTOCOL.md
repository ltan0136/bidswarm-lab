# 实验协议

本协议描述 v0.1 的对照实验：`single`（1 个搜索者）与 `independent`（N 个互不交换信息的搜索者），
在**相同的策略空间、搜索方法和超参**下，用**相同的训练 episode 预算**搜索同一个广告主的出价策略。

## 1. 预算

主预算单位是**每个 (replicate, mode) 的训练 episode 数** `train_episode_budget`。换算全部要求整除，
任何一步不能整除都在配置校验时报错：

```
candidates_total = train_episode_budget / n_train_scenarios     （每个候选在全部训练场景上评估）
per_worker       = candidates_total / n_workers                 （single: 1；independent: N）
rounds           = per_worker / k                               （k = candidates_per_round）
```

例：预算 96、训练场景 4、k = 2 → 24 个候选；single 跑 12 轮；independent（N = 4）每个搜索者 6 个候选、跑 3 轮。

- 协调器在派发每一批任务**之前**静态扣除预算；账本记录 `planned / used / failed / timeouts / retries`，
  `used` 超过 `planned` 会直接报错。训练阶段结束时检查 `used == planned`。
- 失败候选（任一训练 episode 失败）的 fitness 为 `null`，不参与选择，但**照样计入预算**。
- 验证消耗单独入账：每个 (replicate, mode) 计划 `top_k × n_validation` 个 episode；若有效候选少于 K，`used < planned`。
- 测试消耗也单独入账（`evaluate` 写入），不影响训练与验证账目。
- v1 不重试，`retries` 恒为 0。
- **软超时**：每个 episode 记录 `wall_time_s`，超过 `episode_timeout_s` 标为 `status = "timeout"`；
  **不强杀进程**，指标照常保留并参与 fitness（这样结果不依赖机器负载）。超时数量写入账本和报告。

## 2. 重复与配对设计

- 每次实验跑 R 个 replicate，replicate r 的 master seed 为 `seed + r`（`toy_smoke` R = 3，`toy_experiment` R = 20）。
- **配对**：同一 replicate 内，所有模式共用同一组训练 / 验证 / 测试场景。
- 场景种子由 `(master_seed, split, index)` 经 sha256 派生；场景 ID 为 `m<master_seed>-<split>-<index>`。
  构造后检查各 split 之间场景 ID 与种子都不重叠，重叠即报错。
- toy 环境使用共同随机数：同一场景下，不同策略看到完全相同的 pValue、价格和均匀数序列。
- 每个候选的 fitness = 训练场景上主指标的均值。主指标默认 `score_expected`（期望转化计分，方差小），
  可配置为 `score_realized`（实际转化计分）。

## 3. 训练与验证选择

1. 搜索结束后，汇总该模式（全部搜索者）的所有候选。
2. 取训练 fitness 前 K 名（默认 K = 4；同分时先产生的候选优先），在验证场景上评估。
3. 验证均分最高者固定为 selected policy（同分依次比较训练 fitness、训练排名）。验证中有 episode 失败的候选不会被选中。
4. 写入 `selected/<replicate>_<mode>.json`：spec、spec_hash、训练 fitness、验证分、完整短名单与选择规则。

各模式的 K 与验证场景数相同。

## 4. 测试隔离

- `run` 命令**不执行测试**：搜索者只拿到训练场景，选择步骤只拿到验证场景。
- `evaluate` 命令单独读取 `selected/` 中已固定的策略（校验 spec_hash），在测试场景上评估，写入 `test_episodes.jsonl`。
  它还会重新由配置推导场景，并与 `scenarios.json` 比对，不一致即报错。
- `evaluate` 同时评估两个**未调优**的默认基线：`FixedRate(ratio = 1.0)` 与 `DualControl` 默认参数。
  报告中注明它们不是强基线。
- 测试结果不回流到搜索：搜索在 `run` 中已经结束，`evaluate` 不修改候选、谱系或选择。
- 已有测试结果时，`evaluate` 默认拒绝重跑（需 `--force`）。

> **测试隔离是协议边界，不是安全边界。** 测试场景的定义保存在同一个运行目录的 `scenarios.json` 中，
> 任何人或任何代码都可以读取它。本框架保证的是：按协议使用 `run → evaluate → report` 时，搜索与选择代码路径
> 不会读取测试场景或测试结果；它不防御有意的绕过。

## 5. 随机采样对照的等价性

`RandomProposer` 独立均匀采样，不使用任何历史。在这种 proposer 下，`single` 的 24 个候选与 `independent`
四个搜索者各 6 个候选，都是 24 个独立同分布的均匀样本，二者**在分布上等价**。
因此 random 模式下 single 与 independent 的比较**只是正确性检查**（期望差值为 0），不能说明任何搜索方式的优劣。
报告在 proposer = random 时会自动加注。

对 ES（默认）而言，两种模式的差别在于：同样 24 个候选，single 是 1 条 12 轮的长链（首轮随机初始化），independent 是 4 条各 3 轮的短链。
v0.1 只测量这一差别，不实现信息交换。

## 6. Anytime 曲线

每轮结束时记录每个搜索者和模式整体的 best-so-far 训练 fitness，以及该模式已消耗的训练 episode 数
（`anytime.jsonl`）。报告取所有模式都恰好完成一轮的训练 episode 数作为检查点（例：single 每轮 8 个、
independent 每轮 32 个 → 检查点 32 / 64 / 96）。

## 7. 统计与报告

- 每个 replicate 先对测试场景取均值，再对 replicate 报告均值 ± 样本标准差（ddof = 1）。
- 配对差 `independent − single`：对 replicate 做百分位 bootstrap，10000 次重采样，固定种子 20261030，报告 95% 置信区间。
  replicate 数小于 2 时不报告区间；小于 10 时报告会提示区间粗糙。
- 赢家诅咒：对选定策略报告"验证分 − 测试分"的均值 ± 标准差。
- 报告只从产物生成，删除 `report.md` 后 `report` 命令可逐字重建；没有改善也如实显示。

## 8. 评分公式

复刻 AuctionNet `simul_bidding_env/Tracker/PlayerAnalysis.py`：

```
cpa   = cost / (reward + 1e-10)
score = reward × (C / cpa)^2    若 cpa > C
score = reward                  否则
```

上游按 (playerindex, episode) 分组计算后求和再除以 20000；本框架单个 episode **不除以 20000**。
零转化时 cpa 极大，得分约为 0。`score_expected` 以期望转化（获胜且曝光机会的 pValue 之和）为 reward，
`score_realized` 以实际转化为 reward。

## 9. 结果的适用范围

所有 toy 结果都标记 `backend=toy`、`purpose=framework_smoke_test`、`score_formula=auctionnet_neurips_replica`。
它们只用于验证实验框架本身，**不代表 AuctionNet 上的表现，也不代表任何真实广告收益**。
