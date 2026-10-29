# 实验协议

本协议描述三种搜索模式的对照实验：

| 模式 | 搜索者数 | 信息交换 |
|---|---|---|
| `single` | 1 | 无 |
| `independent` | N | 无（`NoExchange`） |
| `exchange` | N（与 independent 相同） | 有（`ShareBest`，见第 6 节） |

三种模式在**相同的策略空间、搜索方法和超参**下，用**相同的训练 episode 预算**搜索同一个广告主的出价策略。
主要比较是 `exchange − independent`（交换本身的作用）和 `independent − single`（拆成多个搜索者的作用）。

## 1. 预算

主预算单位是**每个 (replicate, mode) 的训练 episode 数** `train_episode_budget`。换算全部要求整除，
任何一步不能整除都在配置校验时报错：

```
candidates_total = train_episode_budget / n_train_scenarios     （每个候选在全部训练场景上评估）
per_worker       = candidates_total / n_workers                 （single: 1；independent / exchange: N）
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

- 每次实验跑 R 个 replicate，replicate r 的 master seed 为 `seed + r`（`toy_smoke` R = 3，`toy_experiment` R = 20，
  `toy_exchange` R = 100）。R 的选择见第 9 节与 [POWER_ANALYSIS.md](POWER_ANALYSIS.md)。
- replicate r 的全部结果只取决于配置与 master seed `seed + r`，与 R 和其他模式无关（有测试覆盖）：
  R = 20 的运行恰好是同配置 R = 100 运行的前 20 个 replicate；同一 seed 下加入 `exchange` 模式不改变
  `single` / `independent` 的结果。因此不同运行之间可以按 replicate 配对。
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

`RandomProposer` 独立均匀采样，不使用任何历史，也忽略收到的证据卡。在这种 proposer 下，`single` 的 24 个候选、
`independent` 四个搜索者各 6 个候选、`exchange` 四个搜索者各 6 个候选，都是 24 个独立同分布的均匀样本，
三者**在分布上等价**。因此 random 模式下各模式之间的比较**只是正确性检查**（A/A，期望差值为 0），
不能说明任何搜索方式的优劣；`exchange − independent` 的 A/A 还检查交换管线没有偷偷改变预算或候选。
报告在 proposer = random 时会自动加注。

对 ES（默认）而言，同样 24 个候选，single 是 1 条 12 轮的长链（首轮随机初始化），independent 是 4 条各 3 轮的短链，
exchange 是 4 条各 3 轮、但每轮之后可以换用别人更好的父代的链。

## 6. 信息交换（exchange 模式）

- **时机**：只在轮次边界。每轮所有候选评估完、各搜索者 `observe` 自己的结果之后，协调器调用
  `ExchangePolicy.exchange`，再把送达的证据卡交给各搜索者，然后才开始下一轮的 `propose`。
- **内容（`share_best`，精英迁移）**：每个搜索者收到邻居当前 best-so-far 候选的证据卡
  （candidate_id、来源搜索者、轮次、spec_hash、fitness、归一化坐标）。fitness 为 null 的候选不会被分享。
  拓扑 `all`：收到其他所有搜索者的；`ring`：只收到搜索者 w − 1 的。
- **使用（ES）**：若收到的最好卡片严格优于自己的父代，就采纳它为父代；σ 不变（σ 只由自己子代的成败调整）。
  自己的候选全部失败、没有父代时，也会采纳外来父代。random proposer 忽略卡片。
- **预算**：交换不消耗 episode，只移动已经付过费的信息；exchange 与 independent 的预算换算完全相同。
- **隔离与审计**：每个搜索者的 `info_sources` 只能包含自己的候选或**此前送达给它**的卡片，违反即报错（有测试覆盖）；
  从别人那里采纳的父代体现在子代的 `parent_id` 中。每轮每个搜索者的送达卡片与采纳结果写入 `exchange.jsonl`。
- `single` 与 `independent` 始终使用 `NoExchange`，不受 `[search.exchange]` 配置影响。

## 7. Anytime 曲线

每轮结束时记录每个搜索者和模式整体的 best-so-far 训练 fitness，以及该模式已消耗的训练 episode 数
（`anytime.jsonl`）。报告取所有模式都恰好完成一轮的训练 episode 数作为检查点（例：single 每轮 8 个、
independent 每轮 32 个 → 检查点 32 / 64 / 96）。

## 8. 统计与报告

- 每个 replicate 先对测试场景取均值，再对 replicate 报告均值 ± 样本标准差（ddof = 1）。
- 配对差（`independent − single`、`exchange − independent`、`exchange − single`，按出现的模式）：
  对 replicate 做百分位 bootstrap，10000 次重采样，固定种子 20261030，报告 95% 置信区间；
  同时报告配对符号翻转置换检验的 p 值（20000 次，固定种子 20261031）。
  replicate 数小于 2 时不报告区间；小于 30 时报告提示：百分位 bootstrap 在小样本下偏窄
  （toy 上实测 R = 20 时名义 5% 的假阳性率约为 7.6%），应同时参考置换检验 p 值。
- 多重比较：同时出现多个比较时（三种模式时为 3 个），对主指标的置换检验 p 做 Holm 校正；
  自动结论只有在未校正 CI 不含 0 **且** Holm 校正后 p ≤ 0.05 时才判定"高于 / 低于"，否则写明"不能确认"。
- 赢家诅咒：对选定策略报告"验证分 − 测试分"的均值 ± 标准差。
- 报告只从产物生成，删除 `report.md` 后 `report` 命令可逐字重建；没有改善也如实显示。

## 9. 统计功效与 R 的选择

详见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md)。在 toy 环境、预算 192、ES 的设置下（主指标 `score_expected`）：

| R | 80% 功效下的最小可检出效应（MDE） |
|---|---|
| 20 | 约 3.5 分（约 3.1%） |
| 50 | 约 2.2 分（约 2.0%） |
| 100 | 约 1.6 分（约 1.4%） |

作为参照，同一设置下把训练预算从 96 加倍到 192 带来约 1.4 ~ 1.7 分的测试分提升。信息交换的效应预计不大于这个量级，
因此比较交换效果的实验应使用 **R ≥ 100**；R = 20 只适合检查 ≥ 3.5 分的大效应或做冒烟测试。

## 10. 评分公式

复刻 AuctionNet `simul_bidding_env/Tracker/PlayerAnalysis.py`：

```
cpa   = cost / (reward + 1e-10)
score = reward × (C / cpa)^2    若 cpa > C
score = reward                  否则
```

上游按 (playerindex, episode) 分组计算后求和再除以 20000；本框架单个 episode **不除以 20000**。
零转化时 cpa 极大，得分约为 0。`score_expected` 以期望转化（获胜且曝光机会的 pValue 之和）为 reward，
`score_realized` 以实际转化为 reward。

## 11. 结果的适用范围

所有 toy 结果都标记 `backend=toy`、`purpose=framework_smoke_test`、`score_formula=auctionnet_neurips_replica`。
它们只用于验证实验框架本身，**不代表 AuctionNet 上的表现，也不代表任何真实广告收益**。
