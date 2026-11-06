# AuctionNet 接入说明

**当前状态：`integration_status = not_integrated`。** 接入计划的第 1 步（只读调查与实测）已于 2026-11-06 完成，结论见下文；
本仓库仍然没有 AuctionNet 后端，也没有把 AuctionNet 的代码、数据或权重放进版本库。`backend = "auctionnet"` 在构造后端时
就抛出 `BackendNotIntegratedError`，**绝不静默回退到 toy**（有测试覆盖）。`configs/auctionnet.example.toml` 只是示例，运行即报错。

## 0. 固定版本与环境

| 项目 | 值 |
|---|---|
| 上游仓库 | `https://github.com/alimama-tech/AuctionNet`（Apache-2.0） |
| 固定 commit | `c20de631a40e24669d1d6a8e2acefd8f492f172d`（2025-04-25，截至 2026-11-06 仍是 main 的最新提交，所以 2026-10-30 对 main 的核查与它是同一份代码） |
| 解释器 | CPython 3.9.25（`uv python install 3.9.25`）；上游 README 写的是 3.9.12。`run/run_test.py:14` 的 `from collections import Iterable` 在 Python ≥ 3.10 上会失败 |
| 依赖 | 上游 `requirements.txt`（`torch==1.12.0`，PyPI 上为 `+cu102` 构建；`numpy==1.24.2`、`pandas==2.0.3`、`scipy==1.11.1`、`gin_config==0.5.0` 等），**另需 `einops`**：`requirements.txt` 漏列，但 `simul_bidding_env/PvGenerator/model_utils/model_utils.py:5` 会导入它（本次用 0.8.2）。只用 PyPI 即可装完，不需要额外网络白名单 |
| 体积 | 仓库检出 408 MB（`simul_bidding_env/` 252 MB，含流量生成模型与竞争者模型）；虚拟环境约 1.9 GB |

复现环境：

```bash
git clone https://github.com/alimama-tech/AuctionNet && git -C AuctionNet checkout c20de631a40e24669d1d6a8e2acefd8f492f172d
uv python install 3.9.25 && uv venv -p 3.9.25 an-venv
uv pip install -p an-venv/bin/python -r AuctionNet/requirements.txt einops==0.8.2
```

## 1. 已核实的上游事实（commit `c20de63`）

以下行号都指固定 commit 上的文件。

### 策略接口

- 玩家对象被 `Controller` 赋值 `budget`、`cpa`、`category`（`simul_bidding_env/Controller/Controller.py:95-97`），由
  `run/run_test.py` 读写 `remaining_budget`、调用 `reset()` 与
  `bidding(timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid, historyAuctionResult, historyImpressionResult, historyLeastWinningCost)`。
  参赛策略不必继承 `BaseBiddingStrategy`，有这些属性和方法即可。
- `PlayerAgentWrapper.reset()` 有 2 秒超时（`simul_bidding_env/strategy/player_agent_wrapper.py:36`），`bidding()` 没有超时。

### 评分（`simul_bidding_env/Tracker/PlayerAnalysis.py`）

- `cpa = cost / (reward + 1e-10)`（第 45 行）；`cpa > C` 时 `score = reward × (C / cpa)^2`，否则 `score = reward`（第 94-101 行，beta = 2）。
- 按 `(playerindex, episode)` 计分，汇总时除以 20000（第 119 行）。
- **复刻已数值核对**：三次探针运行（PID、FixedRate 默认、DualControl 默认）中，`src/bidswarm/evaluation/metrics.py` 的
  `auctionnet_score` 与上游每周期得分的差小于 1e-11。

### 两条评测路径

- **离线评测**（`strategy_train_env/`）需要从阿里云 OSS 下载约 80GB 的预生成数据（`pre_generated_dataset/readme_dataset.md`）。
  该主机不在本环境网络白名单内，磁盘也放不下，**不采用**。
- **在线仿真**（`run/run_test.py` + `config/test.gin`）不需要任何额外下载，是接入路径。

### 仿真设定（`config/test.gin`、`Controller.py`）

- 每个投放周期 48 个时间步（`test.gin:25`，绑定到 `run_test.num_tick` 与 `Controller.num_tick`，第 32、39 行），每周期约 50 万次流量，48 个广告主。
- 广告主表固定：预算 2000–4850（`Controller.py:114` 起）、CPA 约束 60–130（第 103 行起）、类别 = 序号 // 8（第 31 行）。
  其余 47 个位置由 PID、IQL、TD3+BC、OnlineLP、CQL、BC、BCQ、MOPO、COMBO 等基线占据（第 38 行起）。
- 预留价 0.0001（`test.gin:8`），剩余预算低于 0.1 的广告主本步出价为 0（`test.gin:10`、`run_test.py:170`）。

## 2. 原开放问题的答案

1. **`tickValue` 是实际转化。** `run_test.py` 记录的 reward 是 `conversion_action_pit` 之和：按曝光后的二项抽样得到
   （`BiddingEnv.py` 中 `_calculate_conversion_action`），不是期望值。
2. **T = 48。** 见上。
3. **在线仿真不需要额外数据或权重**，所需模型都在仓库里；唯一缺口是 `requirements.txt` 漏列的 `einops`。
4. **种子与重置。** 流量只由 episode 序号决定（`NeurIPSPvGen.py:117`、`:133` 等处 `default_rng(seed=self.episode)`）；
   广告主价值截断参数由 `hash((广告主, 0, episode))` 决定（`BiddingEnv.py:27`，整数元组的 hash 不受 `PYTHONHASHSEED` 影响）；
   曝光、转化与超预算丢弃每次调用都用固定种子重建随机数发生器（`BiddingEnv.py:22`、`run_test.py:79`）；
   模块加载时 `np.random.seed(1019)`（`NeurIPSPvGen.py:11`）。`run_test` 总是从 episode 0 开始跑 `num_episode` 个周期（`run_test.py:138`）。
   实测：同一位置的 PID 在两个独立进程中得分逐字相同；但把线程数改为 1 后，汇总得分在约 1e-12 的相对精度上不同，
   **逐字复现需要固定线程数**。
5. **拍卖机制。** 3 个广告位的广义二价：第 k 位支付第 k+1 高的出价（`BiddingEnv.py:55-79`），低于预留价的按预留价；
   曝光概率按位次为 1 / 0.8 / 0.6（第 17 行），第 2 位未曝光时第 3 位也不曝光；价格恰好等于预留价的位置视为未售出（第 117 行）。
   `historyLeastWinningCost` 是第 3 位的市场价（第 53 行），所有广告主相同。
6. **历史参数格式与内核假设一致**（`run_test.py:206-213`）：`historyPValueInfo = [pValue, sigma]`、
   `historyAuctionResult = [xi（是否获得广告位）, slot, cost]`、`historyImpressionResult = [is_exposed, conversion]`，逐步追加、长度相同。
   内核的 FixedRate 与 DualControl 在仿真中跑满 48 步，没有触发 `PolicyInputError`。
7. **上游会直接改写玩家的剩余预算**：`agent.remaining_budget -= cost[i]`（`run_test.py:201`），并用它决定是否出价（第 170 行）。
   内核的 `remaining_budget` 是只读属性，**shim 必须自己持有可写的 `remaining_budget`**（探针中的 shim 就是这样做的）。
8. **超预算处理与 toy 不同。** 上游反复计算超支比例，按比例随机丢弃超支广告主已赢得的流量（种子固定为 1），再重新拍卖，
   直到无人超支（`run_test.py:70-84`、`183-198`）；toy 按预抽取的顺序键截断。
9. **只对曝光收费，与 toy 不同。** 实际扣费为 `cost_pit × is_exposed_pit`（`run_test.py:192`），而 `historyAuctionResult` 的 cost 列
   是广告位价格，无论是否曝光；toy 按获胜收费。**后果：内核从 cost 列累计花费，在 AuctionNet 上会高估已花费金额**。
   探针中 FixedRate 内核记账 3262.6、实际扣费 2511.5（高估 30%），DualControl 记账 3169.0、实际 2400.1（高估 32%），
   预算节奏因此偏保守（预算消耗 87% / 83%）。
10. **numpy 1.24.2 下内核运行正常**（此前只在 numpy 2.0.2 上测过）。
11. **T 在构造时可得**：shim 从配置拿到 48，与 `Controller.num_tick` 一致。

### 调查中新发现的问题

- **静默回退到 PID。** `run_test.initialize_player_agent()` 导入参赛策略失败时只记一条错误日志，然后改用 `PidBiddingStrategy`
  继续评测（`run_test.py:23-45`，回退在第 37-41 行）。照上游方式注入策略，一旦出错跑出来的就是 PID 的成绩。
- **上游会反序列化仓库里的文件**：竞争者用 `torch.jit.load`（如 `simul_bidding_env/strategy/mbrl_mopo_bidding_strategy.py:20`）
  和 `pickle.load`（同文件第 23 行）加载模型与归一化参数。pickle 文件被篡改即可执行任意代码，所以上游只能在固定 commit、
  独立虚拟环境的子进程中运行，不能导入 bidswarm 进程。

## 3. 实测（2026-11-06，4 核 / 15 GB 内存）

用 `scripts/auctionnet_probe.py` 在上游 Python 3.9 环境中跑一个周期（episode 0），玩家位置 0（预算 2900，CPA 约束 100）：

| 运行方式 | 每周期耗时 | 峰值内存 |
|---|---|---|
| 单进程，默认线程数 | 52 s（其中仿真 42 s，其余为加载模型） | 2.75 GB |
| 3 进程并行，默认线程数 | 98–101 s / 进程 | 2.74 GB / 进程 |
| 4 进程并行，`OMP_NUM_THREADS=1` | 45–50 s / 进程，4 个周期共 51 s | 2.71–2.74 GB / 进程 |

单线程多进程的吞吐约为 **280 个周期 / 小时**，是默认多线程单进程的约 4 倍；内存限制最多同时跑约 5 个进程。

| 玩家 | 转化 | 花费 | 实际 CPA | 本周期得分 |
|---|---|---|---|---|
| PID（上游） | 16 | 2900.0 | 181.2 | 4.87 |
| FixedRate 默认参数（内核） | 31 | 2511.5 | 81.0 | 31 |
| DualControl 默认参数（内核） | 25 | 2400.1 | 96.0 | 25 |

这只是一个广告主的一个周期，用来确认流程能跑通，不说明任何策略的优劣。

**计算量**：按 `toy_cooperative.toml` 的协议（R = 100、3 种模式、训练预算 192、验证 4 × 8、测试 16 × 5 组）一次运行约需 75,200 个周期，
本机约 270 小时；R = 20 也要约 54 小时。第一版 AuctionNet 实验必须缩到约 3,000 个周期量级（本机约 11 小时），
例如 R = 10、训练预算 48、其余不变（训练 1,440 + 验证 960 + 测试 800 = 3,200 个周期）。最小可检出效应需要在 AuctionNet 上重新估计。

## 4. 本仓库已经为接入准备的部分

- `src/bidswarm/kernel/`：可移植策略内核。只依赖 numpy 与标准库；在 Python 3.9.25 + numpy 2.0.2 和本次的 numpy 1.24.2 下都实测运行过。
- 内核策略的 `bidding()` 参数列表与上游完全一致，可直接转发。
- `PolicySpec` 是纯数据（JSON），`spec_hash` 是 canonical JSON 的 sha256，可在两个进程之间核对是否同一策略。
- `environments/auctionnet.py`：占位后端，列出阻塞项；`doctor` 命令会打印它们。
- `scripts/auctionnet_probe.py`：第 1 步用的探针，含一个最小 shim 原型。它不是后端。

## 5. 后续步骤

### 第 1 步：只读调查（已完成，见上文）

### 第 2 步：shim 与后端（需要先做的决定与修改）

1. **不经过 `initialize_player_agent`**，直接把 shim 对象交给仿真；shim 记录 `spec_hash` 与 `bidding()` 调用次数，后端在每个周期后核对，
   对不上就报错。
2. shim 持有可写的 `remaining_budget`，在 `reset()` 时用 `Controller` 赋的预算与 CPA 构造内核策略。
3. **修正内核的花费记账**：改为累计 `cost × is_exposed`。这会改变 toy 上的预算节奏，需要先决定 toy 是否也改为按曝光收费
   （建议改，使两个后端的语义一致；改后所有 toy 结果要重跑并在 `TOY_RESULTS.md` 注明代码版本）。
4. **自己实现逐步循环**，以便按 episode 序号划分训练 / 验证 / 测试场景（`run_test` 只能从 0 开始）；用 episode 0、1 与
   `run_test` 的输出逐字比对来验证循环等价。场景 = 固定的玩家位置（同一个广告主）× 互不重叠的 episode 序号。
5. 子进程固定 `OMP_NUM_THREADS=1`，并复用进程跑多个周期，摊薄每次 5–10 秒的模型加载。
6. 上游只在固定 commit 的独立 Python 3.9 虚拟环境中以子进程运行；只有 `bidswarm/kernel/` 进入上游导入路径。

### 第 3 步：子进程冒烟测试

- 用 FixedRate 与 DualControl 默认参数各跑一个周期，核对：出价形状、非负、有限；内核记账的花费等于上游实际扣费；
  本仓库的评分等于上游 `PlayerAnalysis`；同一配置在不同进程数下结果逐字相同。
- 通过后才把 `integration_status` 改为相应状态，并在报告中如实标注后端、上游 commit 与数据来源。
