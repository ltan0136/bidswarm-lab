# AuctionNet 接入说明

**当前状态：`integration_status = integrated_online`。** AuctionNet 的**在线仿真**已接入：bidswarm 在每个进程中启动一个
长驻的上游 Python 3.9 子进程（`src/bidswarm/environments/auctionnet_runner.py`），用上游固定 commit 的仿真器评估内核策略。
离线评测路径（需约 80GB 数据）没有接入，配置中选择它会报错。上游代码、数据和权重都不进本仓库。
任何缺失的上游环境、错误的 commit、runner 崩溃或不变量被破坏都会直接报错，**绝不回退到 toy**。

接入分三步：第 1 步只读调查与实测（第 0–3 节），第 2 步 shim 与后端（第 5 节），第 3 步冒烟测试（第 6 节）。

## 0. 固定版本与环境

| 项目 | 值 |
|---|---|
| 上游仓库 | `https://github.com/alimama-tech/AuctionNet`（Apache-2.0） |
| 固定 commit | `c20de631a40e24669d1d6a8e2acefd8f492f172d`（2025-04-25，截至 2026-11-06 仍是 main 的最新提交，所以 2026-10-30 对 main 的核查与它是同一份代码） |
| 解释器 | CPython 3.9.25（`uv python install 3.9.25`）；上游 README 写的是 3.9.12。`run/run_test.py:14` 的 `from collections import Iterable` 在 Python ≥ 3.10 上会失败 |
| 依赖 | 上游 `requirements.txt`（`torch==1.12.0`，PyPI 上为 `+cu102` 构建；`numpy==1.24.2`、`pandas==2.0.3`、`scipy==1.11.1`、`gin_config==0.5.0` 等），**另需 `einops`**：`requirements.txt` 漏列，但 `simul_bidding_env/PvGenerator/model_utils/model_utils.py:5` 会导入它（本次用 0.8.2）。只用 PyPI 即可装完，不需要额外网络白名单 |
| 体积 | 仓库检出 408 MB（`simul_bidding_env/` 252 MB，含流量生成模型与竞争者模型）；虚拟环境约 1.9 GB |

准备上游环境（只需一次，放在本仓库之外）：

```bash
git clone https://github.com/alimama-tech/AuctionNet && git -C AuctionNet checkout c20de631a40e24669d1d6a8e2acefd8f492f172d
uv python install 3.9.25 && uv venv -p 3.9.25 an-venv
uv pip install -p an-venv/bin/python -r AuctionNet/requirements.txt einops==0.8.2
export BIDSWARM_AUCTIONNET_REPO=$PWD/AuctionNet BIDSWARM_AUCTIONNET_PYTHON=$PWD/an-venv/bin/python
```

然后照常运行（每个工作进程常驻约 2.7 GB 内存，按内存设 `--max-workers`；15 GB 的机器用 4）：

```bash
.venv/bin/python -m bidswarm run --config configs/auctionnet_smoke.toml --max-workers 4
```

```bash
.venv/bin/python -m bidswarm evaluate --run-dir runs/auctionnet_smoke-<UTC 时间戳> --max-workers 4
```

配置中的 `repo_path` / `python_executable` 可以写成 `$BIDSWARM_AUCTIONNET_REPO` 这样的环境变量，清单里记录展开后的路径。
`commit_sha` 必须等于上面的固定 commit，换 commit 前要重做第 1 步的核对。

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
   是广告位价格，无论是否曝光；toy 当时按获胜收费。**后果：内核从 cost 列累计花费，在 AuctionNet 上会高估已花费金额**。
   探针中 FixedRate 内核记账 3262.6、实际扣费 2511.5（高估 30%），DualControl 记账 3169.0、实际 2400.1（高估 32%），
   预算节奏因此偏保守（预算消耗 87% / 83%）。**已修正**：内核改为累计 `cost × is_exposed`，toy 也改为只对曝光收费，
   toy 结果已重跑（[TOY_RESULTS.md](TOY_RESULTS.md)）。
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

这只是一个广告主的一个周期，用来确认流程能跑通，不说明任何策略的优劣。上表两行内核结果是**修正花费记账之前**测的
（FixedRate 在内核认为预算用完后停止出价），修正后的数字见第 6 节。

**计算量**：按 `toy_cooperative.toml` 的协议（R = 100、3 种模式、训练预算 192、验证 4 × 8、测试 16 × 5 组）一次运行约需 75,200 个周期，
本机约 270 小时；R = 20 也要约 54 小时。第一版 AuctionNet 实验必须缩到几千个周期的量级：`configs/auctionnet_pilot.toml`
取 R = 10、训练预算 96、验证 4、测试 8（训练 2,880 + 验证 480 + 测试 400 = 3,760 个周期，本机约 13–14 小时）。
最小可检出效应需要在 AuctionNet 上重新估计。

## 4. 本仓库的组成部分

- `src/bidswarm/kernel/`：可移植策略内核。只依赖 numpy 与标准库；在 Python 3.9.25 + numpy 2.0.2 和 numpy 1.24.2 下都实测运行过。
  已花费金额按 `cost × is_exposed` 累计，与上游扣费一致。
- `src/bidswarm/environments/auctionnet_runner.py`：在上游 Python 3.9 中运行的 runner 与 shim；只导入上游代码和 `kernel`
  （`tests/test_kernel_compat.py` 检查 Python 3.9 语法与导入规则）。
- `src/bidswarm/environments/auctionnet.py`：bidswarm 侧客户端，`Environment` 接口的实现。
- `scripts/auctionnet_probe.py`：第 1 步的探针，通过上游 `run_test` 跑一个周期，也是冒烟测试的参照。
- `scripts/auctionnet_smoke.py`：第 3 步的冒烟测试（第 6 节）。
- `configs/auctionnet_smoke.toml`（端到端冒烟）、`configs/auctionnet_pilot.toml`（第一版实验设计，尚未运行）、
  `configs/auctionnet.example.toml`（字段说明，路径是占位符，运行即报 `AuctionNetSetupError`）。

## 5. 实现（第 2 步）

第 1 步列出的要求逐条落实如下。

1. **不经过 `initialize_player_agent`。** runner 把 shim 对象直接交给 `Controller(player_index, player_agent=shim)`，
   并确认 `Controller.agents[player_index]` 包装的正是这个对象；不存在回退到 PID 的路径。
2. **shim 持有可写的 `remaining_budget`**，并在每个周期的 `reset()` 中用 `Controller` 赋的预算、CPA 与类别构造内核策略。
3. **花费记账已修正**（见上文第 2 节第 9 条）。每个周期结束后，runner 让内核读入最后一步的历史，客户端核对
   内核记账 = 上游实际扣费（相对误差 ≤ 1e-6），并核对 shim 的 `bidding()` 调用次数 = 循环中玩家出价的次数；不一致则中止运行。
4. **自己的逐步循环。** runner 按 `run_test.py:138-213` 实现单个 episode 的循环，复用上游的 `get_winner` 与 `adjust_over_cost`，
   因此可以评估任意 episode 序号。与 `run_test` 的逐字比对见第 6 节。
5. **线程与进程。** runner 设置 `OMP_NUM_THREADS=1` 与 `torch.set_num_threads(1)`；每个 bidswarm 进程（串行时是主进程，
   并行时是每个 spawn 出来的工作进程）在第一次评估时启动一个 runner，之后复用，模型只加载一次。
6. **隔离。** 上游只在独立的 Python 3.9 虚拟环境中以子进程运行，启动时检查检出目录确实位于固定 commit；
   上游打印的内容被重定向到 stderr（写入临时日志，出错时附在异常里），协议只走 stdout。

**场景。** 广告主由 `player_index` 固定（预算、CPA 约束、类别取自上游 `Controller.py` 的表，runner 启动时再核对一次）；
场景种子对 2^31 取模得到 episode 序号，episode 决定流量。`evaluation/splits.py` 检查同一 replicate 的训练 / 验证 / 测试
不共用 episode 序号。每条 episode 记录带 `backend=auctionnet`、`purpose=auctionnet_online_simulation`、上游 commit、
玩家位置与 episode 序号，字段集合包含 toy 记录的全部字段。

**失败语义。** 内核抛异常或返回非法出价（形状不对、非有限、负数）→ 该 episode 记为 `failed`，照常计入预算；
runner 崩溃、超时、自身错误或不变量被破坏 → 抛 `AuctionNetRunnerError`，整个运行中止；上游环境缺失或 commit 不对 →
构造环境时抛 `AuctionNetSetupError`，在写出任何产物之前失败。

## 6. 冒烟测试（第 3 步）

2026-11-06 在本机（4 核 / 15 GB）上运行，代码提交 `66839dd`。

**与上游 `run_test` 逐字比对**（`scripts/auctionnet_smoke.py`，玩家位置 0：预算 2900、CPA 约束 100）：

| 策略（默认参数） | episode | 转化 | 实际扣费 | 本周期得分 | 与 `run_test` | 与 `PlayerAnalysis` 得分之差 |
|---|---|---|---|---|---|---|
| FixedRate | 0 | 31 | 2899.959884713612 | 31 | 转化、扣费逐位相同 | 0 |
| FixedRate | 1 | 2 | 253.40124154696525 | 1.246 | 逐位相同 | 2e-12 |
| DualControl | 0 | 29 | 2899.962487834864 | 29 | 逐位相同 | 0 |
| DualControl | 1 | 10 | 1429.8278664929487 | 4.891 | 逐位相同 | 7e-12 |

每个周期内核记账都等于实际扣费（差 < 1e-12）。同一 runner 中按 episode 0 → 1 → 0 请求、以及换策略后再请求 episode 0，结果逐字相同。
得分的 1e-12 量级差来自上游公式里 `+1e-10` 的位置，不是仿真差异。

比对中发现并修正的问题：第一次比对时转化数全部一致，但扣费在进程跑过第一个周期之后漂移约 1e-9。原因是竞争者是 TorchScript
模型，图执行器在前几次调用时以未优化的方式剖析，之后切换到优化过的图，**上游自己的结果也随进程历史变化**。runner 因此关闭了
图优化（`torch._C._set_graph_executor_optimize(False)`）；比对时探针也用 `--jit-optimize off`。默认设置下的上游 `run_test`
与这里的结果可能在约 1e-9 的相对精度上不同，转化数不受影响。另外 runner 用 `np.sum` 汇总每步数值，与 `PlayerAnalysis` 一致。

**真实上游的集成测试**（`tests/test_auctionnet_integration.py`，3 个周期）：通过。

**端到端**（`configs/auctionnet_smoke.toml`，R = 1，single 与 independent，32 个投放周期）：

- `run` 4 进程 244 秒（训练 16 + 验证 8 个周期），`evaluate` 4 进程 78 秒（测试 8 个周期）；每周期 29–58 秒（含 runner 启动）。
- 全部 32 个周期 `status = ok`；预算账本 used == planned；6 个场景的 episode 序号互不相同。
- 同一运行用 `evaluate --force --max-workers 2` 重跑测试阶段：8 条测试记录与 4 进程时逐字相同（不计 `wall_time_s`），
  即结果与进程数无关。
- 报告的声明与结论都标明 AuctionNet 在线仿真、上游 commit 与玩家位置。一个 replicate 的数字没有统计意义，这里只检查流程。

## 7. 下一步

- 跑 `configs/auctionnet_pilot.toml`（R = 10，约 3,760 个周期，本机约 13–14 小时），并用它的配对差估计 AuctionNet 上的
  最小可检出效应，再决定正式实验的 R。
- 可选的提速：减少每个进程的流量数（会改变环境，必须单独标注）、更大的机器、或按竞争者缓存出价（会丢掉交互，只能做近似）。
