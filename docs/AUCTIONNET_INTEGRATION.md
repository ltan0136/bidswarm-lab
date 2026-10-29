# AuctionNet 接入说明

**当前状态：`integration_status = not_integrated`。** 本仓库没有 AuctionNet 适配层，没有下载 AuctionNet 的代码、数据或权重，
也没有在 AuctionNet 上运行过任何东西。`backend = "auctionnet"` 在构造后端时就抛出 `BackendNotIntegratedError`，
**绝不静默回退到 toy**（有测试覆盖）。`configs/auctionnet.example.toml` 只是示例，运行即报错。

## 1. 已核实的上游事实

> 来源：项目负责人提供的核查记录，标注为 2026-10-30 对 AuctionNet **main 分支**源码的核查。本仓库没有独立复核这些内容。
> **适配前需用实际 commit SHA 复核**：main 分支会变化，下文每一条都应在固定的 commit 上重新确认，并把 SHA 写入配置的 `commit_sha`。

### 策略接口

- `BaseBiddingStrategy(budget, name, cpa, category)`。
- 抽象方法：`reset()` 和
  `bidding(timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid, historyAuctionResult, historyImpressionResult, historyLeastWinningCost)`。
- `bidding()` 返回当前时间步所有广告机会的出价。

### 评分（`simul_bidding_env/Tracker/PlayerAnalysis.py`）

- `cpa = cost / (reward + 1e-10)`。
- `cpa > C` 时 `score = reward × (C / cpa)^2`，否则 `score = reward`。
- 按 `(playerindex, episode)` 分组计算，汇总时 `score = 各组得分之和 / 20000`。
- 零转化时 cpa 极大，得分约等于 0。

本仓库的复刻见 `src/bidswarm/evaluation/metrics.py`，数值测试见 `tests/test_metrics.py`。

### 两条评测路径

- **离线评测**：`strategy_train_env/main/main_test.py`，需要逐机会原始数据；完整预生成数据集约 80GB。
- **在线仿真**：根目录 `main_test.py` + `config/test.gin`，即 `simul_bidding_env`，含流量生成器 `PvGenerator` 和竞争者策略。

### 其他

- 参赛策略的注入方式：修改 `strategy_train_env/bidding_train_env/strategy/__init__.py` 中的 `PlayerBiddingStrategy` 别名。
- 官方环境为 Python 3.9.12，许可证为 Apache-2.0。
- 仓库自带 Abid、PID、OnlineLp、IQL 等基线。

## 2. 开放问题（未核实）

来自核查记录：

1. `tickValue` 记录的是实际转化还是期望值。
2. 每个投放周期的时间步数（据了解竞赛设定为 48）。
3. 在线仿真是否需要额外权重或数据。
4. 种子与重置机制。
5. 拍卖机制细节。

本仓库在设计内核时额外依赖、同样**未核实**的假设：

6. 五个历史参数是否为逐步追加、长度相同的列表，以及每项的列含义是否与 toy 一致
   （`historyPValueInfo = [pValue, sigma]`、`historyAuctionResult = [is_win, slot, cost]`、`historyImpressionResult = [is_exposed, conversion]`）。
   内核要求五个历史列表长度相同，否则抛 `PolicyInputError`。
7. 上游是否在每步之后更新策略对象的剩余预算属性。内核不依赖它：已花费金额从 `historyAuctionResult` 的 cost 列增量累加。
8. 上游在某步花费超过剩余预算时如何处理（toy 中按预抽取的顺序键截断）。
9. 上游是否按获胜计费还是按曝光计费（toy 中按获胜计费）。
10. 上游环境中的 numpy 版本。内核已在 Python 3.9.25 + numpy 2.0.2 上实测运行；未在 numpy 1.x 上测试。
11. 策略构造时能否拿到 T（每周期时间步数）。内核把它作为构造参数 `num_time_steps`（默认 48）。

## 3. 本仓库已经为接入准备的部分

- `src/bidswarm/kernel/`：可移植策略内核。只依赖 numpy 与标准库；每个文件以 `from __future__ import annotations` 开头；
  通过 `ast.parse(feature_version=(3, 9))` 与导入白名单检查；在真实 Python 3.9.25 解释器上实测，与 3.11 的输出逐位一致。
- 内核策略的 `bidding()` 参数列表与上游完全一致，可直接转发。
- `PolicySpec` 是纯数据（JSON），`spec_hash` 是 canonical JSON 的 sha256，可在两个进程之间核对是否同一策略。
- `environments/auctionnet.py`：占位后端，列出阻塞项；`doctor` 命令会打印它们。

## 4. 下一步计划

### 第 1 步：只读调查

- 在一个固定 commit 上克隆 AuctionNet，记录 SHA。不修改上游代码，不下载 80GB 数据集。
- 逐条复核第 1 节的事实，回答第 2 节的开放问题，结论写回本文件（注明 SHA 与文件路径、行号）。
- 确认在线仿真路径（`simul_bidding_env`）是否能在不下载额外数据 / 权重的前提下运行；若需要，记录体积与来源，交由负责人决定。

### 第 2 步：编写 shim

- 新建一个继承 `BaseBiddingStrategy` 的薄适配类，例如 `BidSwarmShimStrategy`：
  - 从环境变量（例如 `BIDSWARM_POLICY_SPEC=/path/to/spec.json`）指定的 JSON 文件读取 `PolicySpec`，校验 `spec_hash`；
  - 构造时调用 `bidswarm.kernel.make_policy(spec, budget, cpa, category, num_time_steps)`；
  - `reset()` 与 `bidding(...)` 原样转发给内核；
  - 不包含任何搜索逻辑。
- 通过 `PlayerBiddingStrategy` 别名注入 shim；只把 `bidswarm/kernel/` 目录放进上游的导入路径。
- 在 `environments/auctionnet.py` 中实现后端：在**子进程**里用上游的 Python 3.9.12 解释器运行评测，解析输出为本仓库的 episode 记录格式，
  并标记 `backend=auctionnet`。在此之前保持 `not_integrated`。

### 第 3 步：子进程冒烟测试

- 用 `FixedRate(ratio=1.0)` 与 `DualControl` 默认参数各跑一个周期，核对：
  - 出价形状、非负、有限；
  - 内核从历史累加的花费与上游记录的花费一致；
  - 本仓库复刻的评分与上游 `PlayerAnalysis` 的输出一致（同一组 cost / reward / C）。
- 只有在冒烟测试通过、开放问题有结论之后，才把 `integration_status` 改为相应状态，并在报告中如实标注后端与数据来源。
