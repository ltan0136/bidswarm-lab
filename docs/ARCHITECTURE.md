# 架构

## 1. 两侧分离：寻找策略 vs 执行策略

```
          寻找策略（本仓库的研究对象，可以复杂）                执行策略（必须简单、稳定、可移植）
 ┌─────────────────────────────────────────────────┐      ┌──────────────────────────────────┐
 │ Coordinator（普通程序，不是大模型）               │      │ bidswarm.kernel                  │
 │   ├─ SearchWorker × N  ── Proposer（ES / 随机）   │ spec │   PolicySpec = name + params     │
 │   ├─ BudgetLedger（静态扣预算）                   │ ───▶ │   FixedRatePolicy / DualControl  │
 │   ├─ ExchangePolicy（v1: NoExchange）             │ JSON │   bidding(8 个 AuctionNet 参数)  │
 │   └─ 选择 / 测试隔离 / 产物写入                    │      │   Python 3.9 + numpy + 标准库    │
 └─────────────────────────────────────────────────┘      └──────────────────────────────────┘
```

- 搜索者之间交换的是**数据**（`PolicySpec`：模板名 + 数值参数），不是代码。`PolicySpec` 只能引用注册表里的模板，
  参数必须有限且在边界内；内核从不 `eval` 或按名字动态导入。
- 执行侧只有一个统一形式：`bid = λ_t × cpa_target × p_adj`，`p_adj = max(pValue − ρ·pValueSigma, 0)`。
  不同模板只是 λ_t 和 ρ 的计算方式不同。
- 将来 AuctionNet 侧只需要一个很薄的 shim：读取 JSON 中的 `PolicySpec`，调用 `kernel.make_policy`，
  在每个时间步把 `bidding()` 转发给内核（见 [AUCTIONNET_INTEGRATION.md](AUCTIONNET_INTEGRATION.md)）。

## 2. 模块边界

| 模块 | 职责 | 依赖规则 |
|---|---|---|
| `kernel/` | `PolicySpec`、校验、`spec_hash`、策略模板、归一化空间互转 | **兼容 Python 3.9**；只依赖 numpy + 标准库；不导入 `bidswarm` 其他子包 |
| `environments/` | 后端接口 `Environment`；`toy` 实现；`auctionnet` 占位（构造即抛 `BackendNotIntegratedError`） | 只依赖 kernel 外的 `types`、`evaluation.metrics` |
| `evaluation/` | 评分公式复刻、场景划分、预算换算与账本、episode 任务函数与执行器、选择规则 | 任务函数是纯函数，可在 spawn 子进程运行 |
| `search/` | proposer、worker（搜索者）、coordinator（轮次调度）、exchange 接口 | 不读取验证 / 测试场景 |
| `artifacts/` | 运行目录、严格 JSON 写入、只读产物的报告生成 | 报告不访问内存状态 |
| `experiment.py` | `run`（搜索 + 验证选择）与 `evaluate`（测试）的编排 | 测试只在 `evaluate` 中出现 |
| `cli.py` | `doctor / run / evaluate / report` | — |

内核的兼容性规则由 `tests/test_kernel_compat.py` 和 `bidswarm doctor` 共同检查：
`ast.parse(src, feature_version=(3, 9))`、每个文件以 `from __future__ import annotations` 开头、
导入白名单（numpy + 标准库 + 内核内相对导入）、禁止 `dataclass(slots=/kw_only=)` 与 `zip(strict=)`。
`ast.parse` 只检查语法，**不能**发现 3.10+ 的标准库 API 调用；PATH 上有 `python3.9` 时会额外在真实 3.9 解释器上编译内核。

## 3. 同步轮次调度

```
for round r = 0, 1, ...:                       # 所有 (replicate, mode) 共用一批；轮数少的模式提前退出
  ① 每个搜索者基于自己的历史 propose(k)       ──▶ candidates（带谱系：parent_id / operator / info_sources）
  ② 按 (replicate, mode, worker, candidate, scenario) 生成 EpisodeTask
  ③ ledger.charge(...)  # 派发前静态扣预算，超出即报错
  ④ EpisodeExecutor.run(tasks)  # 串行或 ProcessPoolExecutor(spawn)；结果按 task_index 排序
  ⑤ 每个候选 fitness = 训练场景主指标均值（任一 episode 失败 → null）
  ⑥ 每个搜索者只 observe 自己候选的结果
  ⑦ ExchangePolicy.exchange(r, 本轮证据卡)  # v1: NoExchange，什么也不送达
  ⑧ 写 candidates.jsonl / episodes.jsonl / anytime.jsonl
搜索结束 → 每个 (replicate, mode) 取训练 top-K → 验证场景评估 → 固定 selected/<replicate>_<mode>.json
（evaluate 命令另行读取 selected/ 并在测试场景上评估）
```

- 所有产物由协调器所在的主进程单点写入，子进程只返回结果字典。
- `run_episode_task(task)` 是模块级纯函数：结果只依赖任务内容（`wall_time_s` 除外），因此串行与并行结果逐字一致
  （有测试覆盖）。
- 子进程使用 spawn 上下文，与 Windows 行为一致；`python -m bidswarm` 的入口有 `__main__` 守卫。

### 随机数

| 用途 | 来源 |
|---|---|
| 场景种子 | `sha256("bidswarm-scenario:<master_seed>:<split>:<index>")` 取 63 位 |
| 广告主（budget / cpa_target / category） | `default_rng([scenario_seed, 2**31 − 1])` |
| 第 t 步的流量与均匀数 | `default_rng([scenario_seed, t])`，固定抽取顺序，与策略无关（CRN） |
| 搜索者 | `sha256("bidswarm-proposer:<master_seed>:<mode>:<worker_id>")` 派生，模式之间互相独立 |

## 4. toy 环境

形状对齐 AuctionNet：一个 episode = 一个投放周期，`num_time_steps` 个时间步（默认 48），
每步 `opportunities_per_step` 个广告机会（默认 300）。

- **外生量**（每个机会）：`pValue ~ Beta(a, b)`；`pValueSigma = pValue × U(sigma_ratio_range)`；
  `p_true = clip(pValue + pValueSigma·z, 0, 1)`；
  `least_winning_cost = price_scale × 时段因子(t) × (pValue / E[pValue])^elasticity × exp(noise·ε)`（对数正态，与 pValue 正相关）；
  以及曝光均匀数、转化均匀数、预算截断顺序键。
- **拍卖**：单坑位二价。`bid > least_winning_cost` 获胜并支付 `least_winning_cost`；获胜后以 `exposure_prob` 曝光；
  曝光且转化均匀数 `< p_true` 记一次转化。**这是对上游多坑位机制的简化**；toy 中按获胜（而非曝光）计费。
- **预算硬上限**：某步获胜花费超过剩余预算时，按预先抽取的顺序键依次保留获胜机会，只要累计花费不超过剩余预算；
  其余视为未获胜。因此花费永远不超过预算（有测试覆盖）。
- **观察**：只传 `bidding()` 的 8 个参数。历史为逐步列表：
  `historyPValueInfo` 每步 `(n, 2) = [pValue, sigma]`，`historyBid (n,)`，
  `historyAuctionResult (n, 3) = [is_win, slot, cost]`（toy 中 slot 在获胜时为 1，否则为 0），
  `historyImpressionResult (n, 2) = [is_exposed, conversion]`，`historyLeastWinningCost (n,)`。
  不传未来流量、`p_true` 或竞争者内部状态。
- `category` 在 toy 中只作为元数据传给策略，不影响流量。
- **episode 输出**：cost、budget、cpa_target、conversions（实际）、expected_conversions（获胜且曝光机会的 pValue 之和）、
  cpa_realized / cpa_expected（零转化时为 `null`，`cpa_defined=false` / `cpa_expected_defined=false`）、
  score_realized / score_expected（上游公式，单个 episode 不除以 20000）、budget_utilization、wins、opportunities、
  cpa_ok、status（ok / failed / timeout）、wall_time_s。策略抛异常、输出形状不对、含非有限值或负值时，episode 记为 `failed`，
  所有指标为 `null`。

## 5. 策略模板

两种模板都从构造参数读取 `budget`、`cpa_target`、`category`，不硬编码广告主数值；已花费金额和期望转化
从历史**增量**计算（每次只处理新增的历史项，O(T)）。剩余预算 ≤ 0 时全部出价为 0；没有机会时返回空数组；
输入含 NaN/Inf 时抛 `PolicyInputError`。

**FixedRatePolicy**：λ ≡ `ratio ∈ [0.3, 3.0]`，ρ = 0，对应上游 Abid 基线。

**DualControlPolicy**（8 维）：

| 参数 | 范围 | 默认（未调优） |
|---|---|---|
| base_ratio | [0.3, 3] | 1.0 |
| cpa_gain | [0, 2] | 1.0 |
| pace_gain | [0, 2] | 1.0 |
| pace_shape | [0.5, 2] | 1.0 |
| lambda_min | [0.1, 1] | 0.2 |
| lambda_max | [1, 5] | 3.0 |
| sigma_discount（ρ） | [0, 1.5] | 0.0 |
| smoothing | [0, 0.9] | 0.5 |

```
plan_frac   = (t / T) ^ pace_shape
spend_frac  = spent / budget
cpa_est     = spent / expected_conversions          （期望转化为 0 时跳过 CPA 项）
λ_target    = base_ratio · exp(−cpa_gain·(cpa_est/cpa_target − 1)) · exp(−pace_gain·(spend_frac − plan_frac))
λ_t         = clip(smoothing·λ_{t−1} + (1 − smoothing)·λ_target, lambda_min, lambda_max)
λ_{−1}      = clip(base_ratio, lambda_min, lambda_max)
```

指数项的参数被裁剪到 [−50, 50] 以保证有限。`lambda_min ≤ lambda_max` 会被校验；由于两者边界在 1 处相接，
边界内的任何点都满足该约束，因此归一化空间 [0, 1]^8 中的每个点都对应一个合法策略。
