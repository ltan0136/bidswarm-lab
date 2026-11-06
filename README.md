# BidSwarm Lab

研究多个"策略研发"搜索者如何通过实验、经验交换、记忆和资源分配，共同改进**同一个广告主**的自动出价策略。

- **蜂群在"寻找策略"一侧**：多个搜索者提出候选策略、花预算做实验、根据结果改进。
- **"执行策略"一侧保持简单稳定**：一个只依赖 numpy 的数值出价函数（`bid = λ_t × cpa_target × p_adj`），
  可以在 AuctionNet 的 Python 3.9 进程里逐时间步运行（在线仿真后端已接入）。

v0.1 是**工程底座 + 可信的"独立搜索"对照组**，外加一个最小的经验交换模式 `cooperative`（环形精英迁移）。没有实现 LLM 或记忆，也不要求搜索结果优于基线。

## 当前能做什么 / 不能做什么

| 能做（已实现并有测试） | 不能做 / 未实现 |
|---|---|
| 内置 toy 环境：形状对齐 AuctionNet 的 `bidding()` 8 个参数，共同随机数（CRN），与 AuctionNet 一样只对曝光收费 | AuctionNet 只接入了**在线仿真**（`integration_status=integrated_online`）；离线评测（需约 80GB 数据）未接入 |
| AuctionNet 在线仿真后端：固定上游 commit，在上游 Python 3.9 子进程中运行，与上游 `run_test` 逐字一致（[docs/AUCTIONNET_INTEGRATION.md](docs/AUCTIONNET_INTEGRATION.md)） | 每个投放周期约 45–50 秒、每进程 2.7 GB；toy 规模的实验在一台 4 核机器上不可行，**还没有跑过任何 AuctionNet 搜索实验** |
| 两个策略模板：FixedRate（1 维）、DualControl（8 维），可移植内核兼容 Python 3.9 | 只有一种交换：环形迁移各自的 best-so-far；没有全连接 / 随机配对拓扑，不共享失败区域或步长 |
| (1+λ)-ES、CMA-ES 与随机采样 proposer；`single`、`independent`、`cooperative` 三种模式 | 没有 LLM proposer、没有记忆、没有动态预算 |
| 同步轮次调度、静态预算账本、谱系记录、进程池并行（spawn） | 没有硬超时（软超时只标记、不杀进程），没有重试 |
| 训练 → 验证选择 → 单独的测试评估；配对 bootstrap CI + 置换检验 + Holm 校正的报告 | 没有 HTML 报告、没有数据库 |
| 统计功效分析：A/A 校准、最小可检出效应、阳性对照（`scripts/power_analysis.py`） | toy 上 R = 20 只能检出 ≥ 3.6 分的效应；比较迁移效果需 R = 100（MDE 约 1.4 分） |
| 评分公式复刻 AuctionNet PlayerAnalysis（NeurIPS 竞赛版），已与上游数值核对 | toy 结果**不代表** AuctionNet 表现；AuctionNet 结果也**不代表**真实广告收益 |

## 安装

需要 Python ≥ 3.11（用到 `tomllib`）。只在虚拟环境里安装，不装全局依赖：

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[dev]"
```

运行依赖只有 numpy，开发依赖只有 pytest。Windows 下把 `.venv/bin/` 换成 `.venv\Scripts\`。

## 命令

```bash
.venv/bin/python -m bidswarm doctor
```

```bash
.venv/bin/python -m bidswarm run --config configs/toy_smoke.toml
```

```bash
.venv/bin/python -m bidswarm evaluate --run-dir runs/<name>-<UTC 时间戳>
```

```bash
.venv/bin/python -m bidswarm report --run-dir runs/<name>-<UTC 时间戳>
```

- `doctor`：打印 Python 与 numpy 版本、内核 Python 3.9 语法检查结果、AuctionNet 的 `integration_status`、固定的上游 commit 与已知限制。不安装任何东西。
- `run --config <path> [--serial] [--max-workers N] [--out runs]`：搜索 + 验证集选择，**不执行测试**。

三种模式在同一训练 episode 预算下比较：`single` 是 1 个搜索者；`independent` 是 N 个互不通信的搜索者；
`cooperative` 与 `independent` 相同（同样的 N、预算和随机数流），只是每 `exchange_interval` 轮（默认 1）把每个搜索者
自己的 best-so-far 沿环 w0→w1→…→w0 送给下一个搜索者，更优的外来候选会取代 ES 父代（σ 不变；CMA-ES 下成为下一代的均值）。迁移不消耗 episode，
最后一轮之后不迁移。报告给出 `independent − single`（只改变搜索者数）与 `cooperative − independent`（只改变是否迁移）两组配对比较。
- `evaluate --run-dir <dir> [--serial] [--max-workers N] [--force]`：读取已固定的选定策略，在测试场景上评估它们和两个未调优基线，写入 `test_episodes.jsonl` 并刷新报告。已有测试结果时需 `--force` 才会重跑。
- `report --run-dir <dir>`：只从产物重建 `report.md`。

`configs/` 下的配置：

| 配置 | 用途 |
|---|---|
| `toy_smoke.toml` | 冒烟测试：R=3，预算 96，三种模式；run + evaluate 串行约 13 秒，4 进程约 5 秒 |
| `toy_experiment.toml` | toy 上的三模式对照：R=20，预算 192（只够检出 ≥ 3.6 分的效应） |
| `toy_experiment_random.toml` | 上一行的 random A/A 对照（各模式在分布上等价） |
| `toy_cooperative.toml` | 同 `toy_experiment.toml`，但 R=100（比较迁移效果用），4 进程约 5 分钟 |
| `toy_cooperative_random.toml` | 上一行的 random A/A 对照 |
| `toy_cooperative_cmaes.toml` | 同 `toy_cooperative.toml`，但 proposer = CMA-ES，用于与 ES 按 replicate 配对比较（结果见 [docs/TOY_RESULTS.md](docs/TOY_RESULTS.md) 第 6 节） |
| `power/*.toml` | 功效分析用的 R=100 数据池（ES / random 预算 192；ES 预算 96、32） |
| `auctionnet_smoke.toml` | AuctionNet 端到端冒烟：R=1，32 个投放周期，4 进程约 8 分钟；需先按 [docs/AUCTIONNET_INTEGRATION.md](docs/AUCTIONNET_INTEGRATION.md) 准备上游环境并设置 `BIDSWARM_AUCTIONNET_REPO` / `BIDSWARM_AUCTIONNET_PYTHON` |
| `auctionnet_pilot.toml` | 第一版 AuctionNet 三模式实验设计（R=10，约 3,760 个周期，4 进程约 13–14 小时）；**尚未运行** |
| `auctionnet.example.toml` | 字段说明示例，`repo_path` 是占位路径，运行时报 `AuctionNetSetupError`（不回退 toy） |

统计功效分析（只读产物，完整参数见 [docs/POWER_ANALYSIS.md](docs/POWER_ANALYSIS.md)）：

```bash
.venv/bin/python scripts/power_analysis.py --aa runs/power/power_random_b192-<时间戳> --es runs/power/power_es_b192-<时间戳>
```

两个共用种子与场景的运行（例如不同 proposer）按 replicate 与模式配对比较测试分和训练 best-so-far：

```bash
.venv/bin/python scripts/compare_runs.py runs/toy_cooperative_cmaes-<时间戳> runs/toy_cooperative-<时间戳>
```

## 产物

每次 `run` 写入独立目录 `runs/<name>-<UTC 时间戳>/`（`runs/` 已在 `.gitignore` 中）：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 配置快照、git commit（如有）、Python/numpy 版本、backend、integration_status、预算换算 |
| `scenarios.json` | 每个 replicate 的 train/validation/test 场景（ID、种子、广告主预算/CPA/类别） |
| `candidates.jsonl` | 每个候选的谱系与训练 fitness（parent_id、operator、params、normalized、sigma、info_sources…） |
| `episodes.jsonl` | 训练与验证阶段的每个 episode |
| `anytime.jsonl` | 每轮结束时每个搜索者及模式整体的 best-so-far 与已用训练 episode 数 |
| `exchanges.jsonl` | `cooperative` 的每次迁移：轮次、发送 / 接收搜索者、候选 ID 与 fitness（其他模式为空文件） |
| `budget.json` | 预算账本：planned / used / failed / timeouts / retries |
| `selected/<replicate>_<mode>.json` | 固定下来的选定策略：spec、spec_hash、候选短名单与验证分、选择规则 |
| `summary.csv` | 每个 (replicate, 模式/基线) 一行 |
| `test_episodes.jsonl` | `evaluate` 之后才有 |
| `report.md` | 只从上述产物生成 |

所有 JSON 都是严格 JSON：`json.dumps(allow_nan=False)`，非有限值写成 `null`。所有 toy 输出都带
`backend=toy`、`purpose=framework_smoke_test`、`score_formula=auctionnet_neurips_replica` 标记；AuctionNet 输出带
`backend=auctionnet`、`purpose=auctionnet_online_simulation`，以及上游 commit、玩家位置和 episode 序号。

## 测试

```bash
.venv/bin/python -m pytest
```

核心测试总耗时约 10 秒，AuctionNet 后端的协议与失败处理用一个桩 runner（`tests/auctionnet_stub_runner.py`）测试，不需要上游代码。
`test_kernel_compat.py` 中两个用真实 `python3.9` 编译的测试只在 PATH 上有 `python3.9` 时运行。
`tests/test_auctionnet_integration.py` 对真实 AuctionNet 跑 3 个周期（约 3 分钟），只在设置了
`BIDSWARM_AUCTIONNET_REPO` 与 `BIDSWARM_AUCTIONNET_PYTHON` 时运行；与上游 `run_test` 的逐字比对见 `scripts/auctionnet_smoke.py`。

## 目录结构

```
src/bidswarm/
  cli.py, config.py, types.py, experiment.py, diagnostics.py
  kernel/        可移植策略内核（Python 3.9，numpy + 标准库）：spec.py, policies.py, registry.py
  environments/  base.py, toy.py, auctionnet.py（客户端）, auctionnet_runner.py（在上游 Python 3.9 中运行）
  search/        proposers.py, worker.py, coordinator.py, exchange.py
  evaluation/    metrics.py, splits.py, budget.py, runner.py, selection.py
  artifacts/     store.py, report.py
configs/  configs/power/  scripts/power_analysis.py  scripts/compare_runs.py  scripts/auctionnet_probe.py  scripts/auctionnet_smoke.py  tests/  docs/
```

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：寻找策略与执行策略的分离、模块边界、轮次调度、toy 环境
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)：预算、配对设计、选择规则、测试隔离、统计方法、R 的选择
- [docs/POWER_ANALYSIS.md](docs/POWER_ANALYSIS.md)：A/A 校准、最小可检出效应、阳性对照
- [docs/TOY_RESULTS.md](docs/TOY_RESULTS.md)：已跑过的 toy 实验记录与关键数字
- [docs/AUCTIONNET_INTEGRATION.md](docs/AUCTIONNET_INTEGRATION.md)：已核实的上游事实、开放问题、接入计划
- [docs/ROADMAP.md](docs/ROADMAP.md)：后续方向
