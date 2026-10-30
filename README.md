# BidSwarm Lab

研究多个"策略研发"搜索者如何通过实验、经验交换、记忆和资源分配，共同改进**同一个广告主**的自动出价策略。

- **蜂群在"寻找策略"一侧**：多个搜索者提出候选策略、花预算做实验、根据结果改进。
- **"执行策略"一侧保持简单稳定**：一个只依赖 numpy 的数值出价函数（`bid = λ_t × cpa_target × p_adj`），
  将来要在 AuctionNet 的 Python 3.9 进程里逐时间步运行。

v0.1 是**工程底座 + 可信的"独立搜索"对照组**，外加一个最小的经验交换模式 `cooperative`（环形精英迁移）。没有实现 LLM 或记忆，也不要求搜索结果优于基线。

## 当前能做什么 / 不能做什么

| 能做（已实现并有测试） | 不能做 / 未实现 |
|---|---|
| 内置 toy 环境：形状对齐 AuctionNet 的 `bidding()` 8 个参数，共同随机数（CRN） | **AuctionNet 后端未接入**（`integration_status=not_integrated`，选择即报错，不回退 toy） |
| 两个策略模板：FixedRate（1 维）、DualControl（8 维），可移植内核兼容 Python 3.9 | 只有一种交换：环形迁移各自的 best-so-far；没有全连接 / 随机配对拓扑，不共享失败区域或步长 |
| (1+λ)-ES 与随机采样 proposer；`single`、`independent`、`cooperative` 三种模式 | 没有 LLM proposer、没有记忆、没有动态预算 |
| 同步轮次调度、静态预算账本、谱系记录、进程池并行（spawn） | 没有硬超时（软超时只标记、不杀进程），没有重试 |
| 训练 → 验证选择 → 单独的测试评估；配对 bootstrap CI + 置换检验 + Holm 校正的报告 | 没有 HTML 报告、没有数据库 |
| 统计功效分析：A/A 校准、最小可检出效应、阳性对照（`scripts/power_analysis.py`） | toy 上 R = 20 只能检出 ≥ 3.5 分的效应；比较迁移效果需 R = 100 |
| 评分公式复刻 AuctionNet PlayerAnalysis（NeurIPS 竞赛版） | toy 结果**不代表** AuctionNet 表现或真实广告收益 |

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

- `doctor`：打印 Python 与 numpy 版本、内核 Python 3.9 语法检查结果、AuctionNet 的 `integration_status=not_integrated` 及阻塞原因。不安装任何东西。
- `run --config <path> [--serial] [--max-workers N] [--out runs]`：搜索 + 验证集选择，**不执行测试**。

三种模式在同一训练 episode 预算下比较：`single` 是 1 个搜索者；`independent` 是 N 个互不通信的搜索者；
`cooperative` 与 `independent` 相同（同样的 N、预算和随机数流），只是每 `exchange_interval` 轮（默认 1）把每个搜索者
自己的 best-so-far 沿环 w0→w1→…→w0 送给下一个搜索者，更优的外来候选会取代 ES 父代（σ 不变）。迁移不消耗 episode，
最后一轮之后不迁移。报告给出 `independent − single`（只改变搜索者数）与 `cooperative − independent`（只改变是否迁移）两组配对比较。
- `evaluate --run-dir <dir> [--serial] [--max-workers N] [--force]`：读取已固定的选定策略，在测试场景上评估它们和两个未调优基线，写入 `test_episodes.jsonl` 并刷新报告。已有测试结果时需 `--force` 才会重跑。
- `report --run-dir <dir>`：只从产物重建 `report.md`。

`configs/` 下的配置：

| 配置 | 用途 |
|---|---|
| `toy_smoke.toml` | 冒烟测试：R=3，预算 96，单机串行约 10 秒 |
| `toy_experiment.toml` | toy 上的三模式对照：R=20，预算 192（只够检出 ≥ 3.5 分的效应） |
| `toy_experiment_random.toml` | 上一行的 random A/A 对照（各模式在分布上等价） |
| `toy_cooperative.toml` | 同 `toy_experiment.toml`，但 R=100（比较迁移效果用），4 进程约 5 分钟 |
| `toy_cooperative_random.toml` | 上一行的 random A/A 对照 |
| `power/*.toml` | 功效分析用的 R=100 数据池（ES / random 预算 192；ES 预算 96、32） |
| `auctionnet.example.toml` | 示例；**运行时必然报 `BackendNotIntegratedError`** |

统计功效分析（只读产物，完整参数见 [docs/POWER_ANALYSIS.md](docs/POWER_ANALYSIS.md)）：

```bash
.venv/bin/python scripts/power_analysis.py --aa runs/power/power_random_b192-<时间戳> --es runs/power/power_es_b192-<时间戳>
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
`backend=toy`、`purpose=framework_smoke_test`、`score_formula=auctionnet_neurips_replica` 标记。

## 测试

```bash
.venv/bin/python -m pytest
```

核心测试总耗时约 4 秒。`test_kernel_compat.py::test_kernel_compiles_on_real_python39` 只在 PATH 上有 `python3.9` 时运行，否则跳过。

## 目录结构

```
src/bidswarm/
  cli.py, config.py, types.py, experiment.py, diagnostics.py
  kernel/        可移植策略内核（Python 3.9，numpy + 标准库）：spec.py, policies.py, registry.py
  environments/  base.py, toy.py, auctionnet.py（占位，构造即报错）
  search/        proposers.py, worker.py, coordinator.py, exchange.py
  evaluation/    metrics.py, splits.py, budget.py, runner.py, selection.py
  artifacts/     store.py, report.py
configs/  configs/power/  scripts/power_analysis.py  tests/  docs/
```

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：寻找策略与执行策略的分离、模块边界、轮次调度、toy 环境
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)：预算、配对设计、选择规则、测试隔离、统计方法、R 的选择
- [docs/POWER_ANALYSIS.md](docs/POWER_ANALYSIS.md)：A/A 校准、最小可检出效应、阳性对照
- [docs/TOY_RESULTS.md](docs/TOY_RESULTS.md)：已跑过的 toy 实验记录与关键数字
- [docs/AUCTIONNET_INTEGRATION.md](docs/AUCTIONNET_INTEGRATION.md)：已核实的上游事实、开放问题、接入计划
- [docs/ROADMAP.md](docs/ROADMAP.md)：后续方向
