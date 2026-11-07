# CLAUDE.md — BidSwarm Lab

多个搜索者在同一训练预算下寻找自动出价策略的实验框架。行为和设计以 `README.md` 和 `docs/` 为准
（尤其是 `docs/ARCHITECTURE.md` 与 `docs/EXPERIMENT_PROTOCOL.md`）；本文件只管一件事：
**改代码时不偏离这些协议**。这是研究代码，跑偏的代价不是报错，而是一个看起来正常、实际上不可信的结论。

## 命令

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -rs          # 约 5 秒；AuctionNet 与 python3.9 相关测试在缺环境时跳过
.venv/bin/python -m bidswarm doctor     # 内核 3.9 语法检查、后端状态
```

改动 `search/`、`evaluation/`、`environments/`、`experiment.py` 或 `artifacts/` 后，再跑一遍 toy 冒烟（与 CI 相同）：

```bash
.venv/bin/python -m bidswarm run --config configs/toy_smoke.toml --out runs
run_dir=$(ls -d runs/toy_smoke-* | tail -1)
.venv/bin/python -m bidswarm evaluate --run-dir "$run_dir"
.venv/bin/python -m bidswarm report --run-dir "$run_dir"
```

被跳过的测试不等于通过：汇报时写明跳过了哪些、为什么。

## 工作方式

1. **非小改动先出计划。** 动手前列出：要改的文件、新增或修改的函数/类及签名、是否影响结果（见下文）、
   要加的测试。计划涉及"受保护的协议"或超过 3 个文件时，停下等确认。
2. **只做任务要求的事。** 不顺手重构、改名、重新格式化没动的代码；想到的改进写进汇报里当建议。
3. **先复用再新增。** 扩展已有模块（`kernel/registry.py`、`search/proposers.py`、`search/exchange.py`、
   `evaluation/`），不另起一套平行实现。运行依赖只有 numpy、开发依赖只有 pytest，新增任何依赖先问。
4. **需求有歧义就问。** 必须继续时，在汇报里写明你采用了哪种理解。
5. **一个任务一个小 commit**，在分支上工作；不直接推 `main`，不 force push。commit 里不要加
   `Co-Authored-By: Claude` 或任何 Claude 署名行。
6. **不擅自启动 AuctionNet 实验。** 一个投放周期约 45–50 秒、每进程 2.7 GB，pilot 要十几个小时；需要跑时先问。

## 测试是合同

- 不许为了让测试通过而修改、删除、跳过、`xfail` 或放宽已有测试和断言。认为测试本身错了，就停下说明理由。
- 新行为要有测试。不 mock 被测对象本身，不把本该算出来的期望值写死。
- 很多协议靠测试守着（见下表）。改动若让这些测试失败，说明改动破坏了协议，不是测试过时。

## 错误必须看得见

- 宽泛的 `except Exception` 只出现在现有边界上，作用是把失败**记录**下来而不是吞掉：策略或环境出错 →
  `status="failed"` 的 episode（`evaluation/runner.py`、`environments/toy.py`），AuctionNet 子进程的协议层。
  新增宽泛捕获必须说明理由，并留下可审计的记录。
- 不静默回退。AuctionNet 配置不全时抛 `AuctionNetSetupError`，**绝不回退到 toy**。
- 失败的候选 fitness 为 `null`，照样计入预算；不要把失败改成 0 分或悄悄丢掉。

## 受保护的协议：改之前先问

| 协议 | 不变量 | 守护它的测试 |
|---|---|---|
| 两侧分离 | `kernel/` 兼容 Python 3.9，只依赖 numpy + 标准库，不导入 `bidswarm` 其他子包，每个文件以 `from __future__ import annotations` 开头；不用 3.10+ 的标准库 API（`ast` 检查发现不了这类问题） | `test_kernel_compat.py`、`bidswarm doctor` |
| 策略是数据 | 搜索者之间只交换 `PolicySpec`（模板名 + 有界数值参数），内核从不 `eval` 或按名字动态导入；以后的 LLM proposer 也只能输出 `PolicySpec` | `test_policies.py`、`kernel/spec.py` 的校验 |
| 测试隔离 | `search/` 不读验证或测试场景；`run` 不执行测试；测试只在 `evaluate` 中出现，且不回流到搜索 | `test_run_writes_artifacts_but_never_touches_test_split` |
| 预算 | 派发前静态扣预算，训练结束时 `used == planned`；三种模式同一训练 episode 预算；迁移不消耗 episode；不重试 | `test_budget_ledger_and_spend`、`test_raising_policy_is_failed_and_still_charged` |
| 配对与随机数 | 场景、广告主、搜索者种子都按 `ARCHITECTURE.md` 的 sha256 方案派生；replicate r 只取决于配置和 `seed + r`；不使用未设种子的全局随机数（`auctionnet_runner.py` 里的 `np.random.seed(1)` 是照搬上游，不要动） | `test_replicate_results_do_not_depend_on_number_of_replicates`、`test_splits_are_disjoint_and_paired` |
| 可复现 | 串行与并行结果逐字一致；`run_episode_task` 是纯函数；所有产物只由主进程写入 | `test_serial_and_parallel_results_are_identical` |
| 产物与报告 | 严格 JSON（`allow_nan=False`，非有限值写 `null`）；`report.md` 只从产物生成，可逐字重建 | `test_artifacts_are_strict_json`、`test_report_rebuilds_identically_from_artifacts` |
| 统计判定 | bootstrap 与置换检验的次数和固定种子、Holm 校正、"CI 不含 0 且校正后 p ≤ 0.05"的判定规则 | `test_report_stats.py` |
| 交换 | 迁移只在同一 (replicate, mode) 内发生；`cooperative` 与 `independent` 共用随机数流直到迁移 | `test_exchange.py`、`test_cooperative_*` |
| AuctionNet | 固定上游 commit；只接入在线仿真；不同 split 不共用 episode 序号 | `test_backend.py` |

## 研究诚信：最容易"跑偏得很像样"的地方

- **会改变结果的改动要明说。** 改环境、评分公式、计费规则、随机数派生、预算换算、选择规则，都会让旧数字
  不可比。必须在汇报里标为"会改变结果"，并在相关文档里注明从哪天起不可混比（参照 2026-11-06 改为按曝光收费的写法）。
- **不为显著性调统计。** 不改检验次数、种子、校正方法或阈值来让结论变成"高于 / 低于"；达不到判定标准就如实写"不能确认"。
- **文档里的数字只能来自真实运行。** 写进 `docs/TOY_RESULTS.md` 等文档的每个数字都要能追溯到一个
  `runs/` 目录和配置；不估算、不补数、不手改 `runs/` 里的产物。
- **不夸大结论。** toy 结果只验证框架，不代表 AuctionNet 表现；AuctionNet 结果也不代表真实广告收益。
- **random proposer 的比较只是正确性检查**（各模式在分布上等价），不能写成"某种模式更好"。
- **路线图上的方向（LLM proposer、记忆、动态预算、新拓扑）不在范围内**，除非明确要求；做的时候也必须保持
  同一预算单位、配对场景、训练 / 验证 / 测试隔离、产物驱动的报告。

## 代码风格

标识符、注释和 docstring 用英文；文档（`docs/`、README）用中文。公共函数写类型标注。
没有配置 ruff 等格式化工具，不要整体引入或批量重排格式。

## 每次任务结束时的汇报

1. 改了什么、为什么，按文件列出，附 `git diff --stat`。
2. 采用了哪些假设，哪里不确定。
3. 跑了哪些命令、结果如何，包括被跳过的测试。
4. 这次改动**是否会改变实验结果**；碰了上表哪些协议；有哪些选择不做的事。

## 从真实跑偏中学到的规则

同类跑偏出现第二次时，在这里加一条规则（能写成测试的就同时加测试）。
格式：`- YYYY-MM-DD — 出了什么问题 → 规则。`

- （暂无）
