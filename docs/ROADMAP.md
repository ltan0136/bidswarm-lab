# 路线图

v0.1 建了工程底座和"独立搜索"对照组；v0.2 加入 `cooperative`（环形精英迁移）与统计功效分析。
除第 1 节注明已实现的部分外，以下方向都**尚未实现**，排列大致按依赖顺序。
每一项都应保持当前协议：同一预算单位、配对场景、训练 / 验证 / 测试隔离、产物驱动的报告。

## 0. 前置：AuctionNet 接入

三步计划（只读调查 → shim → 子进程冒烟测试）已完成，AuctionNet **在线仿真**后端可用（固定上游 commit `c20de63`，
见 [AUCTIONNET_INTEGRATION.md](AUCTIONNET_INTEGRATION.md)）。内核记账与 toy 都已改为按曝光收费，toy 结果已重跑。
下一步是 `configs/auctionnet_pilot.toml`（R = 10，约 13–14 小时），并先在 AuctionNet 上估计最小可检出效应。
在 AuctionNet 实验跑出来之前，下面所有方向的结果都只是 toy 上的框架验证。

## 1. 信息交换

- 已实现：`cooperative` 模式 = `RingExchange`（每 `exchange_interval` 轮把各自的 best-so-far 送给环上下一个搜索者，
  更优者取代 ES 父代）。`info_sources` 记录外来候选 ID，`exchanges.jsonl` 记录每次送达，
  报告给出 `cooperative − independent` 的配对比较（同预算、同随机数流）。
- 待做：其他拓扑（全连接、随机配对）、转发外来精英、共享失败区域或步长（σ）、按迁移间隔与 N 做扫描。
- 已实现的第二种 proposer：`cmaes`。迁移在 CMA-ES 下只在代与代之间生效（外来精英成为下一代均值），
  所以同样的 `exchange_interval` 对两种 proposer 的实际作用不同。
- 检验强度：toy 上迁移带来的测试分差若与"训练预算加倍"相当（约 1.1 ~ 1.7 分）或更小，需要 R ≥ 100 才有约 80% 功效（MDE 约 1.4 分）
  （见 [POWER_ANALYSIS.md](POWER_ANALYSIS.md)）；扫描多种拓扑 / 间隔时要控制多重比较。

## 2. LLM proposer

- 作为 `Proposer` 的一个实现：输入为自己（及允许看到的）候选历史，输出为 `PolicySpec`（纯数据，经校验），不输出代码。
- 需要额外记录：模型名、提示词版本、调用次数与费用（作为单独的预算维度），以及失败 / 非法输出的处理。
- 本仓库目前不调用任何付费模型。
- 对照组：策略空间是固定的 8 维有界实数，这类空间里经典优化器常强于 LLM 提议者，较好的做法是混合
  （例如把 CMA-ES 的内部状态交给 LLM，[arXiv:2603.24647](https://arxiv.org/abs/2603.24647)）。LLM proposer 至少要与
  ES、CMA-ES、random 在同一 episode 预算下比较；toy 上 CMA-ES 与 ES 的配对结果见 [TOY_RESULTS.md](TOY_RESULTS.md) 第 6 节。
- 成本单位：除 episode 外按 token 记账并报告；带历史的提示每次调用更贵，只按调用数对齐会让历史更多的方法
  多拿算力（[arXiv:2610.23977](https://arxiv.org/abs/2610.23977)）。还应加一个“不看历史的 LLM 独立采样”对照。

## 3. 记忆

- 跨 replicate 或跨广告主保留经验（例如过去有效的参数区域），并保证不泄漏验证 / 测试信息。
- 需要定义记忆的写入时机（只在训练阶段）与审计方式（记忆内容进入产物）。

## 4. 动态预算

- 当前预算是静态的：每个搜索者固定 `per_worker` 个候选。动态分配（例如按 anytime 曲线把预算移给更有希望的搜索者）
  需要在账本中记录每次再分配，并保持总 `used ≤ planned`。
- 可能需要异步调度与锁；v1 刻意没有实现。

## 5. 环境变化

- 非平稳流量（价格、pValue 分布、竞争强度随周期变化）、多个广告主、跨类别迁移。
- toy 环境中的 `category` 目前只是元数据，可扩展为影响流量的因素，用来做可控的分布漂移实验。
