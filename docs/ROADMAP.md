# 路线图

v0.1 只建工程底座和"独立搜索"对照组。以下方向都**尚未实现**，排列大致按依赖顺序。
每一项都应保持当前协议：同一预算单位、配对场景、训练 / 验证 / 测试隔离、产物驱动的报告。

## 0. 前置：AuctionNet 接入

见 [AUCTIONNET_INTEGRATION.md](AUCTIONNET_INTEGRATION.md) 的三步计划（只读调查 → shim → 子进程冒烟测试）。
在此之前，下面所有方向的结果都只是 toy 上的框架验证。

## 1. 信息交换

- 已有接口：`search/exchange.py` 中的 `ExchangePolicy.exchange(round_index, outgoing)` 在每轮边界被调用；
  `EvidenceCard` 只有最小字段（candidate_id、source_worker、round、spec_hash、fitness）；`NoExchange` 什么也不送达；
  `SearchWorker.receive_evidence` 在 v1 中收到任何卡片都会报错。
- 待做：定义交换拓扑（全连接、环、随机配对）、共享内容（最优 spec、失败区域、步长）、
  proposer 如何使用外来证据（例如作为候选父代或调整 σ）。
- 谱系的 `info_sources` 字段届时将记录外来候选 ID，用于分析信息流向；隔离测试需要相应改为"只含允许的来源"。
- 需要新的对照：交换 vs 不交换，在相同训练 episode 预算下比较。

## 2. LLM proposer

- 作为 `Proposer` 的一个实现：输入为自己（及允许看到的）候选历史，输出为 `PolicySpec`（纯数据，经校验），不输出代码。
- 需要额外记录：模型名、提示词版本、调用次数与费用（作为单独的预算维度），以及失败 / 非法输出的处理。
- 本仓库目前不调用任何付费模型。

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
