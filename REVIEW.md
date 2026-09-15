# AML v0.4 设计一致性复核

> 范围：`agent-memory-system-design.md` v0.4、ULM 设计、`memory_system/app/*`、`prompts/*`
>
> 结论：核心生命周期已形成可测试闭环；研究性扩展与生产规模能力明确留在交付边界之外。

## 已修复的高风险问题

| 项目 | 当前状态 |
|---|---|
| Embedding 截断/失败污染 | 非原生维度直接失败；真实 provider 失败不写哈希向量；按 `embedding_space` 隔离 |
| Search 时间锚点 | 请求无显式时间时锚定用户最新记忆时间 |
| RRF rewrite 偏置 | rewrite 先在路由内合并，每条路由只投一票 |
| 重排成本与量纲混排 | 仅融合头部单批评分；失败时整次回退同一 RRF 排序 |
| 时序版本 | 显式 `valid_from/valid_to` 过滤，维护双向 supersession 链，图边保留有效区间 |
| 摘要污染证据 | session summary 不作为可返回证据，只作写入上下文 |
| 敏感信息泄漏 | 存储边界默认拒绝；部署开关与请求字段双重授权；不进入共享图或 Scene |
| Debug 合规 | 全文日志默认关闭；用户 purge 同步清理数据库和已启用 JSONL |
| Profile 生命周期 | static/transient/stable 持久化；transient TTL；跨会话支持晋升 stable |
| Scene 生命周期 | 有界 Heat、hot/cold 分层、命中复活，强度单位一致 |
| 经验记忆 | `/feedback` 接收显式结果，蒸馏 procedural AMU；skill 需环境验证 |
| 检索视图 | 事实类意图使用 Memory-Only，叙事/文档意图使用 Memory-Doc |
| Add 并发与幂等 | 同用户串行、revision 冲突重试、request 所有权校验 |
| MemCell | topic shift 分段；episode 保存叙事与压缩视图，原文由 provenance 持有 |

## 当前一致性矩阵

| 设计能力 | 实现状态 | 验证位置 |
|---|---|---|
| 多路召回：dense/sparse/graph/temporal/profile/scene | 已实现 | `scripts/selftest_search.py`、`scripts/selftest_ulm.py` |
| 证据归因与来源预算 | 已实现 | `scripts/selftest_evidence.py` |
| 四操作治理与安全 SUPERSEDE | 已实现 | `scripts/selftest_storage.py`、`scripts/selftest_evidence.py` |
| 版本链与历史时间点查询 | 已实现 | `scripts/selftest_search.py`、`scripts/selftest_evidence.py` |
| Vault 双授权 | 已实现 | `scripts/selftest_search.py` |
| Profile TTL/晋升 | 已实现 | `scripts/selftest_ulm.py` |
| Scene 遗忘/复活 | 已实现 | `scripts/selftest_ulm.py` |
| 显式经验反馈 | 已实现 | `scripts/selftest_experience.py` |
| 用户硬删除 | 已实现 | `scripts/selftest_storage.py`、`scripts/selftest_contract.py` |
| 簇内 A-Mem 邻居演化 | 可选、默认关闭；需消融证明收益 | 不属于 AML v0.4 |
| KV cache 激活记忆 | 可选研究插槽 | 不属于 AML v0.4 |
| 参数化记忆 | 可选研究插槽 | 不属于 AML v0.4 |

## 剩余缺陷与生产风险

### P1：SQLite 仍是单机参考后端

WAL、用户 revision 和进程内锁能保证当前单进程语义，但不能提供多实例请求 claim、跨节点锁或 ANN 索引。生产部署仍需 Postgres/pgvector，并把 request ownership 和 revision 检查放入数据库事务。

### P1：图检索仍在请求内构图

当前 PPR 使用用户图数据在请求期计算；用户三元组规模增大后会成为内存与延迟瓶颈。生产实现应采用稀疏邻接、局部 k-hop 子图或增量图缓存，并增加规模基准。

### P1：向量路仍为精确扫描

SQLite 参考实现按用户读取候选并做 NumPy 相似度计算，正确但不可水平扩展。需要 pgvector HNSW/IVFFlat，并验证 recall/latency 参数。

### P2：治理 LLM 调用按事实串行

新事实命中近邻时仍可能逐条调用治理模型。可以在不改变四操作契约的前提下做批量判定；优化前先记录真实 Add 的事实数和命中率。

### P2：自动保留期依赖部署调度

系统已有显式用户 purge 和 transient profile TTL，但“到期自动硬删除”没有内置调度器。生产环境需由外部作业调用 purge/retention 任务，并监控删除收据与失败重试。

### P2：缺少生产数据规模基准

自测覆盖正确性与回归，不代表 LoCoMo 全量质量、真实 provider 延迟或并发容量。发布门槛应补充：Recall@100、MRR、时间题准确率、Add/Search p95、模型调用量、数据库增长和 purge 完整性。

## 发布判定

AML v0.4 可作为单机研究参考实现和离线评测基线。进入多租户生产前，必须完成 Postgres/pgvector 迁移、数据库级幂等、保留期调度、安全日志接入和规模压测；A-Mem 邻居演化、激活记忆与参数化记忆不应阻塞发布，也不应在没有消融证据时默认开启。