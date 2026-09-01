# 精读笔记总览：论文 → AML 方案模块映射

> 33 篇精读笔记的索引与综合。整理日期：2026-09-01
> 论文 PDF 在 `../`，文本在 `../text/`，笔记在 `./`。
> Nemori 一篇未能在 arXiv 找到，未下载（影响最小，可在第二期前再补）。

## 一、方案模块 ← 论文来源对照表

| AML 方案模块 | 主要借鉴 | 次要借鉴 |
|---|---|---|
| AMU 原子事实 + 字段设计 | A-MEM（笔记结构）、EverMemOS（MemCell 三合一） | MIRIX（类型分类学） |
| 双时间轴 valid_from/to | **Zep**（bi-temporal + 边失效） | MemOS（生命周期） |
| 版本链治理（软废止） | Zep（edge invalidation）、Mem0（ADD/UPDATE/DELETE/NOOP） | Memory-R1（RL 印证操作集） |
| episode 兜底双存储 | Zep（episode 子图）、structural-memory（混合结构韧性） | SeCom（段级粒度） |
| 多字段联合 embedding | A-MEM（content+keywords+tags+context 拼接编码） | LongMemEval（fact-augmented key） |
| 图谱路召回（PPR 多跳） | **HippoRAG / HippoRAG 2** | G-Memory（双向遍历） |
| 查询理解（时间锚定/子问题） | LongMemEval（time-aware query expansion） | HippoRAG（query NER） |
| 重排前相关性过滤 | Memory-R1（Answer Agent 预筛选） | structural-memory（rerank 增益） |
| preference/rule 直通召回 | MemoryOS（长期个人记忆层）、Dynamic Cheatsheet（小抄） | MemoryBank（画像综合） |
| 抽取上下文（摘要+近期窗口） | Mem0（S + m=10） | MemAgent（滚动摘要） |
| 滚动 session 摘要 | MemAgent（分块覆盖更新） | Mem0（异步摘要刷新） |
| 安全与隐私设计 | **survey-security**（生命周期×目标矩阵） | MIRIX（Knowledge Vault 分库） |
| 本地评测 Judge | **Mem0 附录 A**（含时间宽容规则） | LongMemEval（judge 一致性数据） |

## 二、提示词资产清单（可直接改写复用）

1. **AMU 抽取提示词** ← A-MEM Ps1 的 JSON 结构（keywords/context/tags）+ 事实原子化约束 + 时间归一化指令（Zep t_ref 锚定）。
2. **治理判定提示词** ← Mem0 四操作 function call + A-MEM Ps3 结构化决策（should_evolve/actions/connections）。
3. **查询理解提示词** ← LongMemEval time-aware expansion + HippoRAG query NER + 意图分类（七维度）。
4. **本地评测 Answer/Judge 提示词** ← Mem0 附录 A 全套（含相对时间宽容、矛盾取最新、5-6 词答案约束）。

## 三、关键反面教训（避坑）

1. **纯图检索牺牲事实召回**（HippoRAG 2 的修复动机）→ 向量永远作第一召回路，图谱只作扩展。
2. **整体重写导致上下文坍塌**（ACE 对 Dynamic Cheatsheet 的批评）→ 治理只做条目级增量更新。
3. **遗忘曲线与事实召回冲突**（MemoryBank 的拟人化在评测场景有害）→ 衰减只做排序微调，不做淘汰。
4. **DELETE 硬删除丢历史**（Mem0）→ 改为软废止，时序题需要历史版本。
5. **多 agent/重图架构成本不可控**（MIRIX、全量 LLM 判定）→ LLM 调用全部前置 Add 侧，Search 侧仅查询理解 + 轻量重排两次调用。

## 四、笔记索引（按方案相关性排序）

**必读核心**：a-mem / zep / mem0 / hipporag
**评测与维度**：longmemeval / locomo / secom / structural-memory / evo-memory
**经验与进化**：expel / agent-workflow-memory / reasoningbank / ace / memory-r1 / dynamic-cheatsheet / voyager
**系统架构**：memgpt / evermemos / memoryos / memos / mirix / g-memory / memorybank / hipporag2
**参数化（视野）**：larimar / memoryllm / m-plus / mem1 / memagent / titans
**综述**：survey-memory-mechanism / survey-human-to-ai / survey-security
