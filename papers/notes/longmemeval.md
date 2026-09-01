# 精读笔记：LongMemEval — Benchmarking Chat Assistants on Long-Term Interactive Memory

- 出处：ICLR 2025（arXiv:2410.10813）；论文：`papers/longmemeval.pdf`
- 一句话：500 题评测五大能力——信息抽取、多会话推理、时序推理、**知识更新**、**弃权（abstention）**；商业系统在持续交互上准确率掉 30%。

## 对 AML 最重要的三个设计优化（论文实验验证有效）

1. **Session decomposition（值粒度）**：不要整段会话作为一个记忆值——把会话分解为小粒度单元（≈ 方案的 AMU 原子事实 + episode 双层）。
2. **Fact-augmented key expansion（索引键扩展）**：索引时把原始消息扩展为"提取出的事实"作为检索键——即检索键与存储值解耦。方案中 AMU content 是陈述句事实、episode 保留原文，正是此思想；可进一步给 AMU 增加"检索键"字段（问题形式改写，如 "When did Dave start photography?"），提升问句-事实匹配率。
3. **Time-aware query expansion（时间感知查询扩展）**：检索前把查询中的相对时间展开为绝对时间范围，缩小搜索范围——方案 5.1 查询理解的时间锚定直接对应，是论文实测提升召回的关键手段。

## 弃权（abstention）维度

- 评测明确考察"记忆中没有答案时不编造"。对应方案 5.4：低置信返回空 data 数组。
- 论文的 GPT-4o Judge 在各题型上与人类判断一致率 ≥90%（弃权题最难，0.90-0.97）——本地评测时弃权题要单独看分。

## 知识更新维度

- 考察"旧信息被新信息覆盖后回答最新值"。方案的版本链 + Search 默认只返回最新有效版本即为此设计；本地评测需专门构造此类题验证治理模块。

## 三阶段框架

- 论文把长期记忆设计统一分解为 indexing / retrieval / reading 三阶段——方案的 Add 管线（indexing）、Search 管线（retrieval）、平台 Answer（reading，平台固定）恰好一一对应，可作为方案文档的叙述框架。
