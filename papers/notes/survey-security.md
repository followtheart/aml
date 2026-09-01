# 精读笔记：A Survey on Long-Term Memory Security in LLM Agents

- 出处：arXiv:2604.16548（MemTensor + 上海交大, 2026）；论文：`papers/survey-security.pdf`
- 一句话：可写、跨会话持久记忆带来全新威胁面（持久性、状态性、传播性）；提出记忆生命周期框架：六阶段（WRITE / STORE / RETRIEVE / EXECUTE / SHARE&PROPAGATE / FORGET&ROLLBACK）× 四目标（完整性/机密性/可用性/治理）。

## 对 AML 方案的直接价值（安全与隐私维度）

1. **记忆投毒防御**：对话内容中可能藏注入指令——Add 抽取时应把消息内容当作**数据**而非指令（提示词中明确隔离），AMU content 禁止包含指令式文本。
2. **RETRIEVE 阶段安全**：AML 要求禁止跨 user_id 检索——存储层行级隔离（方案已采纳）。
3. **FORGET&ROLLBACK**：版本链 + 软废止天然支持回滚与审计。
4. **治理目标**：AML 评测的"记忆治理"与"认知安全"维度与此框架高度重叠——方案可把六阶段 checklist 作为合规自查表。
5. 平台红线（提示词注入、数据泄漏即取消资格）使安全设计不仅是得分点、更是参赛资格问题。
