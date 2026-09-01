# 精读笔记：LoCoMo — Evaluating Very Long-Term Conversational Memory of LLM Agents

- 出处：ACL 2024（arXiv:2402.17753）；论文：`papers/locomo.pdf`
- 一句话：超长对话记忆基准——每段对话约 300 轮 / 9K tokens / 最多 35 个会话，基于 persona 与时序事件图生成并人工校验；任务含 QA、事件摘要、多模态对话生成。

## 与 AML 的关系

- AML 文本榜的 request_id 中直接出现 `locomo` / `locomo_refined` 字样——LoCoMo 是平台核心数据集形态。本地代理评测必须以它为第一基准。
- QA 题型分类：single-hop、multi-hop、temporal、open-domain——对应 AML 的事实召回/多跳/时序维度。
- 论文结论：LLM 在长对话的**长程时序理解**上显著吃力——方案把时序列为差异化重点是正确的。

## 可复用要点

- 对话按"会话（session）"组织、带说话人双角色——Add 管线应按 session_id 组织 episode，抽取时区分说话人（Mem0 的 Answer 提示词特别提醒"区分记忆中提到的人名与实际用户"）。
- 事件以时序事件图组织——AMU 的 event_time 与实体图设计与之对齐。
- 评测含"对抗性/不可答"类问题（后续研究指出），Search 低置信时返回空数组的策略与此相容。

## 本地评测落地

- 数据集公开（GitHub: snap-research/locomo）；用 gpt-4o-mini 做 Answer 与 Judge，Judge 提示词用 Mem0 附录 A 模板（含时间宽容规则）。
