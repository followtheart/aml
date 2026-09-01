# 精读笔记：MemoryOS — Memory OS of AI Agent

- 出处：EMNLP 2025 Oral（arXiv:2506.06326，北邮+腾讯 AI Lab）；论文：`papers/memoryos.pdf`
- 一句话：OS 启发的三级存储——短期记忆（对话链 FIFO）→ 中期记忆 → 长期个人记忆，四大模块：Storage / Updating / Retrieval / Generation。

## 核心机制

- **热-温-冷三级**：短期（当前会话对话链，FIFO 淘汰）→ 中期（按热度合并升级）→ 长期（个人记忆：画像+偏好持续更新）。
- 存储单元间动态升级/降级（类 OS 页面置换）。

## 对 AML 方案的可复用点

- 分层热度管理思想与 MemGPT 一致；AML 可简化为两级：AMU（长期）+ 会话内近期消息缓存（短期，用于抽取上下文）。
- **长期个人记忆独立成层**：再次印证 preference/profile 应独立类型存储并直通召回。
- 腾讯 AI Lab 作品——腾讯也是 AML 第一期商业榜参赛方，代表国内头部团队的路线选择。
