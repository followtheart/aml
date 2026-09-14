# 大语言模型智能体记忆与上下文管理方法综述

> 整理自 DeepSeek 分享对话（30 篇论文阅读笔记）。每篇论文均有独立笔记，点击方法名即可跳转。
>
> - **笔记目录**：[`notes/`](notes/)
> - **数据来源**：对话原始提取文本 `../deepseek_share_full.txt`

## 笔记索引

| 分类 | 论文笔记 |
| --- | --- |
| 外部记忆与 OS 式管理 | [MemGPT](notes/17-memgpt.md) · [MemoryBank](notes/18-memorybank.md) · [Mem0](notes/14-mem0.md) · [A-Mem](notes/03-a-mem.md) · [EverMemOS](notes/05-evermemos.md) · [MemoryOS](notes/20-memoryos.md) · [MemOS](notes/22-memos.md) · [MIRIX](notes/23-mirix.md) · [Zep](notes/30-zep.md) · [SeCom](notes/26-secom.md) · [Memory-R1](notes/21-memory-r1.md) |
| 经验学习与策略记忆 | [ACE](notes/01-ace.md) · [AWM](notes/02-agent-workflow-memory.md) · [ExpeL](notes/07-expel.md) · [Dynamic Cheatsheet](notes/04-dynamic-cheatsheet.md) · [ReasoningBank](notes/25-reasoningbank.md) · [Voyager](notes/29-voyager.md) · [G-Memory](notes/08-g-memory.md) |
| 图结构与检索增强 | [HippoRAG](notes/09-hipporag.md) · [HippoRAG 2](notes/10-hipporag2.md) · [Zep](notes/30-zep.md) · [A-Mem](notes/03-a-mem.md) |
| 参数化 / 潜在空间记忆 | [MemoryLLM](notes/19-memoryllm.md) · [M+](notes/24-m-plus.md) · [Larimar](notes/11-larimar.md) · [Titans](notes/28-titans.md) · [MEM1](notes/15-mem1.md) · [MemAgent](notes/16-memagent.md) |
| 评测基准与结构研究 | [LoCoMo](notes/12-locomo.md) · [LongMemEval](notes/13-longmemeval.md) · [Evo-Memory](notes/06-evo-memory.md) · [Structural Memory](notes/27-structural-memory.md) |

---

## 1. 引言

大语言模型（LLM）虽具备强大的语言理解与生成能力，但部署后通常是无状态的，受限于固定上下文窗口，难以在长程对话、持续任务和开放环境中保持一致性、积累经验与个性化。近年来，围绕“如何为 LLM 智能体构建记忆”涌现出大量工作，涵盖外部记忆系统、上下文工程、经验学习、知识编辑、参数化记忆、图检索、强化学习驱动管理以及评测基准等方向。本综述对上述 30 篇论文进行系统梳理，总结其核心思想、技术路线、关键差异、主要缺陷与演进趋势。

## 2. 记忆系统的基本维度

理解这些方法可从以下维度切入：

- **存储介质**：明文文本、向量数据库、知识图谱、潜在状态、模型参数。
- **时间尺度**：短期（当前会话）、中期（主题片段）、长期（跨会话/跨任务）。
- **生命周期**：生成、组织、检索、更新、遗忘、演化。
- **控制方式**：启发式规则、LLM 驱动、强化学习驱动、操作系统式调度。
- **评估方式**：精确匹配、LLM-as-Judge、多跳/时序/知识更新等任务。

## 3. 主要技术路线

### 3.1 外部记忆系统与操作系统式管理

这类方法将记忆视为可显式管理的外部资源，强调分层存储、动态更新和检索。

- [MemGPT](notes/17-memgpt.md)：提出“LLM 作为操作系统”的虚拟上下文管理，将上下文窗口视为物理内存，外部存储视为磁盘，通过函数调用在两者之间分页，支持多会话对话和长文档分析。
- [MemoryBank](notes/18-memorybank.md)：引入艾宾浩斯遗忘曲线，构建分层事件摘要和用户画像，支持记忆的选择性遗忘与强化。
- [Mem0](notes/14-mem0.md)：面向生产环境，通过“提取—更新—检索”流程维护外部记忆，支持 ADD/UPDATE/DELETE/NOOP 操作，并可扩展为图记忆。
- [A-Mem](notes/03-a-mem.md)：受 Zettelkasten 启发，将记忆组织为原子笔记，通过动态链接和记忆演化形成互联知识网络。
- [EverMemOS](notes/05-evermemos.md)：提出三阶段记忆生命周期：情景痕迹形成、语义巩固、重建性回忆，模拟生物记忆印迹。
- [MemoryOS](notes/20-memoryos.md)：借鉴操作系统段页式管理，设计 STM/MTM/LPM 三级存储，使用热度分数进行段驱逐和长期画像更新。
- [MemOS](notes/22-memos.md)：提出记忆操作系统，统一明文记忆、激活记忆和参数记忆，通过 MemCube 封装与调度，支持跨类型转换与治理。
- [MIRIX](notes/23-mirix.md)：设计六种专门记忆组件（Core、Episodic、Semantic、Procedural、Resource、Knowledge Vault），由多智能体协调路由与主动检索，并支持多模态。
- [Zep](notes/30-zep.md)：基于时序知识图谱 Graphiti，采用双时间线建模和边失效机制，动态合成对话与业务数据，支持时序推理和知识更新。
- [SeCom](notes/26-secom.md)：研究记忆粒度，提出片段级记忆构建与压缩去噪，平衡轮次级碎片化和会话级噪声。
- [Memory-R1](notes/21-memory-r1.md)：用强化学习训练记忆管理器和答案智能体，学习 ADD/UPDATE/DELETE/NOOP 操作及记忆蒸馏。

### 3.2 经验学习、技能与策略记忆

这类方法强调从智能体自身经验中提取可复用知识，而非仅存储原始对话。

- [ACE](notes/01-ace.md)：将上下文视为持续演化的“战术手册”，通过生成器—反思器—策展人三模块增量更新上下文，解决简洁性偏见与上下文崩溃问题。
- [AWM](notes/02-agent-workflow-memory.md)：从轨迹中归纳可复用工作流（文本描述 + 抽象化动作轨迹），支持离线/在线两种模式，作为记忆指导未来任务。
- [ExpeL](notes/07-expel.md)：从成功和失败轨迹中提取自然语言洞察，测试时检索相关洞察和成功轨迹。
- [Dynamic Cheatsheet](notes/04-dynamic-cheatsheet.md)：在测试时维护动态“备忘单”，存储策略、代码片段和问题解决经验。
- [ReasoningBank](notes/25-reasoningbank.md)：从成功和失败经验中提炼推理策略，并引入记忆感知测试时缩放（MATTS），形成记忆与缩放的协同。
- [Voyager](notes/29-voyager.md)：在 Minecraft 中实现终身学习，包含自动课程、可执行代码技能库和迭代提示机制。
- [G-Memory](notes/08-g-memory.md)：面向多智能体系统，构建洞察图、查询图和交互图三层记忆，支持角色定制化检索与更新。

### 3.3 图结构与检索增强记忆

这类方法利用知识图谱或图算法提升多跳推理和关联检索能力。

- [HippoRAG](notes/09-hipporag.md)：受海马体索引理论启发，用 LLM 构建开放知识图谱，并通过 Personalized PageRank 实现单步多跳检索。
- [HippoRAG 2](notes/10-hipporag2.md)：在 HippoRAG 基础上引入段落节点、Query-to-Triple 匹配和 LLM 识别记忆过滤，提升事实记忆、意义构建和关联性。
- [Zep](notes/30-zep.md) 与 [A-Mem](notes/03-a-mem.md) 也属于图结构记忆，但侧重点不同：Zep 强调时序知识图谱，A-Mem 强调笔记链接与演化。

### 3.4 参数化 / 潜在空间记忆与知识编辑

这类方法将记忆内化到模型参数或潜在状态中，减少对外部检索的依赖。

- [MemoryLLM](notes/19-memoryllm.md)：在 Transformer 每层嵌入固定大小记忆池，通过自更新和随机丢弃实现知识注入与指数遗忘。
- [M+](notes/24-m-plus.md)：扩展 MemoryLLM，引入长期记忆和协同训练检索器，将知识保留从 20k 扩展到 160k+ token。
- [Larimar](notes/11-larimar.md)：用外部情景记忆矩阵控制 LLM 解码，支持一次性知识编辑、选择性遗忘和信息泄漏防护。
- [Titans](notes/28-titans.md)：提出神经长期记忆模块，在测试时通过梯度下降学习记忆，结合动量和遗忘机制，可扩展到 2M 上下文。
- [MEM1](notes/15-mem1.md)：通过强化学习训练智能体在长程多轮任务中维持恒定内存，将推理与记忆整合为内部状态。
- [MemAgent](notes/16-memagent.md)：用多轮对话 RL 训练记忆智能体，采用覆盖式记忆更新，实现从 8K 到 3.5M 的外推。

### 3.5 记忆评测基准与结构研究

- [LoCoMo](notes/12-locomo.md)：超长期对话记忆评测基准，平均 300 轮、9K token。
- [LongMemEval](notes/13-longmemeval.md)：面向聊天助手的长期交互记忆基准，覆盖五类核心记忆能力，历史长度可扩展至 1.5M token。
- [Evo-Memory](notes/06-evo-memory.md)：自演化记忆的流式基准与统一评估框架，将静态数据集重组为任务序列。
- [Structural Memory](notes/27-structural-memory.md)：系统比较 chunks/triples/facts/summaries 等记忆结构与检索方法的研究。

## 4. 方法对比概览（含“主要缺陷”列）

### 4.1 外部记忆与操作系统式管理

| 方法 | 记忆形式 | 更新/检索机制 | 关键创新 | 主要缺陷 |
| --- | --- | --- | --- | --- |
| [MemGPT](notes/17-memgpt.md) | 分层文本 + 函数调用 | 队列管理器 + 分页 | OS 式虚拟上下文管理 | 依赖函数调用能力，弱模型效果差；检索器质量影响大；控制流启发式；多步延迟与成本高 |
| [MemoryBank](notes/18-memorybank.md) | 摘要 + 事件 + 画像 | 遗忘曲线 + 双塔检索 | 心理学遗忘机制 | 遗忘模型简化；依赖人工标注；画像质量依赖 LLM；检索单一；未评估长期演化 |
| [Mem0](notes/14-mem0.md) | 事实 + 图记忆 | 提取-更新-检索 | 生产级记忆操作 | 评估范围有限；图版本 token 翻倍、延迟略高；时间推理绝对值不高；依赖 LLM 操作选择 |
| [A-Mem](notes/03-a-mem.md) | 原子笔记 + 链接 + 演化 | LLM 链接生成 | Zettelkasten 式记忆网络 | 链接生成开销大，误差累积；依赖 LLM 质量；可能过度连接 |
| [EverMemOS](notes/05-evermemos.md) | MemCell + MemScene | 三阶段生命周期 | 印迹启发记忆巩固 | LLM 中介操作增加延迟与成本；评估仅文本；超长时间线未充分测试 |
| [MemoryOS](notes/20-memoryos.md) | STM/MTM/LPM | 段页式 + 热度驱逐 | OS 段页管理 | 段划分依赖 LLM；热度公式手工调参；评估以对话为主；画像维度固定 |
| [MemOS](notes/22-memos.md) | 明文/激活/参数 | MemCube 调度 | 统一记忆操作系统 | 系统复杂度高；跨 LLM 记忆共享未验证；自演化 MemBlock 未实现；评估多对话，多模态待扩展 |
| [MIRIX](notes/23-mirix.md) | 六种记忆组件 | 多智能体路由 + 主动检索 | 多模态多组件记忆 | 依赖强函数调用 LLM；路由可能出错；开放域仍依赖 RAG；评估样本有限；隐私与安全挑战 |
| [Zep](notes/30-zep.md) | 时序知识图谱 | 双时间线 + 边失效 | 时序感知图记忆 | 单会话助手问题下降；DMR 基准不足；社区动态扩展近似；交叉编码器成本高；仅对话记忆评估 |
| [SeCom](notes/26-secom.md) | 片段级记忆 | 对话分割 + 压缩去噪 | 记忆粒度与去噪 | 分割依赖 LLM；压缩率需调优；未探索复杂记忆结构；评估范围有限 |
| [Memory-R1](notes/21-memory-r1.md) | 外部记忆库 | RL 训练操作与蒸馏 | RL 驱动记忆管理 | 评估聚焦对话；两阶段分离训练复杂；依赖 RAG 检索质量；奖励设计有权衡 |

### 4.2 经验、技能与策略记忆

| 方法 | 记忆形式 | 更新/检索机制 | 关键创新 | 主要缺陷 |
| --- | --- | --- | --- | --- |
| [ACE](notes/01-ace.md) | 结构化上下文（战术手册） | 生成-反思-策展增量更新 | 智能体上下文工程 | 依赖反馈信号质量，可能被误导信息污染；简单任务收益有限；在线学习与机器遗忘待探索 |
| [AWM](notes/02-agent-workflow-memory.md) | 工作流 | 归纳 + 检索 | 流程复用 | 工作流动作僵化，动态环境失败；在线噪音风险；仅任务重复性子流程适用 |
| [ExpeL](notes/07-expel.md) | 洞察 + 成功轨迹 | 检索 + 回忆 | 经验学习 | 仅文本观察；依赖闭源 API；上下文窗口限制；缺乏理论保证；经验收集成本高 |
| [Dynamic Cheatsheet](notes/04-dynamic-cheatsheet.md) | 备忘单 | 测试时更新 | 测试时学习 | 小模型效果差；记忆维护挑战；检索噪音；成本开销 |
| [ReasoningBank](notes/25-reasoningbank.md) | 推理策略 | 成功/失败策略检索 + MATTS | 记忆与缩放协同 | 聚焦记忆内容，检索/巩固简单；依赖 LLM-as-Judge；模块化组合未探索；评估范围 Web/SWE |
| [Voyager](notes/29-voyager.md) | 可执行代码技能库 | 自动课程 + 迭代提示 | 终身学习智能体 | GPT-4 成本高；不准确/幻觉；缺乏视觉感知；需人类反馈辅助 3D |
| [G-Memory](notes/08-g-memory.md) | 三层图 | 双向遍历 + 角色定制 | 多智能体记忆 | 评估任务多样性有限；LLM 调用成本；依赖基础 LLM 质量；安全风险 |

### 4.3 图结构与检索增强记忆

| 方法 | 记忆形式 | 更新/检索机制 | 关键创新 | 主要缺陷 |
| --- | --- | --- | --- | --- |
| [HippoRAG](notes/09-hipporag.md) | 知识图谱 | KG + PPR | 单步多跳检索 | 概念-上下文权衡，NER 错误约 48%；OpenIE 质量瓶颈；图搜索误差；可扩展性待验证 |
| [HippoRAG 2](notes/10-hipporag2.md) | KG + 段落节点 | Query-to-Triple + 过滤 | 全面超越标准 RAG | 计算成本高；识别记忆精度仍有约 26% 丢失；图搜索误差；长文档 OpenIE 退化 |
| [Zep](notes/30-zep.md)（见表 4.1） | 时序知识图谱 | 双时间线 + 边失效 | 时序感知图记忆 | 同上 |
| [A-Mem](notes/03-a-mem.md)（见表 4.1） | 原子笔记 + 链接 + 演化 | LLM 链接生成 | Zettelkasten 式记忆网络 | 同上 |

### 4.4 参数化 / 潜在空间记忆与知识编辑

| 方法 | 记忆形式 | 更新/检索机制 | 关键创新 | 主要缺陷 |
| --- | --- | --- | --- | --- |
| [MemoryLLM](notes/19-memoryllm.md) | 潜在空间记忆池 | 自更新 + 随机丢弃 | 参数化可更新记忆 | 记忆容量固定；随机丢弃粗糙；训练成本高；评估范围有限 |
| [M+](notes/24-m-plus.md) | 短期 + 长期记忆 | 协同训练检索器 | 可扩展长期保留 | CPU-GPU 通信开销；短文档性能略降；训练成本高；未扩展到 128k；检索质量约 30% |
| [Larimar](notes/11-larimar.md) | 情景记忆矩阵 | 读写 + 遗忘 | 知识编辑与防泄漏 | 仅支持短事实；泛化能力有限；长上下文依赖递归；未验证问答摘要 |
| [Titans](notes/28-titans.md) | 神经长期记忆 | 测试时梯度更新 | 深度记忆模块 | 训练吞吐量较慢；深度记忆效率权衡；计算资源大；评估范围有限 |
| [MEM1](notes/15-mem1.md) | 内部状态 | RL 训练 | 恒定内存长程智能体 | 依赖可验证奖励；任务范围有限；注意力掩码近似；开放式任务难应用 |
| [MemAgent](notes/16-memagent.md) | 覆盖式记忆 | 多轮 RL | 超长上下文外推 | 依赖可验证奖励；训练数据合成；固定记忆容量瓶颈；评估范围有限 |

### 4.5 评测基准与结构研究

| 方法 | 类型 | 关键贡献 | 主要缺陷 | 适用场景 |
| --- | --- | --- | --- | --- |
| [LoCoMo](notes/12-locomo.md) | 评测基准 | 超长期对话记忆，平均 300 轮、9K token | 混合人机生成，多模态有限，仅英语，依赖闭源模型 | 长期对话记忆评估 |
| [LongMemEval](notes/13-longmemeval.md) | 评测基准 | 五类核心记忆能力，历史长度可扩展至 1.5M token | 证据会话 LLM 模拟，评估依赖 LLM-judge，未评估删除，商业系统规模小 | 聊天助手长期记忆评估 |
| [Evo-Memory](notes/06-evo-memory.md) | 评测基准 | 流式自演化记忆，任务序列化 | API 成本限制，模态局限，记忆质量评估间接，隐私安全 | 自演化记忆评估 |
| [Structural Memory](notes/27-structural-memory.md) | 结构研究 | 系统比较记忆结构与检索方法 | 任务范围有限，噪声类型单一，超参数范围受限 | 记忆结构设计指导 |

## 5. 关键趋势与洞察

1. **从静态 RAG 到动态记忆**：早期方法将记忆视为静态文档检索，近年工作强调动态更新、时序推理和知识演化。
2. **从单一结构到分层/多组件**：MemoryOS、MemOS、MIRIX、EverMemOS 等采用多级存储或多类型记忆，模拟人类记忆系统。
3. **从启发式到 LLM/RL 驱动**：Mem0、A-Mem 依赖 LLM 决策，Memory-R1、MEM1、MemAgent 则用强化学习优化记忆管理。
4. **从外部到内化/参数化**：MemoryLLM、M+、Larimar、Titans 将记忆嵌入潜在空间或参数，减少检索开销。
5. **从文本到图/时序/多模态**：Zep、HippoRAG、A-Mem 使用图结构，MIRIX 支持多模态，Zep 强调时序。
6. **评估走向长程、流式、多任务**：LoCoMo、LongMemEval、Evo-Memory 等基准推动更真实的评估。

## 6. 挑战与未来方向

- **容量与遗忘的平衡**：如何在有限记忆容量下保留关键信息并遗忘冗余。
- **冲突检测与知识更新**：动态环境中事实可能矛盾，需要时序建模和失效机制。
- **多模态记忆**：图像、音频、视频等模态的记忆组织与检索仍待探索。
- **隐私、安全与权限**：长期记忆涉及敏感信息，需要访问控制和遗忘机制。
- **端到端训练与 RL**：如何将记忆管理与推理更紧密地联合优化。
- **统一评估**：现有基准各有侧重，仍需更全面、可扩展的评测体系。

## 7. 结论

上述工作共同勾勒出 LLM 智能体记忆研究的完整图景：从 MemGPT 的操作系统式分页，到 MemoryBank 的遗忘曲线；从 Mem0、A-Mem、EverMemOS 的外部记忆管理，到 MemoryLLM、M+、Titans 的参数化/潜在空间记忆；从 ExpeL、Dynamic Cheatsheet、ReasoningBank 的经验学习，到 Voyager 的技能库；从 HippoRAG、Zep 的图检索，到 Memory-R1、MEM1、MemAgent 的强化学习驱动；再到 LoCoMo、LongMemEval、Evo-Memory 的评测基准。整体趋势是：记忆正从外挂式检索走向系统化、内化、自演化和多组件协同，成为 LLM 智能体实现长程一致性、持续学习与个性化的核心基础设施。同时，各方法在依赖模型能力、计算成本、评估范围、安全隐私等方面仍存在明显缺陷，未来需要在容量管理、冲突消解、多模态、端到端优化和统一评测等方向持续突破。

---

## 附：全部笔记链接

1. [ACE — Agentic Context Engineering](notes/01-ace.md)
2. [AWM — Agent Workflow Memory](notes/02-agent-workflow-memory.md)
3. [A-Mem — Agentic Memory for LLM Agents](notes/03-a-mem.md)
4. [Dynamic Cheatsheet — Test-Time Learning with Adaptive Memory](notes/04-dynamic-cheatsheet.md)
5. [EverMemOS — Self-Organizing Memory OS](notes/05-evermemos.md)
6. [Evo-Memory — Benchmarking Test-Time Learning](notes/06-evo-memory.md)
7. [ExpeL — LLM Agents Are Experiential Learners](notes/07-expel.md)
8. [G-Memory — Hierarchical Memory for Multi-Agent Systems](notes/08-g-memory.md)
9. [HippoRAG — Neurobiologically Inspired Long-Term Memory](notes/09-hipporag.md)
10. [HippoRAG 2 — From RAG to Memory](notes/10-hipporag2.md)
11. [Larimar — LLMs with Episodic Memory Control](notes/11-larimar.md)
12. [LoCoMo — Very Long-Term Conversational Memory 评测](notes/12-locomo.md)
13. [LongMemEval — Long-Term Interactive Memory 评测](notes/13-longmemeval.md)
14. [Mem0 — Production-Ready Scalable Long-Term Memory](notes/14-mem0.md)
15. [MEM1 — Synergize Memory and Reasoning](notes/15-mem1.md)
16. [MemAgent — Multi-Conv RL-Based Memory Agent](notes/16-memagent.md)
17. [MemGPT — Towards LLMs as Operating Systems](notes/17-memgpt.md)
18. [MemoryBank — Long-Term Memory with Forgetting Curve](notes/18-memorybank.md)
19. [MemoryLLM — Self-Updatable LLMs](notes/19-memoryllm.md)
20. [MemoryOS — Memory OS of AI Agent](notes/20-memoryos.md)
21. [Memory-R1 — RL for Memory Management](notes/21-memory-r1.md)
22. [MemOS — A Memory OS for AI System](notes/22-memos.md)
23. [MIRIX — Multi-Agent Memory System](notes/23-mirix.md)
24. [M+ — Scalable Long-Term Memory](notes/24-m-plus.md)
25. [ReasoningBank — Reasoning Memory + MATTS](notes/25-reasoningbank.md)
26. [SeCom — Segment-level Memory + Compression Denoising](notes/26-secom.md)
27. [Structural Memory of LLM Agents](notes/27-structural-memory.md)
28. [Titans — Learning to Memorize at Test Time](notes/28-titans.md)
29. [Voyager — Open-Ended Embodied Agent](notes/29-voyager.md)
30. [Zep — Temporal Knowledge Graph for Agent Memory](notes/30-zep.md)
