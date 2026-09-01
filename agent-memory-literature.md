# LLM Agent 记忆前沿文献清单（2024–2026）

> 面向 AML 参赛系统设计的选读清单。优先顶会（NeurIPS / ICML / ICLR / ACL / AAAI / EMNLP），
> 辅以少量影响力大的 arXiv 系统论文。按设计相关性分组。
> 整理日期：2026-09-01

---

## 一、记忆系统架构（核心方法论）

| 论文 | 出处 | 与 AML 方案的关联 |
|---|---|---|
| **A-MEM: Agentic Memory for LLM Agents** (Xu et al.) | NeurIPS 2025 | Zettelkasten 式自组织记忆：结构化笔记 + 动态链接 + 记忆演化。方案中 AMU 的版本链/治理设计直接对标 |
| **HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs** (Gutiérrez et al.) | NeurIPS 2024 | LLM + 知识图谱 + Personalized PageRank，多跳 QA 提升约 20%。方案中"轻量图谱扩展召回"的理论依据 |
| **From RAG to Memory: Non-Parametric Continual Learning**（HippoRAG 2） | ICML 2025 | HippoRAG 续作，关联记忆任务 +7% |
| **MemGPT: Towards LLMs as Operating Systems** (Packer et al.) | ICLR 2024（arXiv 2023） | OS 式分层记忆 + 自我编辑，后续产品化为 Letta |
| **MemoryBank: Enhancing LLMs with Long-Term Memory** (Zhong et al.) | AAAI 2024 | 艾宾浩斯遗忘曲线启发的记忆衰减，时序排序参考 |
| **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** (Chhikara et al.) | ECAI 2025（arXiv） | 生产级事实抽取 + ADD/Update/Delete/NoOP 控制器。AML 商业榜头部系统，方案 4.3 治理模块的直接参照 |
| **MemoryOS / Memory OS of AI Agent** (Kang et al.) | EMNLP 2025 (Oral) | 热/温/冷三级分层记忆 |
| **MIRIX: Multi-Agent Memory System for LLM-Based Agents** (Wang et al.) | arXiv 2025 | 多智能体、多类型（多模态）记忆架构前沿 |
| **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** (Rasmussen et al.) | arXiv 2025 | 双时间轴时序知识图谱——方案 AMU 双时间轴（event_time / valid_from-to）的直接来源 |
| **G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems** | NeurIPS 2025 | 多智能体层级记忆 |
| **Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science** | arXiv 2025 | 认知科学启发的自组织记忆 |
| **MemOS: An Operating System for Memory-Augmented Generation** | arXiv 2025 | 记忆操作系统化 |
| **EverMemOS** | arXiv 2026 | Engram 启发自组织记忆；LoCoMo 93.05%、LongMemEval 83.00% 报告 SOTA |

## 二、经验记忆与自我进化（Coding Track 相关）

| 论文 | 出处 | 要点 |
|---|---|---|
| **ExpeL: LLM Agents Are Experiential Learners** (Zhao et al.) | AAAI 2024 | 从经验中提取可复用规则，调试经验复用的先驱 |
| **Agent Workflow Memory** (Wang et al.) | ICML 2025 | 工作流级记忆复用，对 SWEContextBench 类任务直接相关 |
| **ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory** | ICLR 2026 | 检索"推理策略"而非原始数据，自我进化记忆 |
| **Agentic Context Engineering (ACE): Evolving Contexts for Self-Improving LMs** | ICLR 2026 | 上下文作为可演化记忆体 |
| **Voyager: An Open-Ended Embodied Agent with LLMs** | TMLR 2024 | 技能库式经验积累 |
| **Memory-R1: Enhancing LLM Agents to Manage Memories via RL** (Yan et al.) | arXiv 2025 | 用 PPO/GRPO 训练记忆管理器，奖励来自下游 QA——写入侧控制的学习化方向 |
| **Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory** | arXiv 2025 | 测试时自适应记忆 |

## 三、参数化 / 架构级记忆（拓展视野）

| 论文 | 出处 | 要点 |
|---|---|---|
| **Larimar: LLMs with Episodic Memory Control** (Das et al.) | ICML 2024 | 架构级可编辑情景记忆，快速写入 |
| **MEMORYLLM: Towards Self-Updatable LLMs** | ICML 2024 | 自更新参数化记忆 |
| **M+: Extending MemoryLLM with Scalable Long-Term Memory** | arXiv 2025 | 上者续作，可扩展长期记忆 |
| **MEM1: Learning to Synergize Memory and Reasoning** | ICLR 2026 | RL 训练紧凑内部状态 |
| **MemAgent: Multi-Conv RL-based Memory Agent** | ICLR 2026 | RL 重塑长上下文记忆 |
| **Titans: Learning to Memorize at Test Time** (Behrouz et al.) | arXiv 2024（Google） | 测试时学习的神经记忆层 |

## 四、基准与评测（本地代理评测直接可用）

| 论文 | 出处 | 要点 |
|---|---|---|
| **LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents** (Maharana et al.) | ACL 2024 | 事实召回/多跳/时序/开放域，AML 文本榜核心数据集形态 |
| **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory** (Wu et al.) | ICLR 2025 | 五能力分解：信息抽取、跨会话、时序、知识更新、弃权——对应 AML 治理与安全维度 |
| **SeCom: Memory Construction and Retrieval for Long-Term Personalized Conversational Agents** | ICLR 2025 | 个性化会话记忆构建与检索 |
| **PersonaMem / "My agent understands me better"** | 2025 | 画像记忆评测 |
| **Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory** | arXiv 2025 | 自进化记忆评测 |
| **HaluMem** | 2025/2026 | 记忆幻觉评测（EverMemOS 报告 90.04% recall） |

## 五、综述（建立全局图谱）

| 论文 | 出处 | 要点 |
|---|---|---|
| **A Survey on the Memory Mechanism of LLM-based Agents** (Zhang et al.) | arXiv 2024（引用量高） | 记忆读写/反思的经典分类 |
| **From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs** (Wu et al.) | arXiv 2025 | 心理学对照分类 |
| **From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms** | Findings of ACL 2026 | 最新演化脉络综述 |
| **A Survey on the Security of Long-Term Memory in LLM Agents** | arXiv 2026 | 记忆投毒/隐私/主权——对应 AML 安全与隐私维度 |
| **On the Structural Memory of LLM Agents** (Zeng et al.) | arXiv 2024 | 记忆结构化选择的受控实证对比 |

---

## 阅读优先级建议（参赛视角）

1. **第一批（必读，直接决定架构）**：A-MEM、HippoRAG、Mem0、Zep —— 分别对应方案的动态链接、图谱召回、治理控制器、双时间轴。
2. **第二批（评测驱动调优）**：LoCoMo、LongMemEval、SeCom —— 决定本地代理评测管线。
3. **第三批（差异化与 Coding Track）**：Agent Workflow Memory、ReasoningBank、ExpeL、Memory-R1。
4. **第四批（防坑）**：记忆安全综述（投毒与隐私正是 AML 评测维度之一）。

## 来源

- [awesome-llm-agent-papers (GitHub)](https://github.com/js-lee-AI/awesome-llm-agent-papers)
- [Awesome-Long-Horizon-Agents (RUC-NLPIR)](https://github.com/RUC-NLPIR/Awesome-Long-Horizon-Agents)
- [awesome-agent-harness (RUCAIBox)](https://github.com/RUCAIBox/awesome-agent-harness)
- [Awesome-Agentic-Reasoning](https://github.com/weitianxin/Awesome-Agentic-Reasoning)
- [Zylos Research: Memory Consolidation in Long-Running AI Agents](https://zylos.ai/research/2026-04-20-memory-consolidation-ai-agents/)
- [rag-memory-playground THEORY.md](https://github.com/Qalipso/rag-memory-playground/blob/main/THEORY.md)
