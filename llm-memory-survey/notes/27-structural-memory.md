# On the Structural Memory of LLM Agents

> 来源文件：`structural-memory.pdf` · 整理自 DeepSeek 对话阅读笔记 · [返回综述](../survey.md)

好的，我已仔细阅读并分析了您提供的论文《On the Structural Memory of LLM Agents》。

这是一篇来自格拉斯哥大学和阿伯丁大学的研究工作，系统性地研究了不同记忆结构

```
（memory  structures）和记忆检索方法（memory     retrieval methods）对LLM智能体性能的
影响。

以下是对该论文核心内容的系统分析与总结：

1. 核心问题与动机

论文指出现有LLM智能体记忆研究中，虽然提出了多种记忆模块，但不同记忆结构在不同任务

中的影响尚未被充分探索。具体来说：

  当前常见的记忆结构包括：文本块（chunks）、知识三元组（knowledge                triples）、原子

  事实（atomic  facts）、摘要（summaries）。

  不同结构各有特点：chunks保留原始上下文，triples捕捉实体关系，atomic             facts提供细粒
  度信息，summaries压缩全局内容。

  但哪种结构更适合哪类任务？它们对噪声的鲁棒性如何？不同检索方法如何与记忆结构配

  合？这些问题缺乏系统研究。

核心动机是首次全面研究记忆结构和检索方法在多种任务上的影响，为LLM智能体记忆系统设
计提供实证指导。

2. 提出的方法：结构记忆与检索框架

论文将记忆模块分为三个关键组件：结构记忆生成、记忆检索、答案生成。

2.1 四种结构记忆生成

记忆结构               定义                         生成方式

Chunks             固定长度连续文本段                  将原始文档按最多     L 个token切分

Knowledge Triples  ⟨head;relation;tail⟩ 三元组   LLM根据提示   P  抽取
                                                         T

                                                        P
Atomic Facts       最小不可分事实，简洁句子               LLM根据提示      生成
                                                         A
                                                        P
Summaries          压缩后的综合描述                   LLM根据提示     生成
                                                         S
                   上述四者的并集    MMixed = C ∪
                                q      q
Mixed                                         组合所有类型
                   T ∪ A ∪ S
                    q   q   q
2.2 三种记忆检索方法
1. Single-step Retrieval：直接检索Top-K最相关记忆。

2. Reranking：先检索Top-K候选，再用LLM根据相关性重排序，选Top-R。

3. Iterative Retrieval：迭代精炼查询，每次检索Top-T，用LLM改写查询，重复N轮后检索
  Top-K。

2.3 两种答案生成方式

  Memory-Only：直接用检索到的记忆作为上下文生成答案。

  Memory-Doc：用检索到的记忆定位原始文档，再用原始文档作为上下文生成答案。

3. 实验设置

  数据集：四个任务、六个数据集

    多跳QA：HotPotQA、2WikiMultihopQA、MuSiQue

    单跳QA：NarrativeQA

    对话理解：LoCoMo

    阅读理解：QuALITY

  评估指标：EM、F1、Accuracy

  实现：GPT-4o-mini-128k，温度0.2，输入窗口4k，chunk最大1k，text-embedding-3-

  small，LangChain。

4. 主要发现

Finding 1：混合记忆提供更平衡的性能

  混合记忆在多数任务上持续优于单一结构。

  在迭代检索下，混合记忆在HotPotQA上F1达82.11%，在2WikiMultihopQA上F1达68.15%。

  Chunks和Summaries擅长长上下文任务（阅读理解、对话理解）：NarrativeQA上chunks

  F1=31.63%，summaries F1=32.26%。

  Knowledge triples和Atomic facts擅长关系推理和精确性：2Wiki上triples  F1=62.06%，
  HotPotQA上atomic facts F1=81.29%。

Finding 2：迭代检索是最优检索方法

  迭代检索在大多数数据集上取得最高分。
  Reranking在中等复杂度数据集上表现良好。

  Single-step retrieval在需要最小上下文整合的任务上有竞争力。

Finding 3：Memory-Doc适合广泛上下文任务，Memory-Only适合精确任务

  NarrativeQA等需要广泛上下文的任务，Memory-Doc更好。

  HotPotQA、LoCoMo等需要精确多跳推理和对话理解的任务，Memory-Only更好。

Finding 4：混合记忆在噪声环境下韧性最强

  随着噪声文档增加，所有结构性能下降，但混合记忆始终最高。

  Chunks下降较慢，triples和summaries下降速率相似。

超参数敏感性

  K（检索数量）：适度增加有益，过大引入噪声，性能下降。

  R（重排序数量）：较小R（如10）常优于更大R，说明精选高相关记忆比大量低相关记忆更

  有效。

  T（每轮迭代检索数量）：增加T可提升查询精炼效果，但过大引入噪声。

  N（迭代轮数）：N=2~3通常最佳，继续增加收益递减。

5. 关键洞察

1. 没有单一最优记忆结构：不同任务适合不同结构，混合记忆提供最平衡的通用方案。

2. 记忆结构与检索方法需协同设计：迭代检索与混合记忆组合效果最佳。

3. 答案生成方式取决于任务需求：广泛上下文任务适合Memory-Doc，精确任务适合Memory-
  Only。

4. 噪声鲁棒性是混合记忆的重要优势：在真实噪声环境下，混合记忆更可靠。

5. 检索数量并非越多越好：过多记忆会引入噪声，精选高相关记忆更关键。

6. 与您之前分析论文的关系

论文              核心焦点                         与本文的关系

本文              结构记忆与检索方法的系统性研究              本工作自身

                                             本文研究记忆结构本身，ExpeL关注
ExpeL           从成功+失败经验提取自然语言洞察
                                             经验学习

论文              核心焦点                         与本文的关系

                                             AWM使用工作流作为记忆，本文比
AWM             工作流记忆（流程复用）
                                             较chunks/triples/facts/summaries

                                             MemoryBank使用摘要和遗忘，本
MemoryBank      遗忘曲线+分层记忆
                                             文系统比较摘要与其他结构

                                             MemGPT使用chunks式分页，本文
MemGPT          OS式虚拟上下文管理
                                             验证chunks的适用场景

                                             HippoRAG使用知识三元组，本文验
HippoRAG        KG+PPR多跳检索
                                             证triples在多跳QA的优势

                                             Mem0使用事实提取和图记忆，本文
Mem0            生产级记忆：提取+更新+图记忆
                                             验证atomic facts和triples的效果

                                             SeCom关注对话记忆粒度，本文关
SeCom           片段级记忆+压缩去噪
                                             注通用记忆结构

                                             MemOS统一多种记忆类型，本文提
MemOS           记忆操作系统
                                             供结构选择的实证依据

                                             ReasoningBank存储推理策略，本
ReasoningBank   从成功+失败中提炼推理策略
                                             文研究底层记忆结构

关键区别：本文不是提出新记忆系统，而是首次系统性地比较不同记忆结构和检索方法，为其他

记忆系统设计提供实证指导。它的发现可以指导ExpeL、AWM、MemoryBank、HippoRAG等
系统选择更合适的记忆结构和检索策略。

7. 局限性与未来方向

  任务范围有限：仅覆盖多跳QA、单跳QA、对话理解、阅读理解，未涉及自演化、社会模拟

  等。

  噪声类型单一：仅考虑随机文档噪声，未探索无关/矛盾信息等更复杂噪声。

  超参数范围受限：受计算资源限制，K、R、T、N的范围有限，可能未达最优。

  未来方向：扩展到更多任务和噪声类型；探索记忆结构在自演化和社交模拟中的作用；更深
  入地研究记忆结构与检索方法的协同优化。

8. 总结

本文的核心贡献在于，它首次系统性地研究了LLM智能体中不同记忆结构（chunks、

knowledge triples、atomic facts、summaries、mixed）和检索方法（single-step、
reranking、iterative）对性能的影响。   主要发现包括：混合记忆提供最平衡的性能，

chunks/summaries擅长长上下文任务，triples/atomic  facts擅长关系推理和精确任务，迭代

检索是最优检索方法，混合记忆在噪声环境下韧性最强。这项工作为LLM智能体记忆系统设计
提供了重要的实证依据，与ExpeL、AWM、MemoryBank、MemGPT、HippoRAG、Mem0、

SeCom、MemOS、ReasoningBank等共同构成了LLM记忆研究的完整技术图谱，标志着记忆

研究从“提出新架构”到“系统评估基础设计选择”的深入发展。
```

