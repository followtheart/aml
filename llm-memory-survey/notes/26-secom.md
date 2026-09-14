# SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents

> 来源文件：`secom.pdf` · 整理自 DeepSeek 对话阅读笔记 · [返回综述](../survey.md)

好的，我已仔细阅读并分析了您提供的论文《SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents》。这是一篇来自清华大学和微软的研究工作，聚焦于长期对话中记忆单元的粒度选择与去噪，提出了片段级记忆构建和压缩去噪方法。

以下是对该论文核心内容的系统分析与总结：

## 1. 核心问题与动机

论文指出现有长期对话代理在检索增强响应生成（RAG）中，记忆单元的粒度选择存在根本性权衡：

轮次级（Turn-level）：粒度太细，相关信息分散在多个交互轮次中，检索到的上下文碎片化、不完整，容易漏掉关键信息。

会话级（Session-level）：粒度太粗，一个会话可能涵盖多个主题，包含大量无关内容，干扰检索和生成。

摘要级（Summarization-based）：将对话压缩为摘要，虽然减少了长度，但会丢失关键细节，导致回答不准确。

核心动机是找到一种既能保持主题连贯性、又能排除无关内容的记忆粒度，同时去除自然语言中固有的冗余噪声，以提升记忆检索的准确性和生成质量。

## 2. 提出的方法：SeCom

```
SeCom包含两个核心组件：对话分割（Segmentation）          和  基于压缩的去噪（Compression-

based Denoising）。

2.1 片段级记忆构建

  使用对话分割模型      f I 将每个会话  c i 分割为 K i 个主题连贯的片段    {s k }。

                                           C
  每个片段作为一个记忆单元         m，记忆库大小     ∣M∣ = ∑    K 。
                                           i=1 i
  零样本分割：使用GPT-4作为分割模型，通过提示词让模型输出JSONL格式的分割索引。也
  支持Mistral-7B和RoBERTa等轻量模型。

  带反思的分割：当有少量标注数据时，将分割提示视为“前缀”，通过LLM自我反思迭代优化
  分割指导   G，类比prefix-tuning的优化过程。

2.2 基于压缩的记忆去噪

  动机：自然语言本身具有冗余性（Shannon,          1951），这种冗余对检索系统来说是噪声。

  方法：在检索前，使用提示压缩模型           f    （LLMLingua-2）对记忆单元进行去噪，压缩率
                              Comp
  75%。

  检索公式：{m     } ← f (u∗,f    (M),N)
             n     R     Comp
  效果：压缩后，查询与相关片段的相似度增加，与无关片段的相似度降低，检索召回率提
  升。

2.3 响应生成

  检索到的N个片段按时间顺序拼接，作为上下文，查询响应生成模型                    f    生成最终响应。
                                                     LLM

3. 实验与主要结果

数据集：LoCoMo（平均300轮、9K        token）、Long-MT-Bench+（重构自MT-Bench+，平

均65.45轮、19K  token）。

评估指标：BLEU、ROUGE、BERTScore、GPT4Score、pairwise比较、人工评估。

主要结果（LoCoMo）

          方法                GPT4Score     BLEU          Rouge-L

          Zero History      24.86         1.94          13.24

          Full History      54.15         6.26          22.39

          Turn-Level (BM25) 65.58         7.05          24.21

          Session-Level
                            63.16         7.45          24.29
          (BM25)

          SumMem            53.87         2.87          16.25

          RecurSum          56.25         2.22          16.25

          ConditionMem      65.92         3.41          17.54

          MemoChat          65.10         6.76          23.65

          SeCom (BM25,
                            71.57         8.07          26.55
          GPT4-Seg)

          SeCom (MPNet,
                            69.33         7.19          24.38
          GPT4-Seg)

主要结果（Long-MT-Bench+）

          方法                GPT4Score     BLEU          Rouge-L

          Full History      63.85         7.51          20.76

          Turn-Level (MPNet) 84.91        12.09         27.82

          Session-Level
                            73.38         8.89          22.79
          (MPNet)

          MemoChat          85.14         12.66         26.87

          SeCom (MPNet,
                            88.81         13.80         27.64
          GPT4-Seg)

消融研究

  记忆粒度：片段级在所有上下文预算下均优于轮次级和会话级。

  压缩去噪：去除后GPT4Score下降最多9.46点，证明去噪机制至关重要。

  分割模型：GPT-4最佳，Mistral-7B次之，RoBERTa仍具竞争力。

对话分割评估

  在DialSeg711、SuperDialSeg、TIAGE上，零样本分割F1和Score均优于基线。

  迁移学习设置下，仅用100个样本反思学习的分割指导即可超越在完整源训练集上训练的基
  线。

Mistral-7B生成器

  即使Mistral-7B拥有32K上下文窗口，能容纳全部历史，Full         History仍不如SeCom，证明
  SeCom的记忆构建和检索机制更有效。

成本分析

  SeCom输入token  1,722，输出135，延迟2.61s；MemoChat输入7,233，输出229，延迟

  5.60s。SeCom更高效且效果更好。

4. 关键洞察

1. 片段级记忆是最佳粒度：平衡了轮次级的碎片化和会话级的噪声，提供主题连贯且无冗余的
  检索单元。

2. 压缩作为去噪机制：LLMLingua-2压缩去除了自然语言冗余，提高了检索准确率，且无需微
  调检索器，即插即用。

3. 对话分割可轻量化：GPT-4、Mistral-7B甚至RoBERTa均可用于分割，反思机制进一步提升

  分割质量。

4. 摘要方法的信息损失问题：SumMem和RecurSum表现不佳，因为摘要丢失了回答所需的细
  节。

5. 鲁棒性：SeCom在不同检索器（BM25/MPNet）、不同分割模型、不同生成器（GPT-

  3.5/Mistral-7B）下均保持优势。

5. 与您之前分析论文的关系

论文                   核心焦点                      与SeCom的关系

SeCom                片段级记忆   + 压缩去噪            本工作自身

                                               SeCom在LoCoMo上评估并显著
LoCoMo               长期对话记忆评测基准
                                               超越基线

                                               MemoryBank用摘要和遗忘机
MemoryBank           遗忘曲线+分层记忆
                                               制；SeCom用片段级和压缩去噪

                                               MemGPT用分页管理；SeCom
MemGPT               OS式虚拟上下文管理
                                               用片段分割和压缩

                                               A-Mem是笔记网络；SeCom是
A-Mem                Zettelkasten式原子笔记
                                               片段级记忆

                                               EverMemOS是认知启发；
EverMemOS            三阶段记忆生命周期
                                               SeCom是分割+压缩

                                               Mem0是外部记忆管理；SeCom
Mem0                 生产级记忆：提取+更新+图记忆
                                               关注记忆粒度和去噪

                                               HippoRAG是检索端图方法；
HippoRAG/HippoRAG 2  KG+PPR多跳检索
                                               SeCom是对话分割+压缩

                                               SeCom可视为记忆构建方法，可
LongMemEval          长期记忆评测基准
                                               在LongMemEval上评估

论文                   核心焦点                      与SeCom的关系

                                               ReasoningBank是经验学习；
ReasoningBank        从成功+失败中提炼推理策略
                                               SeCom是对话记忆构建

                                               MemOS统一三种记忆类型；
MemOS                记忆操作系统
                                               SeCom专注对话片段记忆

关键区别：SeCom的独特贡献在于系统性地研究了记忆粒度对检索增强生成的影响，并提出了

片段级记忆+压缩去噪的解决方案。它不涉及复杂的记忆管理架构，而是聚焦于“记忆单元应该

多大”以及“如何去除记忆中的噪声”这两个基础问题。这与MemoryBank、MemGPT、A-Mem
等关注记忆管理机制的工作形成互补。

6. 局限性与未来方向

  分割依赖LLM：虽然支持轻量模型，但GPT-4效果最佳，成本较高。

  压缩率需调优：75%压缩率在实验中表现最佳，但可能因数据集而异。

  未探索更复杂的记忆结构：如分层、图结构等，片段级是当前最优但可能不是终点。

  评估范围：主要在对话QA上验证，其他任务（如多模态、工具使用）待扩展。

  未来方向：更高效的分割模型、自适应压缩率、与其他记忆架构（如A-Mem、MemOS）的

  结合、多模态对话记忆。

总结

SeCom的核心贡献在于，它系统性地揭示了记忆粒度对长期对话代理性能的关键影响，并提出
了片段级记忆构建和压缩去噪的解决方案。             通过对话分割将长会话分为主题连贯的片段，再通

过LLMLingua-2压缩去除冗余噪声，SeCom在LoCoMo和Long-MT-Bench+上显著超越轮次

级、会话级和摘要级基线，GPT4Score分别达到71.57和88.81。分割模型在DialSeg711、
SuperDialSeg、TIAGE上也表现优异。这项工作与LoCoMo、MemoryBank、MemGPT、A-

Mem、EverMemOS、Mem0、HippoRAG、LongMemEval、ReasoningBank、MemOS等共
同构成了LLM记忆与长期对话研究的完整技术图谱，标志着对话记忆研究从“如何管理记忆”到

“记忆应该以什么粒度存在”的深入探索。
```

