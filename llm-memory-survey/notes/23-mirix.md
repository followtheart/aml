# MIRIX: Multi-Agent Memory System for LLM-Based Agents

> 来源文件：`mirix.pdf` · 整理自 DeepSeek 对话阅读笔记 · [返回综述](../survey.md)

好的，我已仔细阅读并分析了您提供的论文《MIRIX: Multi-Agent Memory System for LLM-Based Agents》。这是一篇来自MIRIX AI的研究工作，提出了一个模块化、多智能体的记忆系统，通过六种专门化记忆组件和八个智能体的协同，使LLM智能体能够持久、准确地记忆和检索多模态信息。

以下是对该论文核心内容的系统分析与总结：

## 1. 核心问题与动机

论文指出现有LLM智能体记忆系统存在根本性局限：

缺乏组合式记忆结构：多数系统将所有历史数据存储在单一扁平存储中，未按专门化记忆类型（如程序性、情景性、语义性）进行路由，导致检索效率低、准确性差。

多模态支持差：以文本为中心的记忆机制在输入主要为非语言形式（如图像、界面布局、地图）时失效。

可扩展性与抽象不足：存储原始输入（尤其是图像）导致内存需求极高，缺乏有效的抽象层来总结和保留关键信息。

核心动机是设计一个模块化、多智能体的记忆系统，具备有效的路由和检索能力，支持多模态、可扩展、可抽象，使LLM智能体真正“记住”用户特定信息。

## 2. 提出的方法：MIRIX

六种记忆组件

```
组件                 功能                         关键字段

                   存储高优先级持久信息（角色、用
Core Memory                                   persona、human
                   户事实）

                                              event_type、summary、
Episodic Memory    存储时间戳事件和交互
                                              details、actor、timestamp

                                              name、summary、details、
Semantic Memory    存储抽象知识和事实
                                              source

Procedural Memory  存储目标导向的流程和脚本               entry_type、description、steps

                                              title、summary、
Resource Memory    存储文档、文件、多媒体
                                              resource_type、content

                   存储敏感逐字信息（凭证、地址、            entry_type、source、
Knowledge Vault
                   API密钥）                     sensitivity、secret_value

多智能体架构

  Meta Memory Manager：负责任务路由，分析输入并决定哪些记忆组件相关。

  六个Memory   Managers：各自维护一种记忆类型，并行更新。

  Chat Agent：与用户交互，执行主动检索并生成响应。

主动检索机制（Active    Retrieval）

  两阶段：先生成当前主题，再用主题从六个组件中各检索Top-10条目，注入系统提示。

  检索结果带标签（如       <episodic_memory>...</episodic_memory> ），使模型明确内容来源。

  支持多种检索函数：       embedding_match 、 bm25_match 、 string_match 。

工作流程

  记忆更新：新输入→自动搜索→Meta          Memory Manager路由→Memory   Managers并行更
  新→确认。

  对话检索：用户查询→Chat       Agent自动搜索（粗检索）→分析查询→选择组件和检索方法

  （精检索）→综合响应。

3. 实验与主要结果

数据集与设置

  ScreenshotVQA：新构建的多模态基准，包含3名PhD学生共约29,000张高分辨率屏幕截

  图，11/21/55个问题。评估：LLM-as-Judge（GPT-4.1）。

  LOCOMO：长对话基准，10段对话，每段约600轮、26k            token，约200个问题。评估：

  LLM-as-Judge。

ScreenshotVQA结果

方法                          准确率                   存储

Gemini（长上下文）                0.1166                236.70 MB

SigLIP（RAG）                 0.4410                15.07 GB

MIRIX                       0.5950                15.89 MB

  MIRIX比RAG准确率提高35%，存储减少99.9%。

  比长上下文准确率提高410%，存储减少93.3%。

LOCOMO结果

          方法           单跳            多跳            开放域          时间

          LangMem      74.47         61.06         67.71        86.92

          Zep          79.43         69.16         73.96        83.33

          Mem0         62.41         57.32         44.79        66.47

          MIRIX        85.11         83.70         65.62        88.39

          Full-Context
                       88.53         77.70         71.88        92.70
          （上界）

  MIRIX总体85.38%，超过最佳基线LangMem       8个百分点以上，接近全上下文上界。

  多跳任务提升最大（+24点），因MIRIX显式存储整合后的事件，无需查询时拼接信息。

  单跳和时间任务也显著领先。

应用

  跨平台应用（React-Electron  + Uvicorn），每1.5秒截屏，相似图像跳过，20张触发更新，

  流式上传降低延迟至<5秒。

  可穿戴设备集成：支持本地/云端混合记忆管理。

  记忆市场愿景：去中心化记忆交换、隐私保护、代币化交易。

4. 关键洞察

1. 六种专门化记忆组件：超越传统的短/长期二分，覆盖情景、语义、程序、资源、核心、敏感

  信息，实现更精细的路由和检索。

2. 多智能体协调：Meta    Memory Manager + 六个Memory Managers，实现并行更新和动态
  路由，避免单智能体瓶颈。

3. 主动检索：自动生成主题并检索，无需用户显式触发，解决模型默认使用过时参数知识的问

  题。

4. 多模态支持：通过抽象层（SQLite存储提取信息而非原始图像），实现99.9%存储压缩，同
  时保持高准确率。

5. 记忆市场：提出个人记忆作为数字资产的新范式，支持隐私保护、去中心化存储和交易。

5. 与您之前分析论文的关系

论文                    核心焦点                     与MIRIX的关系

                      六种记忆组件   + 多智能体路由   +    本工作自身，记忆系统研究的集
MIRIX
                      多模态                      大成者之一

                                               MemOS统一明文/激活/参数；
                      记忆操作系统：三种记忆类型统
MemOS                                          MIRIX六种组件，更细粒度，侧
                      一调度
                                               重多模态

                                               MemoryOS侧重对话记忆分层；
MemoryOS              OS段页式管理   + 热度驱逐
                                               MIRIX支持多模态，六种组件

                                               MemGPT的Core  Memory被
MemGPT                OS式虚拟上下文管理               MIRIX继承；MIRIX扩展为六种

                                               组件+多智能体

                                               Mem0是MIRIX的基线之一；
Mem0                  生产级记忆：提取+更新+图记忆
                                               MIRIX在LOCOMO上远超Mem0

                                               A-Mem是MIRIX基线之一；
A-Mem                 Zettelkasten式原子笔记
                                               MIRIX多智能体路由更系统

                                               EverMemOS是认知启发；
EverMemOS             三阶段记忆生命周期
                                               MIRIX是工程化多智能体系统

                                               MemoryLLM更新参数；MIRIX
MemoryLLM             潜在空间记忆池自更新
                                               更新外部多组件存储

                                               这些训练模型；MIRIX是外部系
MEM1/MemAgent         RL训练模型内化记忆
                                               统+多智能体协调

                                               Memory-R1用RL优化操作；
Memory-R1             RL训练记忆管理操作
                                               MIRIX用多智能体路由+主动检索

                                               HippoRAG是检索端；MIRIX的
HippoRAG/HippoRAG 2   KG+PPR多跳检索
                                               主动检索+多组件路由更全面

                                               Zep是MIRIX基线之一；MIRIX在
Zep                   时序知识图谱
                                               多模态和组件化上更优

论文                    核心焦点                     与MIRIX的关系

                                               MIRIX在LOCOMO上SOTA；
LOCOMO/LongMemEval    长期记忆评测基准
                                               ScreenshotVQA为新多模态基准

关键区别：MIRIX的独特贡献在于六种专门化记忆组件的组合                + 多智能体路由    + 主动检索   + 多

模态支持。它不仅是MemGPT的扩展，更是对Mem0、A-Mem、MemoryOS、EverMemOS

等工作的系统性整合与超越，尤其在多模态和存储效率上实现突破。

6. 局限性与未来方向

  依赖函数调用能力：需要强函数调用能力的LLM（如GPT-4.1-mini），弱模型效果可能下
  降。

  路由准确性：Meta     Memory Manager的路由决策可能出错，导致记忆更新或检索不准确。

  开放域推理：在需要全局理解的开放域问题上，MIRIX仍依赖RAG检索，性能略低于全上下

  文。

  评估范围：ScreenshotVQA样本量有限（3名用户），多模态场景覆盖不够广泛。

  隐私与安全：记忆市场涉及敏感信息交易，需强隐私保护机制。

  未来方向：更挑战性的真实世界基准、更鲁棒的路由机制、更广泛的多模态支持、记忆市场
  的落地。

总结

MIRIX的核心贡献在于，它提出了一个模块化、多智能体的记忆系统，通过六种专门化记忆组件

（Core、Episodic、Semantic、Procedural、Resource、Knowledge Vault）和八个智能体
的协同，实现了对多模态、长期、个性化信息的有效记忆、路由和检索。                      在ScreenshotVQA

上，MIRIX比RAG准确率提高35%的同时存储减少99.9%；在LOCOMO上达到SOTA

85.38%，接近全上下文上界。这项工作与MemOS、MemoryOS、MemGPT、Mem0、A-
Mem、EverMemOS、MemoryLLM、MEM1、MemAgent、Memory-R1、HippoRAG等共同

构成了LLM记忆研究从“外部存储”到“系统架构”、从“单一类型”到“多组件协同”、从“文本”到“多
模态”的完整技术演进图谱，标志着LLM记忆研究进入“多智能体系统化治理”的新阶段。

本回答由   AI 生成，内容仅供参考，请仔细甄别
```

