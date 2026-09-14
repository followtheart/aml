# MemOS: A Memory OS for AI System

> 来源文件：`memos.pdf` · 整理自 DeepSeek 对话阅读笔记 · [返回综述](../survey.md)

好的，我已仔细阅读并分析了您提供的论文《MemOS: A Memory OS for AI System》。这是一篇来自MemTensor、上海算法创新研究院等机构的研究工作，提出了一个面向AI系统的记忆操作系统，是“记忆作为系统资源”这一理念的集大成之作。

以下是对该论文核心内容的系统分析与总结：

## 1. 核心问题与动机

论文指出现有LLM面临根本性瓶颈：缺乏定义良好的记忆管理系统，阻碍了长上下文推理、持续个性化和知识一致性。

现有记忆机制可归纳为三类，但各自存在局限：

参数记忆：知识编码在模型权重中，更新成本高、可解释性差、易灾难性遗忘。

RAG：外部检索增强，但本质是“即时检索+临时组合”的无状态补丁，缺乏生命周期跟踪、版本管理和权限感知调度。

工具化管理（如MemGPT、Mem0）：引入CRUD操作，但停留在接口级实用工具，缺乏对记忆作为核心资源的系统性建模和治理。

核心动机是将记忆视为可调度、可演化的一等系统资源，借鉴操作系统设计原理，构建统一的记忆操作系统。

## 2. 提出的方法：MemOS

核心思想记忆是系统级资源，需要统一表示、调度和演化。

三种记忆类型：明文记忆（外部可编辑知识）、激活记忆（KV缓存、隐藏状态）、参数记忆（模型权重）。

统一封装单元：MemCube，包含记忆载荷和元数据（描述标识、治理属性、行为使用指标）。

MemCube的元数据描述标识：时间戳、来源签名、语义类型。

治理属性：访问控制、生命周期策略、优先级、合规与可追溯性。

行为使用指标：访问频率、最近性、上下文指纹、版本链。

三种记忆类型的转换路径明文 → 激活：高频明文转为激活向量/注意力模板。

明文/激活 → 参数：稳定知识蒸馏为参数模块。

参数 → 明文：冷参数卸载到外部存储。

三层架构

```
层级              核心模块                        职责

                MemReader、Memory API、       解析自然语言为结构化
接口层
                Memory Pipeline             MemoryCall，提供统一API

                MemOperator、MemScheduler、
操作层                                         组织、调度、生命周期管理
                MemLifecycle

                MemGovernance、MemVault、
基础设施层                                       存储、治理、迁移、共享
                MemLoader/Dumper、MemStore

核心模块详解

  MemReader：解析用户输入，提取任务意图、时间范围、主题实体、上下文锚点，输出结构

  化MemoryCall。

  MemOperator：多视角记忆结构化（标签系统、知识图谱、语义分层），支持混合检索（符
  号+语义）和任务对齐路由。

  MemScheduler：类型感知转换与加载，支持跨类型迁移（明文↔激活↔参数），基于任务

  语义、调用频率、内容稳定性动态调度。

  MemLifecycle：五状态有限状态机（Generated      → Activated → Merged → Archived →
  Expired），支持“时间机器”快照回滚和“冻结”机制。

  MemGovernance：三元权限模型（用户身份、记忆对象、调用上下文），生命周期策略，

  隐私控制，审计接口。

  MemVault：中央存储和路由，支持向量库、关系数据库、Blob存储的统一适配。

  MemStore：开放访问接口，支持记忆的发布、订阅和分发。

3. 实验与主要结果

基准：LoCoMo、LongMemEval、PreFEval、PersonaMem

主要结果

在LoCoMo上（GPT-4o-mini）：

  MemOS-1031：Overall LLM Judge = 75.80，F1 = 45.27

  相比Memobase（72.01/50.18）、Mem0（64.57/43.46）显著提升

  在单跳（81.09）、多跳（67.49）、时间推理（75.18）、开放域（55.90）均领先

在LongMemEval上：

  MemOS-1031：Overall = 77.8%，远超Memobase（72.4%）、Mem0（66.4%）

  单会话偏好96.7%，单会话助手67.9%，时间推理77.4%

在PreFEval上：

  0轮和10轮场景下，个性化响应率分别达77.2%           和71.9%，偏好未察觉错误率最低
  （4.6%/7.4%）

在PersonaMem上：

  精确度61.2%，超过Memobase（58.9%）、Zep（57.8%）

检索鲁棒性

  100 QPS下，MemOS保持100%成功率，P99延迟463ms，远优于MemU（7.5%成功率）
  和Zep（26.6%成功率）。

KV记忆加速

  KV注入相比提示注入，TTFT降低高达91.4%（Qwen2.5-72B，长上下文短查询）。

消融研究

  各组件均贡献显著，MTM影响最大，LPM次之，Chain影响最小。

  随chunk size和Top-K增加，性能稳步提升，多跳和时间推理任务受益最大。

4. 关键洞察

1. 记忆即系统资源：将记忆从隐式依赖抽象为一等、可调度、可管理的资源，打破“记忆孤

  岛”。

2. 三种记忆类型统一调度：明文、激活、参数记忆的转换路径，实现知识从感知到巩固的完整

  演化轨迹。
3. MemCube作为核心抽象：标准化封装+元数据治理，使记忆可组合、可迁移、可融合。

4. OS式治理机制：访问控制、版本管理、审计追踪、生命周期管理，确保记忆安全可信。

5. Mem-training范式：从预训练、后训练走向记忆训练，通过显式可控的记忆单元驱动持续演

  化。

6. 记忆市场愿景：支持付费记忆模块、跨平台迁移、去中心化记忆交换。

5. 与您之前分析论文的关系

论文                     核心焦点                     与MemOS的关系

                       记忆操作系统：统一调度三种            本工作自身，是记忆系统研究
MemOS
                       记忆类型                     的集大成者

                                                MemGPT是MemOS的重要思

MemGPT                 OS式虚拟上下文管理               想来源；MemOS扩展为完整
                                                OS架构

论文                     核心焦点                     与MemOS的关系

                                                MemoryOS侧重对话记忆分
MemoryOS               OS段页式管理+热度驱逐
                                                层；MemOS统一三种记忆类型

                       生产级记忆：提取+更新+图记           Mem0是MemOS的基线之一；
Mem0
                       忆                        MemOS更系统化

                                                MemoryBank是MemOS的早
MemoryBank             遗忘曲线+分层记忆                期启发；MemOS引入完整生命

                                                周期

                                                A-Mem是MemOS的基线之
A-Mem                  Zettelkasten式原子笔记
                                                一；MemOS支持跨类型转换

                                                EverMemOS是认知启发；
EverMemOS              三阶段记忆生命周期
                                                MemOS是OS启发，更完整

                                                MemoryLLM更新参数；
MemoryLLM              潜在空间记忆池自更新
                                                MemOS统一参数/激活/明文

                                                这些训练模型；MemOS是外部
MEM1/MemAgent          RL训练模型内化记忆
                                                系统+统一调度

                                                Memory-R1用RL优化操作；
Memory-R1              RL训练记忆管理操作
                                                MemOS提供完整OS架构

                                                HippoRAG是检索端；MemOS
HippoRAG/HippoRAG 2    KG+PPR多跳检索
                                                的MemOperator支持混合检索

                                                Larimar编辑知识；MemOS管
Larimar                情景记忆控制器
                                                理所有记忆类型

Dynamic                                         这些更新提示/外部记忆；
                       测试时学习/经验学习
Cheatsheet/ExpeL                                MemOS统一调度

LoCoMo/LongMemEval     长期记忆评测基准                 MemOS在这些基准上验证

关键区别：MemOS是最完整的记忆系统架构，首次将参数记忆、激活记忆、明文记忆统一在一

个OS式框架下。它不仅是MemGPT的扩展，更是对Mem0、A-Mem、MemoryOS、

EverMemOS等工作的系统性整合与超越。

6. 局限性与未来方向

  跨LLM记忆共享：需扩展记忆交换协议（MIP），定义标准格式、兼容规则和信任机制。

  自演化MemBlock：开发能根据使用反馈自优化、重构、演化的记忆单元。

  可扩展记忆市场：建立去中心化机制，支持资产级交易、协作更新、分布式演化。

  评估范围：主要在对话和个性化任务验证，多模态、具身智能等场景待扩展。

  系统复杂度：完整OS架构带来工程复杂性，需平衡功能与效率。

总结

MemOS的核心贡献在于，它首次系统性地将记忆视为AI系统的核心资源，构建了一个完整的记

忆操作系统。     通过三种记忆类型（明文、激活、参数）的统一调度、MemCube标准化封装、
三层架构（接口/操作/基础设施）、以及OS式治理机制（访问控制、生命周期、审计追踪），

MemOS在LoCoMo、LongMemEval、PreFEval、PersonaMem四大基准上全面领先，100

QPS下保持100%成功率，KV记忆加速降低TTFT达91.4%。这项工作与MemGPT、
MemoryOS、Mem0、A-Mem、EverMemOS、MemoryLLM、MEM1、MemAgent、

Memory-R1、HippoRAG等共同构成了LLM记忆研究从“外部存储”到“系统架构”、从“启发式管
理”到“RL优化”、从“单一类型”到“统一调度”的完整技术演进图谱，标志着LLM记忆研究进入“系

统化治理”的新阶段。
```

