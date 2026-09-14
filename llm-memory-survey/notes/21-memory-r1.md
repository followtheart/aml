# Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning

> 来源文件：`memory-r1.pdf` · 整理自 DeepSeek 对话阅读笔记 · [返回综述](../survey.md)

好的，我已仔细阅读并分析了您提供的论文《Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning》。这是一篇来自慕尼黑大学、慕尼黑机器学习中心等机构的研究工作，首次系统性地将强化学习（RL）引入LLM智能体的外部记忆管理。

以下是对该论文核心内容的系统分析与总结：

## 1. 核心问题与动机

论文指出现有记忆增强型LLM存在根本性缺陷：记忆管理是静态的、启发式驱动的，缺乏基于正确性信号的学习机制。

检索挑战：启发式检索可能返回太少条目（遗漏关键上下文）或太多条目（淹没模型、引入噪声），导致模型在无关内容中分心。

管理挑战：决定“记住什么、更新什么、丢弃什么”至关重要。现有系统（如MemGPT、 Mem0）依赖LLM根据上下文指令选择操作，但没有与正确性挂钩的学习信号。

简单案例失败：用户先说“我收养了一只叫Buddy的狗”，后说“我又收养了一只叫Scout的狗”。原始系统误判为矛盾，发出DELETE+ADD，碎片化记忆；而训练后的智能体应发出UPDATE，整合为“Andrew收养了两只狗，Buddy和Scout”。

核心动机是用强化学习让LLM智能体学会自适应地管理记忆和利用检索到的记忆进行推理。

## 2. 提出的方法：Memory-R1

Memory-R1是一个RL微调的记忆增强LLM框架，包含两个专门智能体：

### 2.1 记忆管理器（Memory Manager）

维护记忆库，为每条新信息选择操作：ADD、UPDATE、DELETE、NOOP。

```
  任务形式化：策略      π 以提取信息    x 和检索到的旧记忆     M    为输入，输出操作      o 和更新内
                 θ                           old
  容
    m′
       。
  PPO训练：采样操作并应用于记忆库，将结果传给冻结的Answer                Agent，答案正确性提供标
  量奖励   r。
  GRPO训练：对每个状态采样        G 个候选动作，计算组相对优势         A
                                                 i
                                                   = r i −mean(r) ，无需显式
                                                       std(r)
  价值函数。

  奖励设计：结果驱动的奖励——操作后更新的记忆库传给冻结的Answer                   Agent，奖励基于答
  案正确性（精确匹配EM）。

2.2 答案智能体（Answer    Agent）

  从记忆库检索60个候选记忆，执行记忆蒸馏策略，选择最相关条目后生成答案。

  策略  π 将问题   q 和检索集  M    映射到答案    y。
       θ                 ret
  同样使用PPO/GRPO微调，奖励为答案与真实答案的EM分数。

2.3 两阶段训练

  先训练Memory   Manager（Answer Agent冻结提供奖励），再训练Answer       Agent

  （Memory  Manager固定）。

  这种解耦避免了归因模糊，同时允许两个组件在交替训练中共同适应。

3. 实验与主要结果

数据集：LoCoMo（主训练集，152训练/81验证/1307测试）、MSC、LongMemEval（零样本

迁移）。

主要结果（LoCoMo）

在LLaMA-3.1-8B上：

  Memory-R1-GRPO：F1=45.02，B1=37.51，J=62.74

  相比最强基线MemoryOS：F1提升28.5%，B1提升34.0%，J提升30.2%

  Memory-R1-PPO：F1=41.05，B1=32.91，J=57.54

在Qwen-2.5-7B上：

  Memory-R1-GRPO：F1=41.72，B1=33.70，J=59.53

  相比MemoryOS：F1提升24.5%，B1提升24.1%，J提升20.0%

泛化与扩展性

  仅在LoCoMo上训练，零样本迁移到MSC和LongMemEval上仍持续提升。

  在Qwen-2.5的3B/7B/14B规模上均优于基座模型，PPO和GRPO持续有效。

消融研究

  移除RL微调的Memory     Manager：PPO下F1从41.0降至34.5，GRPO下从45.0降至37.5，证
  明结果驱动RL比脚本控制更有效。

  移除RL微调的Answer    Agent：PPO下F1仅32.5，GRPO下33.0，远低于完整管线的

  41.0/45.0。

  移除记忆蒸馏：GRPO下F1从45.0降至41.0，J从62.7降至60.1，证明过滤无关记忆减少噪
  声。

  Answer Agent与更强Memory   Manager配对：GPT-4o-mini管理时增益更大（F1      +19.72

  vs +10.10），显示记忆质量与答案质量的正向复合效应。

RL策略比较

  GRPO初期收敛更快（组归一化提供更强早期引导），但最终PPO和GRPO达到相近奖励水

  平。

奖励设计分析

  使用J作为奖励：J分数更高（63.58）但F1和B1较低（33.69/23.36），因为鼓励冗长描述性
  答案。

  使用EM作为奖励：平衡提升（41.05/32.91/57.54），因此默认采用EM。

延迟分析

  Memory-R1不仅精度更高，中位和尾部延迟也更低（GRPO尤其明显），实现帕累托改进而

  非权衡。

4. 关键洞察

1. RL是自适应记忆的关键：通过结果驱动的奖励，模型能学习何时

  ADD/UPDATE/DELETE/NOOP，比手工启发式更智能。

2. 极低监督成本：仅152个QA对即可实现SOTA，说明RL信号效率极高。

3. 记忆蒸馏有效降噪：Answer     Agent学习过滤无关记忆，避免“迷失在中间”。
4. 组件复合效应：更强的Memory       Manager让Answer Agent的增益更大，系统整体受益。

5. 泛化性强：仅在LoCoMo训练，零样本迁移到MSC和LongMemEval仍有效。

6. 效率提升：RL训练不仅提高精度，还降低推理延迟，GRPO尤其高效。

5. 与您之前分析论文的关系

论文                   核心焦点                      与Memory-R1的关系

                                               本工作自身，首次将RL引入记忆
Memory-R1            RL训练记忆管理与利用
                                               管理

                                               Mem0用启发式LLM选择操作；
Mem0                 生产级记忆：提取+更新+图记忆
                                               Memory-R1用RL学习操作策略

                                               MemoryOS用热度公式手工设
MemoryOS             OS段页式管理   + 热度驱逐
                                               计；Memory-R1用RL优化

                                               MemoryBank用心理学启发式；
MemoryBank           遗忘曲线   + 分层记忆
                                               Memory-R1用结果驱动学习

                                               MemGPT用函数调用+启发式控
MemGPT               OS式虚拟上下文管理                制流；Memory-R1用RL优化操作

                                               选择

                                               A-Mem用图链接生成；
A-Mem                Zettelkasten式原子笔记链接
                                               Memory-R1用RL决定更新策略

                                               EverMemOS用认知启发；
EverMemOS            三阶段记忆生命周期
                                               Memory-R1用RL端到端优化

                                               这些训练模型内部记忆；
MEM1/MemAgent        RL训练模型内化记忆                Memory-R1训练外部记忆管理操
                                               作

                                               MemoryLLM更新模型参数；
MemoryLLM            潜在空间记忆池自更新
                                               Memory-R1更新外部记忆库

                                               HippoRAG是检索端图方法；
HippoRAG/HippoRAG 2  KG+PPR多跳检索
                                               Memory-R1是记忆管理端RL方法

关键区别：Memory-R1的独特之处在于将记忆操作选择（ADD/UPDATE/DELETE/NOOP）和
记忆利用（蒸馏）都建模为RL问题，用结果驱动的奖励端到端优化。这与Mem0、MemoryOS

等使用启发式或手工设计规则的系统形成鲜明对比，也与MEM1/MemAgent等训练模型内部记

忆能力的工作不同——Memory-R1训练的是对外部记忆库的管理策略。

6. 局限性与未来方向

  评估聚焦对话数据集：LoCoMo、MSC、LongMemEval均为对话中心，扩展到多模态可能

  引入新挑战。

  两阶段分离训练：Memory      Manager和Answer Agent分别训练以确保稀疏奖励下的稳定
  性，但过程较复杂。端到端多智能体RL可简化训练并实现更丰富协调。

  依赖RAG检索质量：初始检索60个候选记忆，若检索失败则蒸馏也无能为力。

  奖励设计权衡：EM奖励平衡但可能鼓励简短答案；J奖励语义正确但与词汇重叠指标不匹

  配。

  未来方向：端到端多智能体RL、多模态记忆、更丰富的记忆操作（如MERGE、LINK）、更

  大规模验证。

总结

Memory-R1的核心贡献在于，它首次系统性地将强化学习引入LLM智能体的外部记忆管理。                       通
过两个RL微调的专门智能体——Memory          Manager（学习ADD/UPDATE/DELETE/NOOP操

作）和Answer   Agent（学习记忆蒸馏与推理）——Memory-R1仅用152个训练样本就在
LoCoMo上实现SOTA，相比Mem0相对提升28%          F1、34% BLEU-1、30%  LLM-as-Judge，

并零样本泛化到MSC和LongMemEval。这项工作与Mem0、MemoryOS、MemoryBank、A-

Mem、EverMemOS、MemGPT、MEM1、MemAgent等共同构成了LLM记忆研究从“启发式
管理”到“学习型管理”、从“外部系统”到“RL优化”、从“单智能体”到“多智能体协同”的完整技术

演进图谱。
```

