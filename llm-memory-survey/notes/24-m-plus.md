# M+: Extending MemoryLLM with Scalable Long-Term Memory

> 来源文件：`m-plus.pdf` · 整理自 DeepSeek 对话阅读笔记 · [返回综述](../survey.md)

好的，我已仔细阅读并分析了您提供的论文《M+: Extending MemoryLLM with Scalable Long-Term Memory》。这是对MemoryLLM的重要扩展，通过引入长期记忆机制和协同训练的检索器，显著提升了知识保留能力。

以下是对该论文核心内容的系统分析与总结：

## 1. 核心问题与动机

论文指出现有潜在空间记忆方法的根本局限：MemoryLLM虽然将1B参数压缩到隐藏状态形成记忆池，但只能有效保留约20k token以内的信息，超出后知识保留能力急剧下降。

现有方法的不足：

Token级记忆（如MemGPT、MemoryBank）：以文本形式存储，可解释性强，但冗余度高、压缩效率低、冲突解决困难。

潜在空间记忆（如MemoryLLM）：压缩高效、可端到端训练、更接近人类记忆机制，但长期保留能力受限。

KV缓存方法（如H2O、SnapKV）：需对每个查询头、每层单独检索，延迟高，且超过30k后难以召回。

核心动机是在保持潜在空间记忆效率优势的同时，大幅扩展知识保留能力，从20k token扩展到160k token以上。

## 2. 提出的方法：M+

M+在MemoryLLM基础上引入长期记忆（Long-Term Memory, LTM）和协同训练的检索器。

核心结构短期记忆 θ：即MemoryLLM的原始记忆池，L层，每层N个token。

长期记忆 Θ：新增的长期记忆池，L层，每层最多M个token（默认150k）。

Transformer解码器 ϕ：Llama-3.1-8B。

更新过程原MemoryLLM中被随机丢弃的K个token，在M+中不再丢弃，而是存入长期记忆 Θ 。

l每个token分配"age"变量，检索后按时间排序。

当长期记忆达到最大容量M时，丢弃age最大的token。

新的K个token以与MemoryLLM相同的方式生成并加入短期记忆。

生成过程

```
  每层从长期记忆中检索K        个token，按age排序，与短期记忆θ       拼接。
                    0                           l
  查询隐藏状态通过交叉注意力访问长期和短期记忆。

协同训练检索器

  检索器包含查询投影器f        和键投影器f    ，均为两层MLP，输出维度d          = d/20。
                    q         k                     proj
  训练目标：最大化查询隐藏状态与相关记忆tokenθ             的相似度，最小化与不相关tokenθ        的相
                                       +                         −
  似度：

                       min−log(p  ) − log(1 − p )
                                 +           −
                       f ,f
                        q k
  检索器与语言模型协同训练，每层只检索一次（而非每个注意力头单独检索），大幅降低延
  迟。

多LoRA设计

  使用两组LoRA权重：一组用于更新过程（压缩/写入），一组用于生成过程（加载/读取）。

  类比T5编码器-解码器不共享权重的设计。

训练数据课程（三阶段）

  Stage 1：在fineweb-edu上持续训练MemoryLLM，1,200,000步，4周。

  Stage 2：在SlimPajama长文档（4k-64k）上训练，增强长上下文建模，1周。

  Stage 3：引入长期记忆，调整配置（短期10,240         + 长期检索2,560  = 12,800），持续训
  练。

关键配置

  K  = 256，N  = 10240，检索K    = 2560，生成窗口2048。
                            0
  短期+长期=12,800   token，总注意力矩阵形状(12800     + 2048) × 2048。

  8×A100 GPU，DeepSpeed Stage-2。

3. 实验与主要结果

LongBook-QA和LongBook-Event-QA

方法                      LongBook-QA F1      LongBook-Event-QA Acc

Llama-3.1-8B-16k        0.151               0.227

Llama-3.1-8B-SnapKV     0.163               0.229

Llama-3.2-3B-128k       0.169               0.190

Llama-3.1-8B-BM25       0.162               0.215

M+                      0.175               0.239

  M+在所有基线中表现最佳，且使用的token最少（12,800记忆             + 2,048生成）。
  M+优于BM25检索，说明块级检索在需要全局理解的任务中不足。

GPU内存成本

方法                                         GPU内存(MB)

Llama-3.1-8B-SnapKV                        32,574

Llama-3.2-3B-128k                          30,422

M+                                         21,177

方法                                         GPU内存(MB)

Llama-3.1-8B-16k                           19,239

M+(offload)                                17,973

MemoryLLM-8B                               21,176

MemoryLLM-8B(offload)                      17,967

  M+的GPU内存成本低于SnapKV和3B-128k，通过CPU         offload进一步降至最低。

知识保留实验（SQuAD/NaturalQA）

  M+显著优于MemoryLLM-7B，知识保留从<20k扩展到>160k          token。

  M+优于Llama-3.1-8B-SnapKV（后者在30k后失效）。

  检索质量：约30%的ground-truth    token被召回，随机检索仅3%。

LongBench（相对短文档）

          方法           2wiki         hotpot        qasper       musique

          MemoryLLM-
                       27.2          34.0          19.6         13.5
          7B(20k)

          Llama-3.1-
                       34.1          44.7          30.1         32.0
          8B(16k)

          M+(16k)      32.7          38.6          30.4         24.6

  M+在4/6数据集上匹配Llama-3.1-8B，仅hotpot和musique略低（因随机丢弃和跨块注意力

  限制）。

消融研究

  长期记忆有效性：Stage     3显著提升保留能力，从50k扩展到160k+。

  检索器有效性：协同训练检索器远优于注意力检索（M+-Attn）。

  模型质量：M+在2,048     token窗口内困惑度1.9828   vs Llama-3.1-8B 1.9734，基本无损。

  延迟分析：M+比MemoryLLM略高（检索开销），128k输入时M+(offload)额外延迟约1秒
  （3%）。

4. 关键洞察

1. 长期记忆机制：将被丢弃的短期记忆token存入长期记忆，实现知识从短期到长期的迁移，扩

  展保留从20k到160k+。

2. 协同训练检索器：与语言模型联合训练，每层只检索一次（32次），远优于KV缓存方法
  （1024次）。

3. CPU offload：长期记忆存储在CPU，按需加载，GPU内存成本降至最低。

4. 三阶段数据课程：从短文档→长文档→长期记忆，逐步增强长上下文建模能力。

5. 多LoRA设计：读写分离的LoRA权重，使压缩和加载各自优化。

6. 压缩效率：512   token块压缩为256个记忆向量，无损压缩；KV方法需下采样导致信息损失。

5. 与您之前分析论文的关系

论文                   核心焦点                      与M+的关系

                     MemoryLLM + 长期记忆  + 协同    本工作自身，是MemoryLLM的
M+
                     检索器                       直接扩展

                                               M+在MemoryLLM基础上增加长
MemoryLLM            潜在空间记忆池自更新
                                               期记忆和检索器

                                               MemGPT用外部存储分页；
MemGPT               OS式虚拟上下文管理
                                               M+用潜在空间长期记忆

                                               MemoryBank用心理学启发；
MemoryBank           遗忘曲线+分层记忆
                                               M+用随机丢弃+协同检索

                                               MemoryOS管理对话记忆；
MemoryOS             OS段页式管理+热度驱逐
                                               M+管理潜在空间记忆

                                               Mem0是外部记忆；M+是模型内
Mem0                 生产级记忆：提取+更新+图记忆
                                               化记忆

                                               A-Mem是外部笔记系统；M+是
A-Mem                Zettelkasten式原子笔记
                                               潜在空间记忆

                                               EverMemOS是认知启发；M+是
EverMemOS            三阶段记忆生命周期
                                               参数级记忆扩展

论文                   核心焦点                      与M+的关系

                                               这些训练模型；M+训练记忆池
MEM1/MemAgent        RL训练模型内化记忆
                                               +检索器

                                               Memory-R1用RL优化操作；
Memory-R1            RL训练记忆管理操作
                                               M+用协同训练检索器

                                               HippoRAG是检索端；M+是潜在
HippoRAG/HippoRAG 2  KG+PPR多跳检索
                                               空间记忆+检索器

                                               Larimar编辑知识；M+扩展长期
Larimar              情景记忆控制器
                                               保留

                                               MIRIX是多组件外部系统；M+是
MIRIX                六种记忆组件+多智能体
                                               单模型内化记忆

关键区别：M+的独特贡献在于将潜在空间记忆从短期扩展到长期，通过协同训练的检索器实现
高效检索。它是MemoryLLM的直接升级，在保持相同GPU内存开销的情况下，将知识保留从

20k扩展到160k+，代表了潜在空间记忆方法的重要突破。

6. 局限性与未来方向

  CPU-GPU通信开销：长期记忆存储在CPU，offload引入额外延迟（128k时约1秒，3%）。

  相对短文档性能略降：因随机丢弃和跨块注意力限制，在8k/16k                LongBench上略低于
  Llama-3.1-8B。

  训练成本高：三阶段训练共约6周，8×A100。

  未扩展到128k：受GPU资源限制，仅训练到12,800+2,048          token，理论可扩展但未验证。

  检索质量约30%：仍有提升空间。

  未来方向：减少CPU-GPU通信开销、扩展到更大规模、更高效检索器、多模态扩展。

总结

M+的核心贡献在于，它通过引入长期记忆机制和协同训练的检索器，将MemoryLLM的知识保

留能力从20k   token扩展到160k+  token，同时保持相同或更低的GPU内存开销。           在

LongBook-QA和LongBook-Event-QA上，M+以最少的token使用量超越所有基线；在

SQuAD/NaturalQA知识保留实验中，M+显著优于MemoryLLM-7B和SnapKV；在LongBench
短文档上，M+匹配Llama-3.1-8B性能。这项工作与MemoryLLM、MemGPT、

MemoryBank、MemoryOS、Mem0、A-Mem、EverMemOS、MEM1、MemAgent、
Memory-R1、HippoRAG、Larimar、MIRIX等共同构成了LLM记忆研究从“外部存储”到“潜在空

间”、从“短期”到“长期”、从“静态”到“动态”的完整技术演进图谱，标志着潜在空间记忆方法进入

“可扩展长期保留”的新阶段。
```

