# 精读笔记：MemGPT — Towards LLMs as Operating Systems

- 出处：ICLR 2024（arXiv:2310.08560，UC Berkeley）；产品化为 Letta；论文：`papers/memgpt.pdf`
- 一句话：OS 式虚拟上下文管理——主上下文（内存）与外部归档存储（磁盘）之间由 LLM 自主通过 function call 分页换入换出，自我编辑记忆。

## 对 AML 方案的可复用点

- **分层思想**：工作记忆（热）/ 归档记忆（冷）——方案的 AMU 全量索引 + episode 原始库即两层；AML 场景 LLM 自主分页不适用（平台直接调 Search），但"热度分层"可用于重排加分（近期/高频记忆加权）。
- **自我编辑记忆**：函数调用式 ADD/UPDATE 操作的鼻祖——Mem0/Memory-R1 的四操作集均源于此。
- LLM-as-a-Judge 评分提示词最初由 MemGPT 团队发布（Mem0 附录 A 即改编于此）——本地评测 judge 模板的源头。

## 备注

- MemGPT 的 agent 自主决定何时读写记忆，与 AML"平台驱动 Add/Search"形态不同；借鉴其分层与操作语义即可。
- Letta 后续提出 sleep-time compute（空闲时后台重组记忆）——AML 的 Add 同步屏障不允许无限后台处理，但可在 Add 完成后做轻量合并。
