# 精读笔记：Voyager — An Open-Ended Embodied Agent with Large Language Models

- 出处：TMLR 2024（arXiv:2305.16291，NVIDIA 等）；论文：`papers/voyager.pdf`
- 一句话：Minecraft 终身学习 agent 三件套——自动课程、**可执行代码技能库**（存取复杂行为）、带环境反馈/执行错误/自验证的迭代提示。

## 对 AML 方案的可复用点

- **技能库 = 过程性记忆的经典形态**：技能以代码+自然语言描述双重存储，检索时按描述匹配、执行时取代码。Coding Track 的 fix_pattern 可采用此"描述索引 + 内容执行"双层结构。
- 自验证迭代：新技能入库前经执行验证——AML 场景无法执行验证，但 Coding Track 经验入库前可用"测试是否通过"信号做可信度标记（AMU confidence 字段）。
- 组合性：技能可组合引用——图谱中 workflow 节点间可建"composed_of"边。

## 备注

- 具身场景与 AML 文本/代码记忆形态差异较大，主要借鉴其"经验资产化"思想，不直接复用架构。
