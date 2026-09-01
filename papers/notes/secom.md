# 精读笔记：SeCom — On Memory Construction and Retrieval for Personalized Conversational Agents

- 出处：ICLR 2025（arXiv:2502.05589，清华+微软）；论文：`papers/secom.pdf`
- 一句话：记忆单元粒度至关重要——提出会话分割（topic-coherent segment）+ LLMLingua-2 压缩去噪的段级记忆库，LOCOMO 与 Long-MT-Bench+ 上显著优于轮级/会话级/摘要级基线。

## 核心发现（直接影响 AML 方案）

1. **粒度实验结论**：轮级太碎（语义不完整）、会话级太粗（检索噪声大）、摘要级丢细节——**段级（topical segment）最优**。方案 Add 管线的"情景记忆切片（Episode Segmentation）"由此获得实证支持。
2. **压缩去噪**：LLMLingua-2 式 prompt 压缩可作为通用去噪器，提升各粒度下的检索准确率。方案可在 episode 入库时做轻量压缩（保留事实密度），但注意平台对 AMU content 要求"证据原句"——压缩只用于检索键，返回的 content 仍用原句。
3. **个性化场景**：个性化对话记忆（画像维度）的主要参考之一。

## 可复用模块

| SeCom 模块 | 复用到方案 |
|---|---|
| 话题连贯分割模型 | Add 管线的 episode 切分（平台已按 20 消息/2000 词粗分，我们在 chunk 内再按话题细分） |
| 压缩去噪 | AMU 检索键去噪；embedding 用压缩后文本，content 用原文 |
| 粒度消融方法 | 本地评测时做轮级/段级/事实级消融，验证 AMU 粒度选择 |

## 注意

- 分割模型需要训练/微调（论文用标注数据）；AML 场景可用 gpt-4o-mini 做话题边界检测替代，零训练成本。
