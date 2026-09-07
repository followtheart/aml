# 文本数据集接入

本地入口现在支持 JSON 数组、JSONL、CSV 和 Parquet。模型调用仍使用现有
`.env` 配置。本文命令均在 `memory_system` 目录执行。

## 已接入范围

| 数据集 | 本次本地数据 | 评分方式 | 说明 |
|---|---:|---|---|
| 原有 LoCoMo | 10 conversations / 1542 QA | 二元 LLM 判分 | 保留原转换器及现有文件 |
| LoCoMo-Refined | 10 conversations / 1382 QA | 严格二元判分，任一完整候选答案通过即通过 | mem-eval-suite 公开版本，文本模式 |
| LongMemEval-S cleaned | 500 histories / 500 QA | 二元 LLM 判分，保留拒答标记 | 上游公开 cleaned 版本 |
| BEAM-100K | 20 conversations / 400 QA | 每项 rubric 0/0.5/1，取均值 | 仅 100K 档；不是全部 BEAM 档位 |
| CLBench | 1899 tasks / 1899 QA | 全部 rubric 通过才记 1 | 完整上游集合，未假定等于平台长度子集 |
| PersonaMem-v2 text 32k | 200 histories / 5000 QA | 单选精确匹配 | 200 份配套聊天历史已下载；另支持 128k |
| ScriptMem | 4 份题目文件 / 457 QA | 单选、多选集合、排序精确匹配 | 上游未发布剧本原文，须提供真实历史后才能入库评测 |
| LongMemEval-Refined | 无平台原始包 | 取决于授权导出中的题型 | 见下文 |

这些结果统一标为 `local-text-proxy-v1`，**不等同于 AML 官方分数**。
平台负责统一 Answer/Eval，本地保留现有 LLM 配置，并使用本地评分提示词。
模型、提示词、数据版本和评分细节没有完全复刻官方契约。
BEAM 此处实现 rubric 均分，不实现其额外的事件排序/检索辅助指标。
PersonaMem 此处实现 MCQ，不实现开放生成式偏好评分；选项用 SHA256 稳定排列，
不使用上游依赖 Python 进程种子的 `hash()` 排列。

## 安装、下载和准备

```powershell
python -m pip install -r requirements.txt -r requirements-eval.txt
python scripts/download_datasets.py all
```

下载器从官方仓库/Hugging Face 下载；保存 URL、ETag、SHA256、下载时间和
来源说明。重复运行校验已有文件后复用，失败下载保留原文件并清理临时副本。
原文、转换结果、下载参考文件和评测输出已加入 `.gitignore`，不自动提交第三方语料。

```powershell
python scripts/prepare_dataset.py --dataset longmemeval-s --input data/raw/longmemeval-s.json --output data/prepared/longmemeval-s.jsonl
python scripts/prepare_dataset.py --dataset locomo-refined --input data/raw/locomo-refined.json --output data/prepared/locomo-refined.jsonl
python scripts/prepare_dataset.py --dataset beam --input data/raw/beam-100k.parquet --output data/prepared/beam-100k.jsonl
python scripts/prepare_dataset.py --dataset clbench --input data/raw/clbench.jsonl --output data/prepared/clbench.jsonl
python scripts/prepare_dataset.py --dataset personamem-v2 --input data/raw/personamem-v2.csv --history-dir data/raw/personamem-v2 --output data/prepared/personamem-v2-32k.jsonl
```

输出附带 `.manifest.json`，包含实际 conversation/session/QA 数、空消息清理数、
转换输出 SHA256 和本地协议标识。`--convs N` 可以只转换前 N 份历史。
默认不截断原始历史；同一 PersonaMem 历史文件的题目共用一次入库，不同历史
文件即使 persona 相同也保留独立检索范围。

PersonaMem 128k：下载器加 `--persona-size 128k`，转换器加 `--size 128k`。
下载器 `--persona-convs N` 只下载前 N 份历史，转换时应配套 `--convs N`。

## 校验与运行

先检查数据，不产生外部模型调用：

```powershell
python scripts/local_eval.py --data data/prepared/longmemeval-s.jsonl --inspect
python scripts/local_eval.py --data data/prepared/beam-100k.jsonl --inspect
python scripts/local_eval.py --data data/prepared/clbench.jsonl --inspect
python scripts/local_eval.py --data data/prepared/personamem-v2-32k.jsonl --inspect
```

使用当前配置评测选中的一题，保存详细结果：

```powershell
python scripts/local_eval.py --data data/prepared/longmemeval-s.jsonl --convs 1 --limit 1 --output data/results/longmemeval-s.jsonl
```

其他数据集替换 `--data` 即可。也支持直接输入上游数据，例如：

```powershell
python scripts/local_eval.py --dataset beam --data data/raw/beam-100k.parquet --convs 1 --limit 1 --inspect
python scripts/local_eval.py --dataset personamem-v2 --data data/raw/personamem-v2.csv --history-dir data/raw/personamem-v2 --convs 1 --limit 1 --inspect
```

`--limit` 限制总 QA 数，仍然导入这些题目对应的完整历史。大语料读取与执行均按
conversation 流式进行，预检后再执行；不会将 277 MB 的 LongMemEval 整体加载成
Python 对象。CLI 的 `--output` 是逐题 JSONL，现有文件会被覆盖，不支持续跑。
输出含 dataset、conversation_id、qa_id、prediction、score、scoring、protocol、
fake、model；回答/评分失败时额外记录 error_stage/error_type，计零分而不隐藏错误。
rubric 均分允许小数，汇总显示 mean score，不把部分得分截断成整数准确率。

离线运行设置 `$env:AML_FAKE='1'`。Fake 分数无意义；Fake rubric judge 返回
格式正确的零分结果，只验证管线，不验证真实评分质量。

## 接口和数据边界

- 默认按最多 20 条消息 / 2000 个空白分隔词拆分 Add。优先完整消息、句子边界，
  超长无句号内容在词边界拆分；保留所有文本。`--chunk-messages` / `--chunk-words`
  可覆盖上限，改变上限会改变本地评测流程。
- 同一来源 session 的各 chunk 共用 session_id；每个 chunk 的 request_id 唯一。
  user_id 包含数据集和 conversation，Search 使用相同 user_id。
- Search.query 保留原始问题；选择题 options 单独传入。标准答案、gold label、
  evidence、rubrics 只用于评分/输出元数据，不进入 Add 或 Search。
  选择题自然包含正确选项文本，但不会向模型指出哪个选项正确。
- LongMemEval 的 haystack 日期转换为毫秒；无时区来源日期按 UTC 解释以确保跨机器
  一致。保留 question_date、原始证据 ID。重复的来源 session ID 按出现位置区分，
  不丢弃后续出现；实际清理了 12 条空消息。
- PersonaMem 保留官方聊天历史（包括原始 system 文本，以 `[system]` 前缀作为
  user 消息存储，满足文本 Add 契约），不将 CSV 中的 gold preference/profile
  字段另行注入。实际清理了 1 条空消息。
- CLBench 上游把参考文档和问题合在同一 user 消息中。本地保留完整文本用于入库和
  问题，不猜测切分点；system_prompt 用于 Answer，rubrics 只给 Judge。
  因为 Answer 仍可见问题中的原文，它不能单独证明纯检索记忆能力，也不是平台
  0–4k / 16–32k 私有切分的复现。长 query 的模型上下文限制由当前模型决定。
- ScriptMem 保留角色名和叙述，拒绝用 `format_example` 合成示例替代原始剧本。

## ScriptMem 和 Refined 数据

ScriptMem 公开题目在 `data/raw/scriptmem/{angry,enemy,friends,man_earth}.json`。
取得可使用的真实历史后，提供同名历史文件，例如 `data/script-histories/angry.json`：

```json
{
  "conv-0": {
    "session_1_date_time": "Unknown",
    "session_1": [
      {"speaker": "角色名", "text": "这里必须是真实来源文本"}
    ]
  }
}
```

```powershell
python scripts/prepare_dataset.py --dataset scriptmem --input data/raw/scriptmem/angry.json --history-dir data/script-histories --output data/prepared/scriptmem-angry.jsonl
```

没有原文时程序报错并停止，不输出伪造的完整评测数据。
LoCoMo-Refined 已接入用户提供的公开仓库，之前将其概括为未公开不准确。
直接运行以下命令即可获取和检查数据：

```powershell
python scripts/download_datasets.py locomo-refined
python scripts/prepare_dataset.py --dataset locomo-refined --input data/raw/locomo-refined.json --output data/prepared/locomo-refined.jsonl
python scripts/local_eval.py --data data/prepared/locomo-refined.jsonl --inspect
python scripts/local_eval.py --data data/prepared/locomo-refined.jsonl --convs 1 --limit 1 --output data/results/locomo-refined.jsonl
```

共 10 组对话、272 个来源 session、5882 条消息、1382 道题，默认产生 399 次 Add。
保留官方 `sample_id#qNNNN`、原始类别编号、证据 dia_id、说话者与时间。
`answer` 数组表示替代答案；每个元素本身是完整答案（包括数值转字符串），不能
将整个数组当作必须同时包含的事实集合。完全匹配直接通过，否则逐候选严格判分。
评分提示词是基于上游规则的本地改写，记录 `locomo-refined-local-v1`，包含时间
粒度、相对/绝对日期及事实列表规则；仍使用现有 AML_LLM_MODEL，不自动切换为
上游官方 Qwen3-14B，也未实现上游 F1/BLEU，所以不能称为官方复现分数。
逐题输出含官方评分器可读取的 `qa_id`、`predicted_answer`。

保留所有 1382 题，其中 521 题带 `is_multi_modality=true`。当前只输入原始消息
`text`，不下载图片、不注入 blip_caption；这类题缺少视觉证据的限制需随分数披露。
不把 event_summary、observation、session_summary 等辅助标注作为记忆输入。
下载器同时保存 CC-BY-NC-4.0 LICENSE、NOTICE 和公开 questions.jsonl 用于追溯。

LongMemEval-Refined 适配相同 `haystack_*` 字段的授权导出。
无法核实的平台专有字段不能自动推断，可先显式映射成 normalized 格式。
LongMemEval-Refined 入口仅是导入能力，未验证其私有格式。
已下载的公开 LoCoMo-Refined 未核验与平台当前托管包的哈希是否完全一致。

## 验证

```powershell
python scripts/selftest_datasets.py
python scripts/selftest_contract.py
```

测试覆盖时间、重复 ID、空消息、选项排列、多选/排序、rubric 部分得分、缺项拒绝、
题目/答案隔离、长文本无损分块、流式 limit、缺失剧本拒绝及 Add/Search 请求回归。
本次四套公开数据均完成全量转换/校验；每套首份历史首题完成离线完整管线冒烟
（内存 SQLite、Fake 模型；未调用付费模型）。

## 来源

- [AML 文档与 Add/Search 契约](https://agentmemoryleaderboard.ai/docs#doc-contract)
- [AML 公开评测代码及数据不公开说明](https://github.com/AML-memory/agent-memory-leaderboard)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) / [cleaned 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined)（基于 Snap Research LoCoMo 修改，CC-BY-NC-4.0）
- [BEAM](https://github.com/mohammadtavakoli78/BEAM) / [数据](https://huggingface.co/datasets/Mohammadta/BEAM)
- [CLBench](https://github.com/Tencent-Hunyuan/CL-bench) / [数据与许可](https://huggingface.co/datasets/tencent/CL-bench)
- [PersonaMem-v2](https://github.com/bowen-upenn/PersonaMem-v2) / [数据](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2)
- [ScriptMem（原文未包含）](https://github.com/memorax-ai/ScriptMem)

原始语料和评分定义归各上游作者所有，使用与再分发须遵循各自许可。
