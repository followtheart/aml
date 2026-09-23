# JEV 支持關係判斷

流程：現有模型語意選句與前提抽取 → 本地引用校驗 → JEV 獨立判斷 → 分歧回到 LLM 復核 → 原有完整選項驗證與治理約束 → 選擇答案。

JEV 判斷包含初始 `unsupported` 前提，因此可以觸發漏判復核；僅接收題目、日期、完整選項、抽取前提、原文來源及候選選句，不接收初始支持標籤、理由或標準答案。每個前提使用獨立 Choice 問題，答案為 `supported`、`contradicted`、`insufficient` 或 `not_a_premise`。最後一類用於辨認新建議被誤抽為既有個人事實的情形。

JEV 與初始判斷不一致，或 confidence 低於預設 0.8 時，原聊天模型以一次呼叫重審相關選項的全部前提。其他選項保持原判斷。復核必須保留初始前提的精確文字與順序，只能修改支持判斷及引用，不能藉刪除或改寫核心前提升格。所有新引用仍須通過原有來源、角色、原文字串及證據強度檢查；JEV 自身不能提升任何選項的資格。初始抽取漏掉的前提仍由後續完整選項驗證處理。門檻是待校準的操作參數，並非 80% 正確率承諾。

`not_a_premise` 可以觸發復核，但不授權模型直接刪除前提；只有原有本地建議識別規則能在抽取校驗時刪除。這是一項保守限制：本次接入不修復所有建議／個人前提抽取錯誤。

## 配置

在 `memory_system/.env` 設定 `OPENROUTER_API_KEY` 或優先級較高的 `AML_JEV_API_KEY`，重新啟動程序。既有聊天模型、embedding 與 reranker 配置保持獨立。

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `AML_CHOICE_JEV_SUPPORT` | `1` | `0` 關閉；無 key 或 `AML_FAKE=1` 時跳過 |
| `AML_JEV_MODEL` | `typesafe/jev-1.13` | OpenRouter 模型 ID |
| `AML_CHOICE_JEV_MIN_CONFIDENCE` | `0.8` | 低於門檻送交 LLM 復核 |
| `AML_JEV_TIMEOUT_SECONDS` | `20` | 整個 Decisions 階段，包含排隊與退避 |
| `AML_JEV_MAX_REQUEST_BYTES` | `48000` | 序列化請求上限；超限保留原判斷，不截斷來源 |

固定端點為 `https://openrouter.ai/api/alpha/decisions`，Bearer 鑑權；請求包含 `model`、`state`、`questions`。此介面與 chat completions 的契約不同。回應必須完整包含所有問題與分類、有效機率及 confidence；拒絕缺失、重複或非法 ID，不以猜測補齊。

完整功能啟用時，獨立回答預算由 9 增至 11 次提供者呼叫；共用外部預算時不擴張。新增階段各只留一次提供者呼叫額度，避免重試消耗後續至少四次呼叫的預留。JEV 超時、無效回應、LLM 復核失敗或預算不足時保留原判斷，並寫入診斷；呼叫取消會向上傳播。原有引用規則、否定／假設檢查、語意驗證與遺忘約束仍然生效。

## 診斷與驗證

`answer_jev_support` 記錄狀態、跳過／回退原因、模型實際版本、confidence、機率、分歧及復核前後判斷。`choice_alignment_initial` 保存初始抽取校驗結果；`choice_alignment` 保持最後判斷介面。`answer_calls` 中新增 LLM 復核階段 `eval.choice_jev_review`；JEV 提供者計量為 kind `decision`、stage `eval.choice_jev`。版本指紋包含模型和非機密設定，不包含 API key。

在倉庫根目錄執行：

```powershell
.\memory_system\.venv\Scripts\python.exe memory_system/scripts/selftest_jev.py
.\memory_system\.venv\Scripts\python.exe -m unittest discover -s memory_system/scripts -p 'selftest_*choice*.py'
# 預覽固定合成資料，不發出請求
.\memory_system\.venv\Scripts\python.exe memory_system/scripts/smoke_jev.py
# 已配置 key 後，以一次付費請求驗證支持、矛盾、證據不足三種關係
.\memory_system\.venv\Scripts\python.exe memory_system/scripts/smoke_jev.py --live
```

`--live` 缺 key／FAKE 時退出碼為 2；提供者錯誤或判斷未符合簡單合成案例時為 1；通過為 0。此測試只發送固定合成資料。

現有 `scripts/replay_answer_choice.py` 已共用新版回答預算，可使用相同 frozen fixtures，分別設定 `AML_CHOICE_JEV_SUPPORT=0`／`1`，輸出到不同的新檔案做對照。診斷可區分成功執行與回退；啟用開關不代表每題都成功呼叫 JEV。先比較支持度恢復與錯誤提升，再看最終得分、延遲和成本。程式接入與合成 smoke 都不能證明 PersonaMem 準確率提高。

官方參考：[OpenRouter 模型](https://openrouter.ai/typesafe/jev-1.13)、[OpenRouter Decisions 使用範例](https://openrouter.ai/labs/jev/compile)、[TypeSafe Choice 契約](https://docs.typesafe.ai/primitives/choice)、[confidence 定義](https://docs.typesafe.ai/confidence)。

## 本次實作驗證（2026-09-23）

- `VERIFIED`：專案 venv 執行 `memory_system/scripts/run_selftests.py`，42/42 套通過，退出碼 0；其中 JEV transport 14 項、JEV 流程 15 項。最後的路徑引號修正後，再跑設定、提供者相容性與 JEV 兩套，共 39 項通過。
- `VERIFIED`：相關模組 `py_compile`、`git diff --check` 及合成資料 dry-run 通過。
- 獨立 `kiro-review` 結論為 `APPROVED`。審查發現的「刪除 unsupported 核心前提後以次要前提取得 partial 資格」已修復；新增回歸先重現選 A 的失敗，再驗證修正後選 B。
- 原始 `memory-debug.jsonl`、`search-debug.jsonl`、`personamem-v2-32k.jsonl` 的 SHA-256 與分析快照一致。
- 已修正 `.env` 解析：移除值外且由空白分隔的行尾註解，保留引號內的 `#`、未分隔的字面 `#`、Windows 反斜線及路徑內的 apostrophe；既有環境變數仍優先。實際 `.env` 檔案未被改寫。
- `VERIFIED`：配置憑證後執行 `smoke_jev.py --live`，一次真實請求的 supported／contradicted／insufficient 三類合成案例全部通過。實際模型為 `typesafe/jev-1.13-20260917`，用量 570 input + 134 output tokens。
- `VERIFIED`：以既有 frozen fixtures 執行單題 QA3 真實回答回放，`fake=false`，5 次提供者呼叫，選 C、代理分數 1。JEV 判斷 4 個前提，A／C／D 的分歧或低信心觸發 LLM 復核，`answer_jev_support.status=reviewed`，後續語意驗證通過。[原始回放紀錄](runs/jev-live-20260923T0327/qa3.jsonl)。
- 26 題整體分數、延遲與成本變化尚未做對照評測；上述單題成功不能證明整體準確率提升。已啟動的長駐服務仍需重新啟動，才能讀取新 `.env` 與程式。
