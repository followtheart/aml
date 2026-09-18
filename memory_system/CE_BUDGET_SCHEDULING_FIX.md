# CE 預算調度修正

2026-09-18；搜尋策略 `graph_cascade_v8`。修復範圍是本地 `BudgetExceeded` 的調度與下階段預算保留，不修改模型、Gold、原始評測檔案或正式 `.env`。

## 問題與修正

舊排程先依剩餘額度一次建立全部批次，再讓佇列競爭兩個並行名額。第一批完成後，下一批可能在第二批仍持有大額 token 預留時啟動，立即被 `reserve_tokens()` 拒絕；待第二批結算後也不會重新准入。

新排程採有界波次：

1. 每波最多 `AML_RERANK_CONCURRENCY` 批，整波都完成計帳並釋放預留後才安排下一波。
2. 依最新剩餘 tokens 與 calls 重新計算批次大小。單份文件大於均分額度但可放入整體餘額時，降為單批處理。
3. 可負擔時，保留單份完整 listwise 候選所需的 token 空間及一次呼叫。token 空間由排程扣除，CE 仍由原有 `reserve_tokens()` 做最終准入檢查；沒有重複預留 CE 成本。
4. 呼叫保留由 `Budget.reserved_calls` 與 `before_call()` 執行，因此現有 429 重試也不能占用留給 listwise 的名額。CE 階段的 `finally` 釋放該名額。
5. listwise 預估與正式送出共用同一個完整請求上界：prompt、schema、system、框架及輸出額度。總額度不足整個池時，只減少整份候選，記錄 `token_budget`，不截斷原文。
6. 若連一份 listwise 都不可負擔，保留已取得的 CE 順序與既有有界 fallback。不可行的 listwise 不會無條件鎖住最後一次 CE 呼叫。

所有 CE 波次仍共用單一 `AML_CE_DEADLINE_SECONDS`；沒有按波次重置期限、提高 64k 預設值，或新增已送出請求的重試。波次屏障可能降低一快一慢請求下的排程吞吐，換取明確的結算與准入順序；應從新 trace 觀察實際期限內的評分覆蓋。

這個修正不保證每次都評完 50 份。真正的 token／call 上限、單份輸入上限、provider 錯誤或 deadline 仍會造成明確降級；未知分數保持未知。格式修復的第二次 LLM 呼叫仍是有剩餘預算才執行。

單份候選的 headroom 是資源保留，不會繞過 fine／listwise 的數量限制，也不保證當初可負擔的那份候選一定被後續排序選中。來源覆蓋與答案採用仍是另外的品質問題。

## 診斷欄位

`cascade.cross_encoder` 新增：

- `scheduler: settled_waves`。
- `waves`：各波可用 token 額度及批次編號。
- `batches[].wave`、`batches[].input_bound`。
- `listwise_token_headroom`、`listwise_call_headroom`。
- 未派發候選的 `reason: token_budget / call_budget / stage_deadline`。

`cascade.omitted` 可另外看到 listwise 的 `token_budget`。這些欄位區分實際額度不足、期限到達及已呼叫 provider 的失敗，避免只用一個 `BudgetExceeded` 推斷原因。

## 如何提高預算

在 `memory_system/.env` 設定並重新啟動載入設定的服務。例如以下是較寬鬆的實驗設定，並非已驗證的最優值：

```dotenv
AML_SEARCH_MAX_TOKENS=128000
AML_SEARCH_MAX_CALLS=20
AML_SEARCH_DEADLINE_SECONDS=300
AML_CE_TIMEOUT_SECONDS=20
AML_CE_DEADLINE_SECONDS=45
```

| 設定 | 範圍 |
|---|---|
| `AML_SEARCH_MAX_TOKENS` | 單次 Search 所有受計帳 provider 呼叫的累計 tokens 上限 |
| `AML_SEARCH_MAX_CALLS` | 單次 Search 的 provider 呼叫次數；重試亦計入 |
| `AML_SEARCH_DEADLINE_SECONDS` | 單次 Search 總期限 |
| `AML_CE_TIMEOUT_SECONDS` | 單次 CE 呼叫期限 |
| `AML_CE_DEADLINE_SECONDS` | 所有 CE 波次合計的階段期限 |

v7 日誌中的搜尋總期限為 300 秒，但 CE 階段仍為 12 秒；只增加總期限不會改變 CE 子階段限制。本輪最後檢查時，正式 `.env` 已是上述 128,000／20／300／20／45 的設定；本次修復沒有改寫該檔案。增加本地額度不會提高服務商的帳戶限流或模型上下文上限，且可能增加實際成本與延遲。

## 驗證

新增 `scripts/selftest_ce_scheduler.py`，使用現行 CE／metrics／budget 與本機 HTTP mock。所有憑證與端點為測試專用值，沒有外部模型呼叫。

- 修改前：7 個初始案例中 3 個失敗；兩個 45 份候選案例只評到 30 份，單份大文件案例為 0/1。
- 修改後：9 個排程回歸案例通過，包含完整候選及分數映射、整波結算、動態縮批、大文件降並行、真實不足、呼叫限制、429 保留名額、listwise 整份縮池、逾時與外部取消。45 份候選均恰好評一次，原文未改，結束時 token／call 預留為 0。
- 整體回歸：`scripts/run_selftests.py` **27/27 套件通過，388 項 unittest＋16 項 HTTP 契約檢查，共 404 項**，exit 0。輸出：[full-selftests-final.txt](C:/Users/aapoo/Desktop/aml/memory_system/runs/ce-budget-20260918/full-selftests-final.txt)。
- `compileall` 與 `git diff --check` 通過。獨立審查另驗證兩份大文件降單批處理，沒有阻擋項。

這些是工程契約與受控時序的驗證，不是修正後 PersonaMem 分數。先前的 9/26 結果與 v7 分析報告仍保留為歷史基線；本次未重跑真實模型評測。
