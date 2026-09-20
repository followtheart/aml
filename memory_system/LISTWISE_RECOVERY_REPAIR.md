# Listwise 誤刪修復與固定候選驗證

2026-09-20。搜尋策略：`graph_cascade_v10`。答案策略仍為 `verified-source-choice-v1`。

## 問題與修復

封存日誌中的 Q13、Q15 都已將相關原文送入 listwise，但合法的 `irrelevant` 結果把 10 個候選全部刪除。格式修復無法發現這種語義誤判；剩下的 6 條忘記規則又讓原本的狀態判斷誤認為已取回證據。

現在鏈路為：圖融合 → coarse → Cross-Encoder fine → 主 listwise → 有界刪除復核 → 原始證據打包。

- 合法 listwise 結果有刪除時，復核尚未被保留候選完整承載的原始候選；部分刪除也適用。
- 復核判斷原文對問題或選項的背景、興趣、限制及反證是否有用，不要求一個來源證明整個答案。候選混有不同主題時，一條相關來源即可提出恢復。
- 提案必須提供該候選自己的來源 ID、逐字引用，以及問題／選項中的逐字關聯片段。本地程式檢查 ID、可見原文、角色、引用與主體；第三人引述、假設和主題提問不能冒稱本人經歷。
- 只恢復原來的候選單元，按原 submitted 順序附加；原先保留的排序、群組、原文及來源不被覆蓋，也不新增個人事實或 supported coverage。最終仍走原本的完整單元／原子群組打包。
- 未完成復核的候選與「已判無用」分開記錄。沒有普通證據、只有規則時回報 `not_found`；如果還存在未完成的復核則回報 `incomplete`。`local_eval` 採用完整的 `search_degraded`，不再只看 rerank 狀態。

程式入口：[listwise_recovery.py](app/listwise_recovery.py)、[cascade_rerank.py](app/cascade_rerank.py)、[復核提示](../prompts/05d_listwise_recovery.txt)。搜尋狀態修改位於 [search_pipeline.py](app/search_pipeline.py) 與 [local_eval.py](scripts/local_eval.py)。

## 邊界與成本

一次搜尋最多新增 **1 次實際 provider 呼叫**，最多復核 10 個候選。完整提示不得超過 `min(12000, AML_RERANK_MAX_PROMPT_BYTES)` UTF-8 bytes；超限候選整項略過，後面的較小候選仍可進入。

復核與主 listwise、格式修復共用 `AML_RERANK_DEADLINE_SECONDS`，並受 Search 剩餘時間限制。本次環境的階段上限是 45 秒。prompt、schema、system 與輸出預留共用同一個 request token budget；底層傳輸重試不能另花第二個復核呼叫。沒有提高 `.env` 預算。

`cascade.listwise` 現在保留 `primary_selected_ids`、`primary_irrelevant_ids` 及 `recovery`。後者記錄已審／恢復／未審候選、完整來源對照、引用判定、prompt hash 與實際准入上限。

復核的相關性仍由模型判斷。逐字校驗能排除串位及偽造引用，不能證明語義判斷必然正確。容量不足、角色／主體核驗不通過、格式錯誤或逾時時，維持既有選中證據並明確降級。

## 固定候選真模型對照

輸入封存於 `runs/personamem-20260920-log-audit/inputs.zip`，SHA-256：`abd4f92171fc0728df404d4dff31c09978b8ce83e9a2be19823170711fc45f2d`。

回放固定原本的候選、fine 分數及主 listwise 決策，只執行新的刪除復核與實際打包函式。每題在呼叫模型前，先驗證重建的舊答案包 hash 與封存日誌相同。答案對照另外呼叫現有 `eval_scoring.evaluate`，模型為 `openai/qwen3-14b`；沒有重新抽取、召回或執行 CE。

| 題目 | 原普通證據單元 | 修復後單元 | 答案前 → 後 | 原文檢查 |
|---|---:|---:|---|---|
| Q13 | 0 | 2 | A → **D，正確** | 恢復 S11:3 的動漫政治／歷史主題提問；兩個候選承載同一條獨立原文 |
| Q15 | 0 | 1 | A → **B，正確** | 恢復 S7:1 的騎車與舊腿傷背景 |
| Q16 | 7 | 7 | A → **A，正確** | 原本保留的限定病史證據仍在 |
| Q22 | 7 | 8 | C → **C，正確** | 原先的電影主題證據保留，另外恢復一個候選 |

Q15 實際核驗引用：`I used to ride there a lot before an old leg injury started acting up again.` 最終答案包保留完整來源訊息，沒有只留下這個片段。

四題對照為 **2/4 → 4/4**。這是針對已知誤刪與控制題的固定輸入驗證，不能當作全量 Personamem 準確率。結果、原始模型回應、prompt 與 hash 保存在 [targets-final.jsonl](runs/listwise-repair/targets-final.jsonl)。

Q13 還有明確殘留：S6:2「觀看長篇字幕影集」在復核中仍被判為無用，未恢復。另有候選超過復核容量或引用判定無效，所以 Q13／Q15 仍可能標記 `partial`；救回關鍵來源不代表整份清單已完成復核。

## 26 題回放結果

原有 **155 個普通證據單元全部保留**；另外恢復 7 個通過來源校驗的候選，最終打包 162 個。恢復分布為 Q2×1、Q13×2、Q14×1、Q15×1、Q20×1、Q22×1。105 個原被刪候選中，85 個完成有效復核，20 個未完成：10 個超過提示容量、10 個模型提案未通過引用／主體校驗。每個未完成項都有獨立紀錄。

26 題復核狀態為 `ok` 13 題、`recovered` 3 題、`partial` 10 題；沒有逾時或 provider 錯誤。共新增 26 次 provider 呼叫、66,489 個實際計費 token。復核耗時中位數 **11.32 秒**，最大 **27.16 秒**；這是獨立回放的復核時間，不是正式搜尋端到端延遲。

Q13／Q15／Q16／Q22 在全 26 題回放與四題答案對照中，復核 prompt hash 和最終 packet hash 均相同。關鍵來源的完整文字另經 [target-source-checks.json](runs/listwise-repair/target-source-checks.json) 驗證。統計及版本指紋見 [final-summary.json](runs/listwise-repair/final-summary.json)，完整回放見 [all26-final.jsonl](runs/listwise-repair/all26-final.jsonl)。本版 pipeline fingerprint 為 `d365eed05b417ac84da5c1978c3a0fdb1589e17b9ae7ceb7d9181145810ca5c1`。

7 個恢復項不能全部算作已證明相關的獨立證據。例如 Q22 新增項把騎車／腿傷引文連到「American history」，關聯偏弱，雖然來源和角色合法，仍可能增加噪音；本次 Q22 答案維持正確。這個案例與 Q13 未救回的影集背景，說明模型的 usefulness 判斷仍有誤保留與誤刪風險。本次沒有為此再增加第二個語義模型呼叫，也沒有把恢復項升級成答案端的已證實前提。

## 離線驗證與重現

完整隔離自測 **33/33 組通過**：555 項 unittest，加 16 項 contract 檢查。新增復核套件有 33 項測試，涵蓋全刪／部分刪除、原選中項保留、跨候選引用、第三人／假設、否定／過往背景、整來源准入、呼叫／token／時間上限及取消釋放。

獨立審查發現並修正了句首 `My friend wrote/says` 等引述識別，以及空白／純代名詞引用的漏洞。RED、GREEN、全量自測及審查紀錄均在 [runs/listwise-repair](runs/listwise-repair/)；最終審查為 [APPROVED](runs/listwise-repair/review.md)。

原始 memory/search 日誌、結果及 prepared 資料四個檔案均未變更，見 [input-integrity.json](runs/listwise-repair/input-integrity.json)。回放結果使用另外的檔案，不覆寫原始結果。

```powershell
memory_system\.venv\Scripts\python.exe memory_system/scripts/run_selftests.py
memory_system\.venv\Scripts\python.exe memory_system/scripts/replay_listwise_recovery.py --questions all --output memory_system/runs/listwise-repair/new-replay.jsonl
memory_system\.venv\Scripts\python.exe memory_system/scripts/replay_listwise_recovery.py --questions 13,15,16,22 --answer --output memory_system/runs/listwise-repair/new-targets.jsonl
```

回放檔案以 exclusive create 寫入，已存在時會拒絕覆寫。正式效能仍需後續重新執行完整搜尋鏈路評估；本次固定回放使用獨立的復核預算，不能模擬前序階段已消耗的時間與 token。
