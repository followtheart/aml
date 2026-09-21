# 正確答案漏斗修復與驗證

本次依 A–F 設計修復候選保留、引用校驗、語境、前提分類與格式失敗處理。下文記錄初次修復時，對原始 26 題凍結資料和離線正反例的驗證；當時未重新呼叫模型，不能把離線關卡改善當作答對率提升。

後續實際評測已由原本 **10／26（38.46%）變為 12／26（46.15%）**：新增答對 QA 1、13、21，退步 QA 14。重新寫入的 AMU 及模型提案也有變化，這不是單變量消融。後續漏斗仍指出 soft fused-head 在 listwise 變成硬保留、首次 support 未引用使後續核驗無來源、引用類型與遺忘約束範圍等待處理問題；本提交未聲稱解決這些新發現。

## 實作對照

| 設計 | 實際行為 |
| --- | --- |
| A1 | 沿用既有 `option:*` requirements，每個選項最多保留一個可見來源代表；總數不超過該層名額的一半。以選項主題覆蓋與使用者來源優先，保留理由記為 `option_topic_coverage`，沿用既有機制傳到 listwise。 |
| A2 | 採用設計允許的 A1 路徑，不增加逐選項 CE 呼叫。 |
| A3 | 保留名額也執行可見來源去重；完整被其他候選涵蓋的片段不優先佔位。相同事件的不同片段、時態／否定限定及圖關聯義務仍分別保留。 |
| B1 | 有有效個人來源 anchor、且無效引用數不超過有效引用數時，剝除無效引用並記錄 `citation_dropped`。重複引用不重複計數；被丟棄引用與錯誤原因另存於 `dropped_citations`。 |
| B2 | 使用引用所在句子的第一人稱語境；親友歸屬依明確主詞子句切分，保留否定、假設與引述守門。 |
| B3 | 在完整 persona 訊息節錄之前標記 `source_role=persona`，由既有 packet hash 保護；裁判看到 `persona (first-party profile)`。不靠節錄開頭重新猜身分。 |
| B4 | claim 增加 `premise_type`，保留舊輸入的 `unknown` 相容值。興趣表述新增 into／big on／obsessed／follow；類型必須與表述一致，不能把所有權、病況、頻率或職業偽裝成興趣。 |
| C1 | 語義檢查附上同 request、訊息索引相距最多 2 的可見來源，標記 `anchor=false`，僅用於解析指代；不額外取回隱藏歷史。 |
| C2 | 支持提案與原文 anchor 有效、但語義否決時，至多做一次完整可見包復核。復核失敗、超額或再次拒絕均保留原否決；呼叫、token 與時間預算有保留額度。 |
| D1 | 移除明確的未來建議補語，重算 claims、primary claim 與 kind；保留陳述主詞、所有權、否定、頻率及歷史限定。無差異化資訊的「日常活動／需求」可視為一般建議背景。 |
| D2 | `partial` 若核心前提單獨通過、完整選項也通過、缺失次要項仍有有效 anchor 且無本地錯誤，可進入 supported 選擇層。原始 status 保留 partial，額外記 `selection_tier` 與原因，避免偽造已逐項驗證的紀錄。 |
| E | 唯一近似片段可修正空白／撇號及受限的單字內漏字／重字；不允許任意同義改寫或字元替換。支持提取及一次修復仍失敗後，改為完整選項的 generic-only 獨立驗證，再套用遺忘約束。沒有合格選項仍明確失敗，不強選未驗證的個人事實。 |
| F | 實作 `inferred` 選擇層，順序為 supported → partial → inferred → generic。預設關閉；僅允許具 anchor 的興趣／經驗推論，仍需語義驗證。所有權、診斷、頻率、專業能力等不得靠此層補造。 |

主題同義詞僅用於檢索名額，包括 vinyl／LP／pressing、crypto／blockchain／stablecoin、cholesterol／lipid；**不會因此產生新的個人事實或跳過 entailment**。原有 `_supported_coverage_ids` 與這種主題名額保留保持區別。

## 凍結重放結果

重放先對所有選項運行新邏輯，最後才讀取 gold 作報告標註。原始 memory、search、results 三個檔案的 SHA256 均與凍結快照相同。

| QA | 原本的關卡狀態 | 重放確認 |
| --- | --- | --- |
| 1 / D | partial | 移除輔導員及散步建議後，本地狀態為 supported。 |
| 8 / D | persona 節錄未標識，語義否決 | 該題 7 張 persona 來源卡恢復身分；需重新執行裁判才知道是否改判。 |
| 10 / C | unsupported | 移除 `your day-to-day activities` 偽前提後為 generic。 |
| 13 / D | unsupported | `into anime` 興趣前提通過本地引用檢查，整個選項為 partial。 |
| 15 / B | unsupported | 有效腿傷自述不再被助手引用連帶否決；移除新增散步建議後為 supported。 |
| 19 / B | unsupported | 恢復 persona 語境、剝除無效引用並移除紀念品建議後為 partial；不表示原有遺忘約束已解除。 |
| 20 / B | 黑膠來源在 fine 被截掉 | `amu_a9763e28af3140c8` 進入 fine 與 listwise 數量名額。 |
| 21 / C | 指代所需助手語境未送裁判 | `C:option` 與 `C:0` 現在附帶相鄰語境；layer-2 那條來源未在此次名額重放中恢復。 |
| 24 / B | 支持結構失敗，沒有可重放 claims | 以格式失敗→修復→generic-only 驗證的合成端到端案例驗證；不能從舊紀錄推導新答案。 |
| 25 / A | 膽固醇來源在 fine 被截掉 | `amu_b116caef58bc441b` 進入 fine 與 listwise 數量名額。 |

26 題共有 90 張 persona 來源卡恢復宣告身分。QA 6、16 的既有正確選項在本地引用檢查也由 partial 改為 supported。QA 0、4、5、17、18、23 的模型首次 unsupported 提案仍需新模型判斷；本次不把詞義相近、假設擁有或詢問某主題直接當成既有習慣。

重放固定舊 coarse 候選與 CE 分數，未執行新的 coarse 召回、CE、listwise 排序、語義復核或最終選擇。listwise 重放只驗證數量名額，不宣稱必然通過實際 prompt／token 預算。

## 設定與可重現命令

- 搜尋版本：`graph_cascade_v11-option-coverage`。
- 回答版本：`verified-source-choice-v3-funnel`。
- `AML_CHOICE_ENTAILMENT_REVIEW=1`：預設開啟一次語境復核。
- `AML_CHOICE_ALLOW_INFERRED=0`：預設關閉；應先沿用受控驗證的正反例與未參與設計的題目評估後，再決定是否啟用。
- 兩個開關均記入評測版本 metadata。

在專案根目錄執行：

重放命令依賴本機凍結快照目錄；大型原始日誌、結果與逐題分析資料不納入版本控制。

```powershell
.\memory_system\.venv\Scripts\python.exe memory_system/scripts/run_selftests.py

.\memory_system\.venv\Scripts\python.exe memory_system/scripts/replay_funnel_repairs.py --snapshot-dir memory_system/data/analysis/personamem-v2-32k-20260921-funnel --output memory_system/data/analysis/personamem-v2-32k-20260921-funnel/repair_replay.json

.\memory_system\.venv\Scripts\python.exe memory_system/data/analysis/personamem-v2-32k-20260921-funnel/trace_gold.py
```

`trace_gold.py` 使用與原始版本 SHA256 完全一致的 `answer_choice.baseline.py`，保留舊漏斗 26 → 25 → 16 → 13 → 11 → 11 → 10 → 10 的可重現性。`replay_funnel_repairs.py` 才是新程式的離線反事實重放；兩者不可混讀。

## 驗證記錄

最終執行 `run_selftests.py`：**38／38 個測試套件通過，退出碼 0**，其中新增漏斗修復測試 24 個。既有契約套件亦通過 16／16 項，涵蓋應用載入及 `/health`。另外，8 個新增／修改 Python 檔案通過 AST 語法檢查，`git diff --check` 通過。

新增測試先取得 RED：初始 8 個漏斗案例在舊程式失敗 7 個。後續擴充涵蓋第三方、否定、假設、能力陳述、過去事件、類型偽裝、興趣同義表述、persona hash 竄改、去重、低 CE 選項來源、復核預算、格式修復與遺忘約束。

獨立審查曾發現並修復建議誤刪、親友子句歸屬、類型標籤繞過與興趣正例回歸；最終限定審查通過。格式錯誤處理另覆蓋 provider 包裝的 `ResponseParseError`，一般傳輸失敗不會被偽裝成格式失敗。

全套測試輸出位於 `data/analysis/personamem-v2-32k-20260921-funnel/selftests-repair.txt`。原先敏感資料預設測試與已提交的 `15501fc` 行為不一致，本次更新為既有預設並增加明確 opt-out 的實際 storage 過濾檢查；沒有改動 API 敏感資料預設。
