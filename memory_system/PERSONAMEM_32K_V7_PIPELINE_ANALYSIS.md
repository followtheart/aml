# PersonaMem 32k：記錄、召回與圖融合級聯重排審計

後續更新：本報告保留 v7 執行的原始分析；其 CE 預算調度問題已在 `graph_cascade_v8` 修正，見 [修正與驗證](C:/Users/aapoo/Desktop/aml/memory_system/CE_BUDGET_SCHEDULING_FIX.md)。下方的指紋一致性與只讀範圍均指審計當時；未重跑原始 PersonaMem 評測。

審計日期：2026-09-18。分析對象為本次 `memory-debug.jsonl`、`search-debug.jsonl` 與 `personamem-v2-32k.jsonl`，結果 **9/26＝34.62%**。本報告只分析與離線重現，未修改生產程式、原始日誌、評測資料或模型設定，也未呼叫外部模型。

**最明確的工程缺陷在 J 階段：並行 CE 批次把暫時的預算預留衝突當成永久失敗，26 次搜尋全部部分降級，1,300 個粗排候選只有 645 個得到 CE 分數。** 最後裝包沒有再遺失已選候選，但這不代表先前裁剪或答案採用正確。另有明確的答案端問題：Q9 已拿到遺忘規則仍違反；Q25 已拿到膽固醇自述仍選一般建議。

不能把 17 題錯誤都歸為檢索失敗。部分 Gold 把「問過某話題」提升成「擁有某物／具有某偏好」，超出可見來源的支持強度；應與可重現的工程缺陷分開處理。

## 1. 資料對齊與版本確認

| 驗證項 | 本次結果 |
|---|---|
| 對話 | `911d1d0140349ab4dd01`，189 則原始訊息，95 user／94 assistant |
| Add 記錄 | 14 次提交，231 個記憶快照列，229 個唯一 AMU ID |
| 搜尋／作答 | 26／26；全部 search ID 與 packet hash 對應成功 |
| 題目／選項／Gold | 與 prepared dataset 逐項對齊，分數核對 26/26 |
| 原文 | Add 日誌中的 189 則正文與 prepared dataset 完全一致 |
| 管線 | `graph_cascade_v7`；答案策略 `direct_evidence_v2` |
| 模型 | `qwen3-14b`；embedding `text-embedding-v4`；CE `qwen3-rerank`；非 fake 模式 |
| 程式與提示 | 日誌記錄的 51 個檔案 SHA-256 均與目前工作區一致 |
| 搜尋時間 | 中國時間 2026-09-18 16:27:42 至 16:34:25 |
| 實際限制 | 每次搜尋 300 秒、12 次 provider 呼叫、64,000 tokens；CE 並行 2，階段期限 12 秒；coarse 50 → fine 12 → LLM 至多 10 |

程式指紋：`474a815e91bdbc6b07d34c698646b9b896e7ffd98086da8fdab28f70e00751af`。

計數、逐題來源、候選去留與輸入 SHA-256 收錄於 [audit.json](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-v7-audit/audit.json)，可用 [analyze.py](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-v7-audit/analyze.py) 重算。來源編號 `S8:7` 表示第 8 個 Add chunk 的第 7 則訊息，兩者皆從 0 起算；對應 memory JSONL 第 9 行。`Q16` 表示 `qa_id=16`，對應本次 search／result JSONL 第 17 行。

## 2. 本次鏈路實際做了什麼

```text
189 則訊息 → 14 次 Add → 229 個唯一記憶 ID
                                  ↓
26 次規劃 → 多路召回 3,314 條 route entries
          → 圖融合 2,000 個題內去重候選
          → 證據準備 1,921 個候選
             ├─ 156 個受保護規則（每題 6 個）──────────────┐
             └─ 1,765 個普通候選                        │
                 → coarse 1,300                         │
                 → CE 成功 645；未評分 655                │
                 → fine 312                             │
                 → LLM 輸入 248 → 保留 130                │
                                                        ↓
                    選中 286 → 裝包 286 → 答對 9/26
```

上圖是 26 題的累計候選出現次數，不能當成跨題唯一記憶數。未評分候選仍可透過有限保留名額進 fine，故 CE 成功數不是後續所有階段的母集合。

| 階段／訊號 | 數據 | 含義 |
|---|---:|---|
| 召回路徑 | vector 1,040；source_text 1,040；full_text 520；graph 144；scene 168；profile_rule 321；temporal 81 | 有圖與場景召回，不能說它們未執行；跨路徑包含重複 |
| 圖融合候選 | 每題 63–126，平均 76.92 | 大部分題目並非完全沒有候選 |
| 證據準備 | 2,000 → 1,921 | 79 次 `same_preference` 合併；不是 79 條來源全失 |
| CE 批次 | 159 批：87 成功、72 `BudgetExceeded` | 26/26 搜尋的 CE 與總重排狀態皆為 `partial` |
| CE 候選覆蓋 | 645/1,300＝49.62% | 平均只評 24.81/50；未評分不等於低相關 |
| LLM listwise | 26/26 `ok` | 格式與呼叫成功，不等於選擇正確 |
| 候選裁剪原因 | coarse 465；fine 988；listwise limit 52；prompt budget 12；irrelevant 118 | 大量損失發生在最後裝包之前 |
| 裝包 | 選中 286、輸出 286；packet omission 0 | 本次沒有已選候選被最後裝包再次刪除 |
| 證據群組 | 19 組；沒有整組裝包丟失 | 保持原子性有效，但無法補回進組之前被裁掉的證據 |
| 最終包大小 | 平均 8,900、最大 17,199 bytes；上限 32,000 | 單純擴大最終包上限不會補回前段候選 |
| 搜尋耗時 | 平均 14.46 秒，10.30–22.62 秒 | 本次不是 300 秒搜尋總期限耗盡 |

`packet omission=0` 僅指項目／群組。證據單元內仍可能有來源正文因 `unit_budget` 或 `source_limit` 未展開，例如 Q19 的 S8:3。審計另外檢查來源正文及關鍵原句，沒有把「ID 存在」直接等同「模型看到了必要內容」。

## 3. J 階段的可重現缺陷：暫時預留被當成永久超支

相關程式：[cascade_rerank.py:117](C:/Users/aapoo/Desktop/aml/memory_system/app/cascade_rerank.py:117)、[cross_encoder.py:92](C:/Users/aapoo/Desktop/aml/memory_system/app/cross_encoder.py:92)、[budget.py:34](C:/Users/aapoo/Desktop/aml/memory_system/app/budget.py:34)。

目前先以「當下剩餘額度 ÷ 並行數」一次決定批次大小，再讓各批次競爭並行名額。每批先保留保守輸入上界；若 `已用 + 已預留 + 新批預留 > 上限`，立即拋出 `BudgetExceeded`。級聯層把它記成批次失敗，沒有等在途請求結算後再評估，也不重新縮批。

因此，某批完成並釋出執行名額時，另一批仍保留大量額度；排隊批次拿到名額後可能立刻失敗。即使稍後額度恢復，它們也不會再嘗試。預算保護本身需要保留，缺陷在調度層將「暫時無可用預留」與「無法在總額度內完成」混為一談。

本次證據：

- 72 個錯誤全為本地 `BudgetExceeded`，沒有 CE provider 超時或網路故障記錄。
- 每題最終記帳只有 18,287–28,930 tokens，平均 23,172.77；呼叫 5–7 次，均低於 64,000／12 的上限。
- **每個失敗批次的保守輸入上界，加上該題最終已用 tokens，都仍低於 64,000，72/72 成立。** 這支持等待結算後逐批重新准入；不證明把所有失敗批次一起補跑也必然夠用。
- 查詢及四個選項對每份 CE 文件重複計入保守上界，使每題 50 份候選的累計上界達 137,041–188,392。不能直接把這個上界當成真實模型用量；也不能依最終低用量就移除預算保護。

用現行生產函式與 `httpx.MockTransport` 控制時序，已離線重現：

| 時點 | 已用 | 在途預留 | 新批需要 | 結果 |
|---|---:|---:|---:|---|
| 第二批仍在途、第三批取得執行名額 | 10,000 | 30,000 | 30,000 | 70,000 > 64,000，第三批永久失敗 |
| 前兩批全部結算 | 18,000 | 0 | 同一批 30,000 | 此時可准入 |
| 對相同第三批重試並結算 | 26,000 | 0 | — | 15 份候選全部取得分數 |

重現前級聯只評 30/45 份候選；等待結算後第三批能成功。這是調度機制的因果重現，**不是原始執行的精確時序重播，也不是修復後準確率預測**。腳本：[reproduce_reservation.py](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-v7-audit/reproduce_reservation.py)；結果：[reservation-reproduction.json](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-v7-audit/reservation-reproduction.json)。外部呼叫為 0。

Q16 尤其直接：本人童年輕微氣喘的 fact `amu_7528b17bce3b44d6` 與 episode `amu_8abfce446e48480a` 都進 coarse，但分別落在失敗的 CE 第 4／3 批，之後皆被 `fine_limit` 排除。它們的分數是「未知」，不能解釋成 qwen3-rerank 判定氣喘不相關。

既有回歸與小型真實 smoke 通過未涵蓋這個條件：先前真實案例僅 17 個候選，未形成這次 50 個候選、5–7 批、在途預留與新批准入交錯的壓力。保護不超支的測試不能代替「等待可釋放額度時不無故丟候選」的測試。

## 4. 線索在哪裡斷掉

此處使用人工核對的原文線索與反證，不是資料集提供的官方 relevant-document 標註。來源可追蹤不等於 Gold 前提成立；受保護規則視為繞過排序階段。來源亦可能經同源代表或 dependency 進包，不能只比較單一 AMU ID。

```text
26 題原文線索 → 26 題有記憶載體 → 25 題召回
→ 25 題融合／準備 → 24 題 coarse → 18 題 fine
→ 17 題送入 listwise → 14 題保留且來源可見
→ 其中 6 題答對
另有 Q2、Q7、Q12：追蹤線索未走完整鏈路，仍答對
```

17 題失敗按「最早可觀察的追蹤線索斷點」分布：

| 斷點 | 失敗題 | 題數 |
|---|---|---:|
| 召回前後 | Q5 | 1 |
| coarse | Q14 | 1 |
| fine | Q4、Q8、Q11、Q16 | 4 |
| fine → listwise 名額 | Q3 | 1 |
| listwise 判為 irrelevant | Q10、Q18 | 2 |
| 原文線索已入包，仍答錯 | Q0、Q9、Q17、Q20、Q21、Q23、Q24、Q25 | 8 |

因此是 **9 題回答前有線索流失，8 題線索已到答案端**。這不是互斥、已證明的根因分類：例如 Q5 的追蹤線索本身就不能證明 Gold 所需的「運動傷害造成下背痛」。恢復線索不代表一定修正答案。

幾個需要不同修法的代表案例：

- **Q3：fine 選入後，又被 listwise 配額擠掉。** S8:7 本人旅行故事有 `waxing my board`、`still-glass waves`。`amu_df9b9ed1f2bd45c5` CE 0.278750、fine 第 9，卻在進 LLM 前以 `listwise_limit` 丟棄。兩個未取得 CE 分數的社區活動候選獲保留名額，最終又被 LLM 判 irrelevant。應重新檢查 12→10 的覆蓋與未知候選配額；不是擴大最後 packet 就能解決。
- **Q8：一般知識勝過個人用途。** 原始 persona 明說 Facebook 用於家人動態，只有大型 episode `amu_3c50f12135b14360` 攜帶這項個人資訊，CE 0.234182 後被 fine 排除。一般 Facebook／網路效應提問卻入包。必要原句仍在該 episode 的重排預覽中，不能歸咎預覽把 Facebook 截掉；這暴露記憶粒度和「主題相關／個人前提支持」之間的差距。
- **Q14：事件召回與粗排代表不足。** S2:12 是本人安排家人去漫展；event `amu_32ff9ff04dfc4a8a` 沒有召回，攜帶同源的 episode `amu_7a0339cc06694966` 僅 temporal 路徑命中，融合第 94，於 coarse 被排除。未見它因期限或 foresight 過濾而掉落的證據。
- **Q9：規則到達，答案仍違反。** 忘記現代電子音樂節偏好的規則 `amu_b6a068b826964715` 受保護且完整位於包內；預測 B 仍依這個偏好推薦音樂節。歷史 assistant 推薦也透過 episode 留在包內。應處理約束與歷史內容的衝突、答案選项的規則違反，不能只增加規則召回。
- **Q25：支持個人化的證據已經完整。** `amu_82e1c659b89e4a04` 是第一個普通證據，CE 0.4036，S5:5 第一人稱 `my diet and exercise` 原文完整到達。這支持「留意膽固醇」，無需推論已確診高血脂；Gold A 正是依此個人化。預測仍選通用 D，問題位於答案採用／選項判斷邊界。結果只保存字母，無法再斷言模型內部具體原因。

## 5. 記錄語義、規劃與圖融合的限制

### 記下來、建好索引，不代表記對了

229 個唯一記憶快照包含 episode 53、preference 64、fact 80、rule 7、event 20、plan 4、profile 1；有向量與全文索引各 229，143 個記憶帶有共 190 個 triples。這排除了「普遍未寫入／普遍未索引」的說法，不能證明記憶語義正確，也不能把提交快照當成最終資料庫狀態。

例如 S7:7 是對照片中狗主人的假設問題，卻產生使用者是狗主人的 asserted fact `amu_653d4b4bfeba4d5e` 及 preference `amu_8a42112205c84575`。前者本次未進最終包，後者進入 Q23。又如 S3:8 是 assistant 主動提供替代音樂活動，`amu_f6e1c475dc304850` 卻摘要成使用者請求替代，並進 Q9／Q14。

這些例子支持修正主體與語氣的語義驗證。提交日誌沒有完整抽取／治理模型原始回應，尚不能把錯誤精確歸到哪一次內部模型呼叫。更不能為提高評測分數，把「詢問」普遍改寫為「擁有／喜歡」。

### 規劃成功格式與 lexical coverage 不等於前提正確

- Q4 把選項建議去做的 `infused loaves or bagel making` 擴成既有 experience 查詢；Q24 的選項 A 查詢出現原選項沒有的 `home bread baking experience`。這是可見的前提漂移，未證明單獨造成錯答。
- Q8、Q16 的 option coverage 會因表面字詞把一般 Facebook／呼吸資訊視為有覆蓋；Q25 有膽固醇自述，對應 option 卻仍顯示缺失。coverage 應是召回提示，不能當成語義支持證明。
- Q9、Q13 規劃超時，回退為 fact；規劃函式自己的 8 秒限制不會因搜尋總期限設為 300 秒而消失。Q13 仍答對，Q9 的遺忘規則仍入包，故規劃超時不是兩題結果的充分解釋。

### 圖有執行，但主要連接來源依賴

本次融合圖共 461 條邊：`requires_source` 187、`derived_from` 187，合計 **81.13%**；`retrieval_bridge` 53；正反 `relation_path` 合計 34，佔 **7.38%**。26 題 graph weight 全為 0.1，24 題 preference、2 題 fact，沒有 multi_hop 分類，followup 全為 `not_needed`。

目前這批圖訊號主要在保存來源依賴與候選關聯；並沒有觀察到針對各選項的個人前提建立完整支持／反證鏈。本資料多為個人化建議，也不是專門的多跳基準。上述結果既不能證明圖融合沒有價值，也不能用「已有圖」推斷跨證據推理已解決。優先增加圖權重可能放大同源重複或錯誤摘要。

### 只有約束也被標成 retrieved

每題保留 6 個遺忘規則是必要的約束保護；它們累計 156/286＝54.55% 的包項目。Q2、Q7、Q18 普通候選全被 listwise 拒絕，最後只剩 6 個規則，仍回報 `retrieved`、未 abstain。Q2／Q7 通用選項恰好答對，Q18 答錯。

建議獨立回報「存在約束」與「存在可回答問題的證據」。不應靠刪除規則來讓指標好看，也不能把 Q2／Q7 答對視為來源反證成功到達。

## 6. 26 題摘要

Gold／預測採原始選項字母。正確表示 benchmark 得分，並不自動表示個人前提有充分來源。

| 題 | 領域 | Gold→預測 | 最主要可觀察情況 |
|---|---|---|---|
| Q0 | Health | A→C | 拉伸、呼吸原文到達；Gold 的每日瑜伽＋冥想前提更強，不能簡稱明確證據被忽略 |
| Q1 | Communication | D→D ✓ | 本人學生事件及尋求學校資源的證據完整到達 |
| Q2 | Health | D→D ✓ | 緊張會議線索被 listwise 拒絕；只剩規則，選通用答案得分 |
| Q3 | Travel | C→B | 本人衝浪故事 CE／fine 保留，12→10 時丟失 |
| Q4 | Food | D→C | 烘焙故事 fine 丟失；原文為 assistant 擬稿，個人歸屬亦弱 |
| Q5 | Health | A→C | 追蹤線索未召回；可見原文未直接支持 Gold 的舊運動傷害／下背痛 |
| Q6 | Outdoors | C→C ✓ | 本人花粉困擾來源完整到達 |
| Q7 | Home | B→B ✓ | 第三人植物故事 fine 丟失；只剩規則仍選對通用答案 |
| Q8 | Communication | D→A | Facebook 家庭用途的 episode fine 丟失，一般 Facebook 知識入包 |
| Q9 | Events | A→B | 忘記電子音樂節偏好的規則完整，答案仍依該偏好作答 |
| Q10 | Fashion | C→A | 服裝故事 fine 丟失，另一奢侈品線索被 listwise 拒絕；個人偏好原本也不明確 |
| Q11 | Motivation | A→D | 第三人學生故事載體 CE 預算失敗後 fine 丟失；無法據包內內容指控模型認錯人 |
| Q12 | Health | B→B ✓ | 第三人母親故事 fine 丟失；仍選對，不能證明反證有被採用 |
| Q13 | Entertainment | D→D ✓ | 動漫興趣的替代來源完整；規劃超時、部分載體丟失未阻止答對 |
| Q14 | Travel | A→B | 漫展 event 未召回，同源 episode 融合第 94、coarse 丟失 |
| Q15 | Wellbeing | B→B ✓ | 本人舊腿傷及騎車限制完整；不宜單用 personal_evidence 布林否定來源 |
| Q16 | Health | A→C | 童年氣喘兩個載體皆 CE 預算失敗→fine 丟失 |
| Q17 | Automotive | B→D | 汽車來源已到達；美國／進口技術詢問未證明偏好實用美製車 |
| Q18 | Outdoors | B→C | 小花園條件句 CE／fine 保留，listwise 拒絕；包內只有規則 |
| Q19 | Decor | B→B ✓ | 照片線索部分到達，含 assistant 示例；得分不等於已證明本人家庭照片收藏 |
| Q20 | Shopping | B→D | LP 音質詢問到達，未證明收藏稀有國際唱片 |
| Q21 | Finance | C→B | 數位資產興趣主要來源與依賴內容到達，仍選一般理財；優先查答案採用 |
| Q22 | Entertainment | C→C ✓ | 舊電影興趣來源到達，支持個人化方向 |
| Q23 | Outdoors | A→D | 湖水溫度／細菌詢問皆到達；不等於已證明游泳喜好 |
| Q24 | Food | B→D | 香料文化詢問皆到達；不等於擁有充足香料架 |
| Q25 | Food | A→D | 膽固醇 profile 與第一人稱原文完整到達，仍選一般零食建議 |

完整逐題原句、AMU、CE 分數與裁剪原因：[Q0–Q12](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-v7-audit/review-q0-12.md)、[Q13–Q25](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-v7-audit/review-q13-25.md)。

與保存的 v4 同題結果比較，分數仍為 9/26；Q2、Q15、Q22 改善，Q3、Q8、Q25 退步。但 Add 的唯一記憶數已從 233 變成 229，並非固定同一記憶庫只切換排序器，故不能把此差異當作純 RRF 對圖融合的 A/B 因果結論。

## 7. 建議修復順序與驗收方式

下列是根據本次證據提出的下一步，尚未實作，也尚未用真實模型重跑驗證分數改善。

| 優先 | 工作 | 驗收標準 |
|---|---|---|
| P0 | 將 CE 預留不足改為可等待／重新准入的調度狀態；結算後重算可用額度，必要時縮批；為後續 listwise 保留額度 | 上述受控時序不再永久遺失第 3 批；仍不超 calls／tokens／deadline；取消後不遺留預留；真正不足有明確理由 |
| P1 | 檢查 coarse／fine／listwise 配額的來源與前提覆蓋，限制無 CE 分數候選擠掉已確認重要證據的方式 | 固定候選快照重播 Q3／Q8／Q14／Q16；同源去重、unknown fallback、前提代表均有明確去留原因；未知分數不得當低分 |
| P1 | 對答案選項檢查已明示的遺忘約束，保存選項支持／反證所引用的來源 ID | 固定 Q9 packet 排除依據被遺忘偏好作答；Q25／Q21 用同一 packet 驗證已存在的支持是否被正確採用 |
| P1 | 語義抽取保留主體、條件、時間、引用者與支持強度；規劃的選項前提不得新增事實 | 狗主人假設不變成本人身份；assistant 改寫不變成本人自述；舊氣喘不變成目前發病；建議活動不變成既有經驗 |
| P2 | 分開約束數／回答證據數與狀態；coverage 明確維持啟發式角色 | Q2／Q7／Q18 不再以只有規則的包表示有回答證據；Q16／Q25 的真假覆蓋可被診斷 |
| P2 | 固定記憶快照、查詢與答案模型做圖訊號及級聯消融；另列弱支持 Gold | 分開測來源到達率、CE 評分覆蓋、約束違反率、答案採用率與最終得分，不靠虛構個人偏好提高分數 |

尤其不建議先增加圖權重、提高最終 packet 上限或一律放大搜尋 timeout。這三項都未對準本次最明確的預算調度缺陷，而且 Q9／Q25 的必要資訊已經在答案輸入中。

驗證界線：此報告以原始日誌、不可變來源對齊與受控 HTTP mock 重現為依據；沒有重新建立記憶庫、沒有重新呼叫 CE／LLM、沒有修改 Gold。可證明缺陷及證據去留，不能保證修正後增加幾個百分點。單一對話的 26 題、每類 1–5 題，也不足以把各領域百分比解釋成穩定的領域能力差異。
