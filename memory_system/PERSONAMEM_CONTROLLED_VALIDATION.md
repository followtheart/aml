已完成三組受控驗證：**固定輸入後，答案端的錯誤仍可重現；Q22 可定位為 listwise 對候選組合的敏感性；Q1 則存在確定性的改寫漏判與否定誤判。** 這次沒有重新抽取記憶、重新召回或呼叫 Cross-Encoder，也沒有修改 production 邏輯。

實驗使用上一輪 `personamem-v9-rerun-audit/inputs.zip`，題號沿用資料集的零起始 QA ID。模型為 `openai/qwen3-14b`、temperature 0、關閉 thinking。共 **48 次實際 provider 呼叫、190,799 個 provider 回報 tokens**。每個條件重複 3 次；3 次不是統計顯著性證明，也不是新的全量 benchmark 分數。

**固定答案包：同一 prompt 的答案沒有翻轉。**

五題 baseline 使用 production `eval_scoring.answer_prompt`，與先前封存重建的 SHA-256 逐字一致。另一個 diagnostic 條件保留全部證據文字與順序，增加 item ID，要求每個選項列出個人前提、來源原文、來源角色、敘述主體、約束及最終選擇。它改變了輸出指令、system、工具格式與輸出上限，因此是獨立條件，不能與 baseline 混稱同一 prompt。Gold 只供回應後評分，未送給模型。

| 題目 | Gold | 原始 prompt，3 次 | 逐選項 diagnostic，3 次 | 可支持的判斷 |
|---|---|---|---|---|
| Q3 | C | B / B / B | B / B / B | 衝浪原文已在答案包，仍選泛化建議 |
| Q9 | A | B / B / B | B / B / B | 忘記約束已送達，最終選擇仍與約束衝突 |
| Q11 | A | D / D / D | D / D / D | 第三人 Marcus 的敘述被當成個人經歷 |
| Q12 | B | B / B / B | B / B / B | 本輪固定包的正確答案穩定 |
| Q15 | B | B / B / B | B / B / B | 本輪固定包的正確答案穩定，但引用品質有波動 |

這組對照能排除「本次重試又抽取出不同記憶」的影響。它不能單獨解釋 Q12／Q15 在兩次歷史全管線執行之間的翻轉，因為這次沒有重放舊答案包。

逐選項輸出提供了更具體的失敗證據，但不是模型內部因果過程的證明：

- **Q3：無關約束被擴大套用。** S8:7 的使用者原文包含 `waxing my board`、`catching those early, still-glass waves`，已出現在答案包。三次 diagnostic 都以「忘記家庭旅遊拍照」等六條規則否決衝浪選項 C；這些規則沒有撤回衝浪經驗。引文是真的，約束與選項的關聯卻錯了。選項 A 的說明還混入了 B 的一般海濱活動，顯示選項與判斷也可能錯配。
- **Q9：約束判斷與最後答案不一致。** 三次都將 B 標為 `constraint.applies=true`，說明它違反電子音樂節的忘記約束，最後卻仍回傳 B。模型也把一般活動建議 A 標為受該約束限制；因此，僅以它產生的布林值做硬排除仍不可靠，必須驗證約束到底限制哪個前提。
- **Q11：來源角色與敘述主體同時錯置。** 引用的 `Last spring, Marcus was mentoring a senior...` 位於 S0:2 的 **assistant** 訊息，內容明確描述 Marcus。三次 diagnostic 都把它標為 `source_role=user`、`subject=current_user`，並據此支持 B／C／D。這裡「訊息存在」不等於「使用者經歷成立」；選項的第一人稱敘述也沒有因此獲得合法主體。
- **Q12：答對不等於已證明主體解析正確。** 三次選 B；模型主要表示 remote schooling 的個人前提缺乏支持，並未穩定輸出「Claire 是第三人」的直接核對。三次還都漏掉各選項要求的 assessment 欄位，改成頂層 assessment。
- **Q15：答對與引用可靠性可以分離。** 第一次正確引用 S7:1 的騎車與舊腿傷原文；第二次只引用衍生記憶摘要，第三次混合原文與摘要。後兩次還改寫了要求逐字擷取的選項前提。三次答案都是 B，完整引文檢查只有第一次通過。

| Diagnostic 題目 | JSON Schema 合格 | 選項前提、原文與角色字面核對合格 | 解讀 |
|---|---:|---:|---|
| Q3 | 0/3 | 3/3 | citations 超過上限；即使引文存在，約束仍被錯套 |
| Q9 | 3/3 | 3/3 | 三次都選了自己標為受約束限制的 B |
| Q11 | 3/3 | 0/3 | assistant 原文被冒標為 user |
| Q12 | 0/3 | 3/3 | 缺少必要欄位，不能算完整結構化診斷 |
| Q15 | 3/3 | 1/3 | 後兩次引用或逐字前提不合格 |
| 合計 | 9/15 | 10/15 | **同時通過兩項僅 4/15；不代表語義正確率** |

所以「要求模型提供理由」本身沒有修復答案端。下一步應驗證四個獨立條件：引文可追溯、主體對得上、約束適用於該前提、最後選擇符合已驗證的判定。不能把一份可解析 JSON 視為這四件事都成立。

**固定 fine 清單：Q16 與 Q22 的行為不同。**

共同候選的文字、CE 分數與順序均固定。原生 listwise 的 10 筆輸出上限是 256 tokens、12 筆是 288；審查發現這會混入輸出預算差異，因此另外補了「10 筆、288 tokens」各 3 次。下表的 10／12 比較統一使用 288 tokens；原生 256 的結果另列作參考。

| 題目與條件 | prompt bytes | 原文在輸入中 | 3 次輸出結果 |
|---|---:|---|---|
| Q16：10 筆，256 tokens | 22,537 | S6:0 全文 | 3/3 保留，每次留下 9 筆 |
| Q16：10 筆，288 tokens | 22,537 | S6:0 全文 | 3/3 保留，每次留下 9 筆 |
| Q16：12 筆，288 tokens | 28,051 | S6:0 全文 | 3/3 結構無效，不能計算合法保留率 |
| Q22：10 筆，256 tokens | 17,366 | S3:17、S7:3、S7:5 全文 | 3/3 將全部候選列為 irrelevant |
| Q22：10 筆，288 tokens | 17,366 | 同上 | 3/3 將全部候選列為 irrelevant |
| Q22：12 筆，288 tokens | 23,986 | 同上，加上 episode 承載方式 | 3/3 留下 6 筆；三段目標原文均保留 |

Q16 的 S6:0 同時包含兒時 mild asthma 與 `hasn’t really bothered me for a long time`，是帶有限定條件的歷史資訊。固定本輪 10 筆後，它在兩種輸出上限下都被保留。因此，**這次沒有重現本輪清單的病史誤刪，也不能據此聲稱歷史翻轉的原因已修復**。

Q16 強制送入 12 筆時，三次都把 12 個索引全部放進同一個 group，違反每組 2–4 筆的結構要求。該 prompt 也超過 production 的 24,000 bytes 上限。這是旁路容量探測，沒有執行 production 的 repair／fallback；不應把它解讀成生產環境必然清空證據，也不應當成提高 limit 就能部署的結果。

Q22 的關鍵區別是：**10 筆時原文已經存在，消失發生於 listwise 的 irrelevant 判定。**

| 原文 | 10 筆中已有的承載候選 | full12 增加的 episode |
|---|---|---|
| S3:17，99 字元 | `amu_8434425e8e9d44c4` | fine 第 11 筆 `amu_dd31a99c6fcc408b` |
| S7:3，131 字元 | `amu_481ad20e6d7049aa` | fine 第 12 筆 `amu_c22d627181b34001` |
| S7:5，109 字元 | `amu_89bbcf07dda641a6` | 同一個第 12 筆 episode |

在固定輸出上限後，加入兩個 episode 仍使三段原文從全刪變成全保留，並改變了共同候選的保留決策。這支持「listwise 對候選組合、敘事與上下文呈現敏感」的診斷；**不能把差異全歸因於名額數字，也不能稱為補回原本未召回的原文**。新增 episode 同時帶來敘事、鄰接來源與關係資訊，本實驗沒有再將這些因素拆開。

Q22 的 full12 距 24,000 bytes 上限只剩 14 bytes，單改 LLM_LIMIT=12 對這個案例有用，但不是穩健的普遍修復。較值得後續驗證的是：在既有預算內，以承載完整上下文的 episode 取代重複摘要；對「所有同主題候選都被標成 irrelevant」做有依據的復核。兩者都需要保留主體、假設與否定檢查，不能只因主題相同就強留。本次只測到 listwise 輸出，沒有重新產生 Q22 最終答案，不能宣稱答題分數已提升。

**Q1 無模型語義回歸：7/10 符合期望，語義 gate 仍為 RED。**

六個正例使用完全相同的 S1:12 原文、相同 `_rank_text`；全部正負例的訊息角色都是 user。真實入口是未修改的 `search_coverage.annotate`，沒有呼叫模型。人工期望針對「學生向使用者透露困擾」的事件層級支持，不代表整個回答或所有選項敘述都獲得證明。

| 案例 | 期望 | 實測 |
|---|---|---|
| 舊 planner：`experience with being present for students sharing troubling matters` | 支持 | 支持 |
| 本輪 planner：`ways to be present without having all the answers when a student shares something troubling` | 支持同一事件 | 不支持 |
| 同事件的 plain／synonyms／disclosed 三種改寫 | 支持 | 都支持 |
| `listening to a student sharing deeply troubling personal circumstances` | 支持同一事件 | 不支持 |
| 第三人 Karen 引文 | 不支持使用者本人經歷 | 正確拒絕 |
| 明確虛構／假設場景 | 不支持實際經歷 | 正確拒絕 |
| `I have never been present...` | 不支持 | 正確拒絕 |
| `No student ever shared troubling personal circumstances with me.` | 不支持 | **錯誤支持** |

本輪 planner 改寫只匹配 2/6 個概念，先被 lexical gate 擋住；獨立重放 scope 判斷時，`without having all the answers` 也被當成與來源不一致的否定。listening 改寫只有 3/5 個概念，同樣沒有過門檻。相反地，`No student...with me` 匹配 3/3，`me` 觸發本人指涉，而現有否定識別沒有涵蓋這個 `No` 量詞範圍，產生假陽性。

後續修復應先分清「事件沒有發生」與「不知道如何回應事件」的否定範圍，再處理同義前提的概念匹配；單純降低 overlap 門檻有放大假陽性的風險。第三人、假設、`never`、`No student` 四個負例需作為同等重要的回歸條件。

**實驗與驗證產物已保存。**

- [凍結輸入、完整 prompts、SHA-256、候選與 oracle](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-controlled-validation/fixtures.json)
- [原始 42 次呼叫與回應](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-controlled-validation/calls.jsonl)；[固定 288 上限的 6 次補充呼叫](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-controlled-validation/matched-cap/calls.jsonl)
- [後驗核對：格式、字面引文、原文保留、各條件 hash](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-controlled-validation/audited-results.json)
- [Q1 所有案例、witness、概念與 scope gate](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-controlled-validation/q1-paraphrases.json)
- [原始輸入及 53 個 production 檔案指紋核對](C:/Users/aapoo/Desktop/aml/memory_system/runs/personamem-controlled-validation/integrity-checks.json)

同一條件下 prompt 與 request hash 均固定；完整 hash 在 fixture 及後驗結果中。歷史 prompt 是本地重建，並非當時原始 HTTP 擷取。執行使用單次調用、不自動修復或重試，避免改 prompt 後混入同一條件；中斷後已有 started 紀錄但無結果的呼叫不會自動重送。

新增工具為 [controlled_validation.py](C:/Users/aapoo/Desktop/aml/memory_system/scripts/controlled_validation.py)、[audit_controlled_validation.py](C:/Users/aapoo/Desktop/aml/memory_system/scripts/audit_controlled_validation.py)、[probe_q1_paraphrases.py](C:/Users/aapoo/Desktop/aml/memory_system/scripts/probe_q1_paraphrases.py) 與離線 selftest。這些是診斷工具；原先的 `run_experiment.ps1` 使用者修改已保留。

最新驗證：診斷工具 **13/13 selftests 通過**；Q1 `--strict` 按預期 exit 1，明列三個不符合語義期望的案例。原始兩份 debug 日誌、results、prepared data 的 SHA 均未改變；53 個受追蹤的 production 程式指紋均與封存執行一致。此次是定位問題與建立可重跑回歸，沒有宣稱 pipeline 已修復。

只重算已有結果，不會呼叫模型：

```powershell
memory_system/.venv/Scripts/python.exe memory_system/scripts/audit_controlled_validation.py audit
memory_system/.venv/Scripts/python.exe memory_system/scripts/probe_q1_paraphrases.py --strict
memory_system/.venv/Scripts/python.exe memory_system/scripts/run_selftests.py selftest_controlled_validation.py
```

在新的輸出目錄重做同一套 provider 實驗：

```powershell
memory_system/.venv/Scripts/python.exe memory_system/scripts/controlled_validation.py prepare --output memory_system/runs/controlled-new
memory_system/.venv/Scripts/python.exe memory_system/scripts/controlled_validation.py run --output memory_system/runs/controlled-new
memory_system/.venv/Scripts/python.exe memory_system/scripts/audit_controlled_validation.py prepare-matched --output memory_system/runs/controlled-new
memory_system/.venv/Scripts/python.exe memory_system/scripts/controlled_validation.py run --output memory_system/runs/controlled-new/matched-cap
memory_system/.venv/Scripts/python.exe memory_system/scripts/audit_controlled_validation.py audit --output memory_system/runs/controlled-new
```
