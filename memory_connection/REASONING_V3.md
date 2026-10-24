# 日期、多跳前提與雙側橋接優化

**歷史實驗版本，已撤回。** 未測會話結果顯示退步，見 [失敗報告](REASONING_V3_REPORT.md)；目前實作與最後一輪設計見 [REASONING_V4.md](REASONING_V4.md)。

本輪保持 `memory_connection/app` 與 `memory_system` 獨立；後者只提供既有資料集與共同 Answer／Judge。修改前的 v2 程式與提示詞已先保存為不可覆寫的比較快照。

## 實作

1. `temporal.py` 以來源自己的 Unix 毫秒時間產生可讀 UTC 日期，並為有限的英文相對日期提供明確的日／區間註解：today、yesterday、tomorrow、last weekday、last weekend、last week。保留原詞位置與使用的日曆慣例；缺少時間或週末陳述的 last weekend 不猜測精確日期。這些是時間註解，不是新記憶事實。其他相對時間仍由模型依可讀來源日期理解。
2. `references.py` 在原文精判內提供可讀日期與註解。新提示詞區分「某天提到」與「某天發生」，禁止以現在、另一則消息或儲存版本時間代替原文陳述時間。返回事件另外帶 `source_annotations`，原始 `content`、`sources`、引文與位置不改寫。
3. 精判先輸出同批原文的 `premise_links`，再逐事件裁決；不同批次共用最多 4,000 bytes 的完整原始來源作閱讀上下文。跨事件連接只是推理輔助，不會覆蓋負面裁決，也不能作為引用。每個有效裁決仍必須引用自身允許的原文來源。
4. 重排改為對去重後的每個原始來源評分，事件取其來源的最高分，再執行來源多樣性排序。這避免無關的最後一句蓋過上下文中的姓名、地名或日期。模型不讀聯想視角作為事實。
5. `reserve_bridges` 在既有 16 個名額內最多保留 2 個橋接候選：觀測路徑須全部是雙側衍生連接，最佳路徑名次不超過 8，且帶來未覆蓋來源。有有效語義重排時只考慮排序前 32 名；重排降級時不把粗排名次冒充語義名次。保護前半部及已有雙側路徑，並不擴大精判池。這是路徑代理，不能當作已證明該事件只會出現在 C11。
6. 精判明確標記 `answer_role`：specific、bridge、background 或 unspecified。最終封裝優先具體屬性及必要前提，保留同類內的排序；不再讓一般描述與具體細節一律同權。partial 不因標籤本身被降級。反證保留及原文引用校驗維持。

新配置：`MC_BRIDGE_SLOTS`（預設 2，設 0 關閉保留）、`MC_PREMISE_CONTEXT_BYTES`（預設 4000，設 0 關閉跨批次上下文）。服務仍只使用 `MC_` 配置。

## 開發校準與已知限制

只使用上一輪已測會話 conv-41、conv-47、conv-26 的七個案例迭代。日期提及／事件時間、相對日期換算及「故鄉＝國名」的前提連接取得修復。城堡夢境的原始證據已進入精判並被返回，但共同 Answer 模型仍可能回答概括性的「生動的夢」。所有開發嘗試（包括失敗、重排超時及不同提示詞版本）保存在 `development.jsonl` 和用量紀錄中；不計入正式成績。

81 項測試通過，新增覆蓋相對日期邊界、缺失與歧義、原文及引文位置不變、來源級重排、跨批上下文的大小與來源、橋接名額上限、重排降級，以及具體 partial 證據的封裝優先順序。

## 凍結的未測會話實驗

Run：`runs/optimization-v3-holdout-20261024`。排除所有前兩輪使用過的五個會話：conv-26、conv-41、conv-43、conv-47、conv-50。固定種子 20261025，抽到：

| 會話 | 完整消息數 | 抽樣題數 |
|---|---:|---:|
| conv-48 | 681 | 12 |
| conv-49 | 509 | 12 |
| conv-30 | 369 | 12 |
| 合計 | 1,559 | 36 |

v2／v3 各跑 C00、C10、C01、C11，合計 288 筆。索引與預生成查詢視角／向量完全共用，兩版本的粗召回序列必須相同。top_k=10、證據 24,000 bytes、精判上限 16，搜尋／重排／精判預算仍為 120／20／80 秒，模型與上一輪相同。新版逐來源重排可能增加重排請求數，費用及延遲會如實記錄。

共同評測 adapter 現在會把原有 `evidence_spans` 放在證據物件前端，**兩個版本都使用相同格式**；新版另外提供時間註解。Answer／Judge 提示詞及 `memory_system` 程式不改動。因此本輪的 before 成績必須與本輪 after 比較，不能直接和上一輪不同會話、不同證據包格式的總分相比。

共同查詢預處理採每階段最多三次有界重試，失敗嘗試的成本及耗時保留。所有已完成查詢視角都保存並重放；不重跑已完成得分。核心程式、提示詞、評測入口、抽樣和資料雜湊已凍結，正式測試期間不據成績調參。

## 重現

```powershell
python -m unittest discover -s memory_connection/tests -q
python -m memory_connection.scripts.reasoning_benchmark --phase plan
python -u -m memory_connection.scripts.reasoning_benchmark --phase all
python -m memory_connection.scripts.analyze_optimization memory_connection/runs/optimization-v3-holdout-20261024
```

同一 run 可冪等續跑並跳過既有結果；不要改動凍結程式後覆寫該 run。時間註解會接受獨立重算校驗，返回的所有原文與引用仍逐條檢查。最終需同時看配對準確率、四階段來源召回、C11 相對兩種單側擴展的得分，以及保留槽實際救回多少標註來源；三個會話仍不足以聲稱全量泛化。
