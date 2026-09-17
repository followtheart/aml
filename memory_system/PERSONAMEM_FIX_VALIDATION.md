# PersonaMem 入包與答題修正驗證

日期：2026-09-17。依據 [瓶頸分析](data/results/personamem-v2-32k.bottlenecks.md) 修正搜尋裝包、選項對齊與回放工具。

本次已完成程式修正及離線驗證；未呼叫真實模型重跑答案，因此原有 **8/26** 尚未被新的答對率取代。原「記住 23 → 召回 23 → 入包 15 → 答對 8」使用弱線索口徑，不視為嚴格包含的漏斗。

## 已完成的修正

| 瓶頸 | 修正行為 |
|---|---|
| 個人化視圖直接排除 fact、plan | 按使用者歸屬及來源判斷，保留合格的 fact、plan、event、episode；一般選擇題不因帶有 options 就套用個人化篩選。 |
| 相同核心画像固定前置，占去過半容量 | 召回的画像遵循當題排名；額外補入的画像按相關性選取，預算上限為 `min(CORE_PROFILE_TOKEN_BUDGET, 總預算 / 5)`，預設上限 1600。 |
| unknown 時間前綴及冗長來源消耗容量 | 正文省略無意義時間前綴；來源採原文連續片段，保留位置與追溯資料；必要引文優先於來源數量限制，包含必要 assistant 解釋及較晚的 user 訊息。文件型任務仍保留完整來源。 |
| 來源去重導致後續記憶失去對齊資格 | 在壓縮與去重前判斷個人證據資格，隨包傳遞；雜湊 v2 覆蓋正文、記憶類型、來源及資格欄位。既有 v1 包仍可讀取。 |
| 對齊只看三種記憶類型，且不看來源 | 對齊讀取已驗證包內所有合格使用者證據及實際可見來源；不再全面屏蔽地點、職業，而要求引用與具體選項前提相符。 |
| 捏造引用、錯把 generic 當無支持 | 契約要求 `option_claim`、`evidence_id`、原文 `evidence`、`unsupported_claims`；程式檢查選項片段及證據原文。無效引用降級並留下原因；有有效支持時不再因 generic 標籤清掉支持。 |
| 同分依字母取勝、弱線索壓過通用答案 | 明示同分及全無支持狀態；不要求 weak 一律優於通用選項。自動選答預設關閉，開啟時也僅接受唯一、有效且無缺失前提的 strong 支持。遺忘限制須附約束引用。 |
| 無法區分版本、裝包遺失與對齊錯誤 | trace/result 補入預算、最後排名、排除原因、覆蓋清單、`search_id`、`packet_hash`、程式／提示版本及模型設定；對齊另記輸入證據 ID 和引用校驗結果。 |

`evidence_token_budget` 沿用既有欄位名稱，計價單位仍是 **UTF-8 bytes**，不是模型 tokenizer 的 token 數。壓縮在包驗證及雜湊生成之前完成，驗證與答題使用相同正文。

## 離線驗證結果

- 全部 **15/15 個自測套件通過，合計 229 項測試／檢查**，包含 27 項 PersonaMem 瓶頸回歸測試。
- 覆蓋必要來源保留、一般知識不冒充個人事實、具名使用者摘要、來源去重、包欄位篡改、偽造引用、選項同分，以及 API 序列化後包內容一致性。
- 原始 26 題及兩組重裝包各 26 題，均通過按使用者、問題、選項配對及包雜湊檢查；排除自測 trace，拒絕含糊重複配對。
- `compileall` 與 `git diff --check` 通過；獨立程式複核通過。
- 上述驗證與裝包回放的模型供應商呼叫數為 **0**。

保持日誌記錄的候選排名固定，使用目前正式裝包函式重放：

| 方案 | 題數 | 平均入包條數 | 平均正文 bytes |
|---|---:|---:|---:|
| 原始 12k 包 | 26 | 19.81 | 11937.27 |
| 修正後，仍為 12k | 26 | 27.12 | 11968.85 |
| 修正後，32k | 26 | 50.00 | 24148.54 |

同為 12k 時，平均條數增加約 **36.9%**。32k 組達到本輪 `top_k=50`；此數字是容量利用指標，不能等同正確證據覆蓋或答對率。

產物：

- [12k 逐題重裝包](data/results/personamem-v2-32k.repacked-12k-v2.jsonl)及[摘要](data/results/personamem-v2-32k.repacked-12k-v2.summary.json)。
- [32k 逐題重裝包](data/results/personamem-v2-32k.repacked-32k-v2.jsonl)及[摘要](data/results/personamem-v2-32k.repacked-32k-v2.summary.json)。

回放以日誌中的記憶快照聯集及可觀察的依賴重建來源，並非完整歷史資料庫。新增視圖可接納的 ID 另行列出，未替它們捏造新重排分數。這兩組產物標記為 `offline_repacked_not_verified`，尚未經新的模型充分性驗證，也未重新搜尋或產生答案。來源歸屬及引用存在性檢查不能證明語義蘊含；地點誤配興趣等問題仍須用真實模型回放檢驗。

## 重現命令

以下 PowerShell 命令均在專案根目錄執行，使用 `memory_system` 自己的虛擬環境。

```powershell
# 隔離的離線自測，不讀寫正式資料庫或呼叫模型
& memory_system\.venv\Scripts\python.exe memory_system\scripts\run_selftests.py

# 僅檢查原始記錄配對與雜湊，不呼叫模型
& memory_system\.venv\Scripts\python.exe memory_system\scripts\validate_choice_alignment.py --traces memory_system\logs\search-debug.jsonl --data memory_system\data\prepared\personamem-v2-32k.jsonl --inspect

# 固定舊排名重裝包；輸出檔必須尚不存在，原始日誌與結果不會被覆寫
& memory_system\.venv\Scripts\python.exe memory_system\scripts\replay_evidence_packets.py --memory-log memory_system\logs\memory-debug.jsonl --traces memory_system\logs\search-debug.jsonl --budget 12000 --output memory_system\data\results\personamem-repack-12k-new.jsonl

# 檢查實驗性重裝包，不呼叫模型
& memory_system\.venv\Scripts\python.exe memory_system\scripts\validate_choice_alignment.py --traces memory_system\data\results\personamem-repack-12k-new.jsonl --data memory_system\data\prepared\personamem-v2-32k.jsonl --allow-repacked --inspect
```

將 `--budget` 改為 `32000` 並使用另一個新輸出名稱，可重現第二組裝包比較。

## 真實模型驗收入口

以下命令**會呼叫目前設定的模型**，本次未執行。須使用真實模式及既有有效供應商設定；工具不會重新 Add 歷史，也不會修改原始結果。先固定原始包，可單獨檢驗對齊與答題修正：

```powershell
& memory_system\.venv\Scripts\python.exe memory_system\scripts\validate_choice_alignment.py --traces memory_system\logs\search-debug.jsonl --data memory_system\data\prepared\personamem-v2-32k.jsonl --answer --output memory_system\data\results\personamem-alignment-answer-new.jsonl
```

省略 `--answer` 只重跑對齊，仍會呼叫模型。要比較組合修正，可改用重裝包路徑並加入 `--allow-repacked --answer`，但必須保留其「尚未重新驗證充分性」的實驗標記。完整搜尋收益則需要在相同記憶庫上重新 Search。

下一輪應按題分別核對：直接支持、弱話題線索、矛盾／遺忘約束、只存在於來源正文的證據；再比較視圖、排名、入包、對齊可見性、有效引用及答案。正向個人化題與通用／遺忘題須分組評估，不能用入包條數代替答對率。
