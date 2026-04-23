# Stage 1 偵察報告 (recon.md)

## 專案一句話摘要

OpenMythos 是 Kye Gomez 撰寫的開源 PyTorch 函式庫，對 Anthropic Claude Mythos 模型的疑似架構進行理論性重建，實作了 Recurrent-Depth Transformer (RDT) + Mixture-of-Experts (MoE) + Multi-Latent Attention (MLA)。

> ⚠️ **重要免責聲明**：這是社群驅動的推測性重建，不代表 Anthropic 的任何官方實作或洩漏。

---

## 技術棧

| 類別 | 技術 | 版本 |
|------|------|------|
| Language | Python | ≥3.10 |
| DL Framework | PyTorch | 2.11.0 |
| Tokenizer | Hugging Face Transformers | ≥4.40.0 |
| Dataset loading | Hugging Face Datasets | ≥2.18.0 |
| Optional attention | flash-attn | ≥2.8.3 (optional) |
| Package manager | Poetry | — |
| Test framework | pytest | ≥8.1.1 |
| Linter | ruff | ≥0.5.1 |
| Formatter | black | ≥23.1 |
| Training parallelism | PyTorch FSDP (FullyShardedDataParallel) | — |
| Training logging | loguru | — |

---

## 目錄結構（3 層深）

```
OpenMythos/
├── open_mythos/             # 主要 Python 套件
│   ├── __init__.py          # 公開 API 匯出
│   ├── main.py              # 核心模型（所有主要 class）
│   ├── moda.py              # 替代架構：MoDA + DeepSeek MoE
│   ├── tokenizer.py         # HuggingFace tokenizer wrapper
│   └── variants.py          # 預設 1B→1T 規格設定
├── training/
│   ├── 3b_fine_web_edu.py   # 3B 模型 FineWeb-Edu 預訓練腳本
│   └── requirements.txt     # 訓練專用依賴
├── tests/
│   ├── __init__.py
│   ├── test_main.py         # 主要單元測試（含 RoPE、注意力、ACT 等）
│   ├── test_rope_debug.py   # RoPE 除錯測試
│   ├── test_tokenizer.py    # tokenizer 測試
│   ├── bench_vs_transformer.py # 對比基準測試
│   └── small_benchmark.py   # 小型性能基準
├── examples/
│   ├── moda_example.py      # MoDA 模型使用範例
│   └── variants_example.py  # 模型規格使用範例
├── docs/
│   ├── open_mythos.md       # OpenMythos class 完整 API 參考
│   └── datasets.md          # 訓練資料集推薦
├── README.md                # 專案主要說明（含架構理論說明）
├── pyproject.toml           # Poetry 套件設定
├── requirements.txt         # 基本依賴
├── example.py               # 頂層快速使用範例
└── LICENSE                  # MIT
```

**架構模式**：Pure Library（無 web server、無 CLI）+ 獨立訓練腳本。

---

## 既有文件掃描

### `docs/open_mythos.md`
- **來源**：`docs/open_mythos.md`
- **內容**：完整的 `OpenMythos` class API 參考，包含 constructor、`forward`、`generate`、所有子模組說明、`MythosConfig` 欄位表格
- **品質**：高品質，與程式碼高度一致

### `docs/datasets.md`
- **來源**：`docs/datasets.md`
- **內容**：訓練資料集推薦（FineWeb-Edu、OpenHermes、OpenWebMath）與 token budget 建議

### `README.md`
- **內容**：安裝說明、使用範例、架構理論解說、7 個模型規格表格、訓練指令、參考文獻列表

---

## 既有文件 vs 程式碼落差分析

| 序號 | 落差描述 |
|------|----------|
| 1 | `README.md` 的 `forward` 函式簽名寫 `def forward(self, input_ids, n_loops, kv_cache)`（無 `start_pos`），但 `main.py:992` 實際有 `start_pos: int = 0` 參數 |
| 2 | `docs/open_mythos.md` 的 `forward` 參數表同樣缺少 `start_pos`；這是 autoregressive decode 的關鍵參數 |
| 3 | `README.md` 提到 `mythos_7b()` 變體（README line 124），但 `variants.py` 中並不存在 `mythos_7b` 函式；跳過 7B 直接是 10B |
| 4 | `__init__.py:52` 匯出 `load_tokenizer`、`get_vocab_size`，但這兩個函式在任何模組中均未定義（⚠️ 損壞的匯出） |
| 5 | `moda.py` 是功能完整的替代模型（MoDAModel），但 `__init__.py` 未匯出它；只能透過 `from open_mythos.moda import MoDAModel` 存取 |
| 6 | `training/3b_fine_web_edu.py` 的 docstring 說使用 DDP，但實際用的是 FSDP（FullyShardedDataParallel），更強大但設定不同 |

---

## 架構識別

- **專案類型**：Pure Python library（ML research）
- **並行模式**：無（library 不運行 server）；訓練腳本支援 FSDP 多 GPU
- **無** Dockerfile、docker-compose、CI/CD、.env 檔案
- **有** pyproject.toml（Poetry）、requirements.txt（簡化版）
