# OpenMythos 專案總覽與速查

## 專案一句話摘要

OpenMythos 是 Kye Gomez 基於公開研究文獻，以 PyTorch 理論重建的 Claude Mythos 疑似架構：**Recurrent-Depth Transformer (RDT)** + **Mixture-of-Experts (MoE)** + **Multi-Latent Attention (MLA)**，提供從 1B 到 1T 參數的預設規格，並附訓練腳本與替代架構（MoDA）。

> ⚠️ 這是推測性社群重建，非 Anthropic 官方實作或洩漏。

---

## 技術棧總覽

| 類別 | 技術 | 版本 | 用途 |
|------|------|------|------|
| Runtime | Python | ≥3.10 | 語言環境 |
| DL Framework | PyTorch | 2.11.0 | 張量運算、autograd、分散式訓練 |
| Tokenizer | Hugging Face Transformers | ≥4.40.0 | tokenizer 載入 |
| 資料集 | Hugging Face Datasets | ≥2.18.0 | FineWeb-Edu 串流載入 |
| 選用加速 | flash-attn | ≥2.8.3 | Flash Attention 2（需 CUDA） |
| 套件管理 | Poetry | — | 依賴管理與發布 |
| 訓練平行化 | PyTorch FSDP | — | 多 GPU 訓練（FullyShardedDataParallel） |
| 測試 | pytest | ≥8.1.1 | 單元測試 |
| Lint | ruff | ≥0.5.1 | 風格檢查 |
| 格式化 | black | ≥23.1 | 程式碼格式 |

---

## 關鍵指令速查

```bash
# 安裝
pip install open-mythos
pip install open-mythos[flash]   # 含 Flash Attention 2

# 開發安裝
poetry install

# 執行測試
pytest tests/

# 程式碼品質
ruff check open_mythos/
black --check open_mythos/

# 單 GPU 訓練（3B 模型，FineWeb-Edu）
python training/3b_fine_web_edu.py

# 多 GPU 訓練
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    training/3b_fine_web_edu.py
```

---

## 基本使用

```python
import torch
from open_mythos import OpenMythos, MythosConfig, mythos_3b

# 方式 1：自定義小型設定（快速實驗）
cfg = MythosConfig(
    vocab_size=32000, dim=256, n_heads=8, n_kv_heads=4,
    max_seq_len=128, max_loop_iters=4, attn_type="mla",
    n_experts=8, n_shared_experts=1, n_experts_per_tok=2,
    expert_dim=64, lora_rank=8,
    kv_lora_rank=32, q_lora_rank=64,
    qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=16,
)
model = OpenMythos(cfg)

# 方式 2：使用預設規格
cfg = mythos_3b()
model = OpenMythos(cfg)

# 前向傳播
ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(ids, n_loops=4)     # (2, 16, vocab_size)

# 生成
out = model.generate(ids, max_new_tokens=32, n_loops=8)  # (2, 48)

# 驗證穩定性（spectral radius < 1）
A = model.recurrent.injection.get_A()
rho = torch.linalg.eigvals(A).abs().max().item()
print(f"ρ(A) = {rho:.4f}  (must be < 1)")
```

---

## 文件地圖

| 文件 | 說明 |
|------|------|
| [`INDEX.md`](.trace/INDEX.md) | 本文件：專案總覽與速查 |
| [`CODEBASE_MAP.md`](.trace/CODEBASE_MAP.md) | 程式碼地圖：目錄說明 + 「我想改 X 看哪裡？」 |
| [`ARCHITECTURE.md`](.trace/ARCHITECTURE.md) | 系統架構：Mermaid 圖、元件清單、設計決策 |
| [`DATA_MODEL.md`](.trace/DATA_MODEL.md) | 資料模型：Tensor 形狀、Config 欄位、KV cache 結構 |
| [`API_SURFACE.md`](.trace/API_SURFACE.md) | API 參考：class 方法、函式簽名、使用範例 |
| [`DEV_GUIDE.md`](.trace/DEV_GUIDE.md) | 開發者上手：環境建置、測試、除錯技巧 |
| [`DISCOVERY_LOG.md`](.trace/DISCOVERY_LOG.md) | 探索紀錄：落差、TODO、待調查問題 |

---

## 原始文件

| 文件 | 說明 |
|------|------|
| [`docs/open_mythos.md`](docs/open_mythos.md) | `OpenMythos` class 完整 API 參考（官方） |
| [`docs/datasets.md`](docs/datasets.md) | 訓練資料集推薦與 token budget |
| [`README.md`](README.md) | 安裝、架構理論、模型規格表格、參考文獻 |

---

## 專案術語表

| 術語 | 說明 |
|------|------|
| **RDT** | Recurrent-Depth Transformer：相同 weights 跑 T 次 loop 的 transformer |
| **Prelude** | 前置標準 transformer blocks（跑一次） |
| **Recurrent Block** | 被 loop 執行 T 次的單一 transformer block（核心） |
| **Coda** | 尾部標準 transformer blocks（跑一次） |
| **LTI Injection** | Linear Time-Invariant 注入：h_{t+1} = A·h_t + B·e + Trans(h_t, e) |
| **ACT** | Adaptive Computation Time：每個位置自動決定何時停止 loop |
| **LoRA Adapter** | 深度適應器：per-loop 的低秩矩陣微調（非訓練用 LoRA，而是架構元件） |
| **MLA** | Multi-Latent Attention（DeepSeek-V2 風格）：壓縮 KV cache |
| **GQA** | Grouped Query Attention：共享 KV heads |
| **MoE** | Mixture of Experts：稀疏路由的 FFN 集合 |
| **Routed Expert** | 由 router top-K 選中的專家 FFN |
| **Shared Expert** | 每個 token 必定通過的共享 FFN |
| **MoDA** | Mixture-of-Depths Attention：跨層 depth KV cache 的替代注意力機制 |
| **FSDP** | FullyShardedDataParallel：PyTorch 多 GPU 分散式訓練 |
| **Depth Extrapolation** | 訓練用 N loops，推理時用 N+k loops（更深的推理能力） |
| **Spectral Radius ρ(A)** | A 矩陣最大特徵值絕對值；ρ(A) < 1 保證 hidden state 收斂 |
| **Parcae** | Prairie et al. 2026 提出的穩定 looped LM 訓練方法論 |
