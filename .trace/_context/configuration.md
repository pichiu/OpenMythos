# Stage 2.6 設定與環境 (configuration.md)

## 設定載入機制

OpenMythos 是純 library，**無環境變數設定**，所有設定透過 `MythosConfig` dataclass 注入。

---

## 主要設定：MythosConfig

**所在**：`open_mythos/main.py:17–80`

`MythosConfig` 是 `@dataclass`，所有欄位有預設值，無需任何設定檔。

### 欄位完整表格

| 欄位 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `vocab_size` | int | 32000 | token 詞彙表大小 |
| `dim` | int | 2048 | hidden dimension |
| `n_heads` | int | 16 | query heads 數量 |
| `n_kv_heads` | int | 4 | KV heads（GQA 用） |
| `max_seq_len` | int | 4096 | 最大序列長度（RoPE 預計算） |
| `max_loop_iters` | int | 16 | 預設 recurrent loop 深度 T |
| `prelude_layers` | int | 2 | Prelude 層數 |
| `coda_layers` | int | 2 | Coda 層數 |
| `attn_type` | str | `"mla"` | `"gqa"` 或 `"mla"` |
| `kv_lora_rank` | int | 512 | [MLA] KV 壓縮 latent 維度 |
| `q_lora_rank` | int | 1536 | [MLA] Q 壓縮 latent 維度 |
| `qk_rope_head_dim` | int | 64 | [MLA] RoPE head dim |
| `qk_nope_head_dim` | int | 128 | [MLA] non-RoPE head dim |
| `v_head_dim` | int | 128 | [MLA] value head dim |
| `n_experts` | int | 64 | routed expert 總數 |
| `n_shared_experts` | int | 2 | shared expert 數（永遠啟用） |
| `n_experts_per_tok` | int | 4 | top-K routed experts per token |
| `expert_dim` | int | 512 | 每個 expert 的 hidden dim |
| `act_threshold` | float | 0.99 | ACT 累積停止閾值 |
| `rope_theta` | float | 500000.0 | RoPE 基礎頻率（LLaMA-3 預設） |
| `lora_rank` | int | 16 | depth-wise LoRA rank |
| `max_output_tokens` | int | 4096 | generate 的最大輸出長度 |
| `dropout` | float | 0.0 | dropout 率（0=關閉，0.1 適合預訓練） |

### 設定使用範例

```python
# 最小設定（快速測試）
cfg = MythosConfig(
    vocab_size=8192, dim=256, n_heads=4, n_kv_heads=2,
    max_seq_len=512, max_loop_iters=4,
    prelude_layers=1, coda_layers=1, attn_type="gqa",
    n_experts=8, n_shared_experts=1, n_experts_per_tok=2,
    expert_dim=64, lora_rank=4,
)

# 使用預設規格（推薦）
from open_mythos.variants import mythos_3b
cfg = mythos_3b()

# 手動覆蓋特定欄位
cfg = mythos_3b()
cfg.vocab_size = 50000   # ⚠️ dataclass 並非 frozen，可以直接賦值
cfg.max_seq_len = 2048
```

---

## 訓練腳本設定（Literal Constants）

**所在**：`training/3b_fine_web_edu.py:374–386`

```python
# 以下均為函式內的 literal 常數，無 CLI/config 介面
seq_len = 2048
micro_batch = 4
target_tokens = 30_000_000_000
grad_accum = max(1, 256 // (world_size * micro_batch))
global_batch_tok = world_size * micro_batch * grad_accum * seq_len
total_steps = target_tokens // global_batch_tok
warmup_steps = 2000
lr = 3e-4
wd = 0.1
log_every = 10
ckpt_every = 1000
ckpt_dir = "checkpoints"
dataset_subset = "sample-10BT"   # ← 切換這個換資料集
```

**設計意圖**（docstring line 337）：「所有超參數為 literal 常數，每次 run 釘死設定；刻意避免 CLI/config 層以保持檔案自我說明性。」

---

## 環境變數

僅有訓練腳本讀取 PyTorch/torchrun 標準環境變數：

| 環境變數 | 讀取位置 | 用途 |
|---------|---------|------|
| `RANK` | `training/3b_fine_web_edu.py:342` | 全域 rank；若未設定（== -1）則為單 GPU 模式 |
| `LOCAL_RANK` | line 347 | 本機 GPU index |
| `WORLD_SIZE` | line 348 | 分散式 world size |

這些均由 `torchrun` 自動設定，不需手動配置。

---

## 優先順序（Precedence）

由於無 config file、無 env var 影響模型行為，只有：

```
程式碼預設值（MythosConfig 欄位預設）
    ↓ 若使用 variants.py
工廠函式（mythos_1b() 等）的完整設定
    ↓ 若需調整
呼叫方直接賦值（cfg.vocab_size = X）
    ↓
傳入 OpenMythos(cfg)
```

---

## Secret 管理

- **無 secrets**：此 library 無 API key、無資料庫密碼
- 唯一可能的「secret」是 HuggingFace Hub token（若使用私有 tokenizer），但須使用者自行設定 `HF_TOKEN` env var 或 `huggingface-cli login`，library 未做任何處理

---

## Checkpoint 設定

訓練腳本的 checkpoint 格式（`training/3b_fine_web_edu.py:239–248`）：

```python
{
    "step": int,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "cfg": MythosConfig,      # 完整 config object 序列化進去
    "vocab_size": int,
}
```

checkpoint 路徑：`checkpoints/step_{N:07d}.pt`（7 位零填充，確保字典序 = 時間序）

**保留策略**：預設 keep last 3（`save_checkpoint(..., keep_last=3)`），可調整參數。

---

## MoDAConfig（替代模型設定）

**所在**：`moda.py:59–123`

| 欄位組 | 關鍵欄位 |
|--------|---------|
| 基礎 Transformer | `vocab_size`, `d_model`, `n_layers`, `n_heads_q`, `n_heads_kv`, `head_dim`, `max_seq_len` |
| 位置編碼 | `rope_base`（預設 10000.0，比 `MythosConfig` 的 500000.0 慢很多） |
| MoE | `n_shared_experts`, `n_routed_experts`, `n_activated_experts`, `expert_hidden_dim` |
| 負載均衡 | `moe_balance_alpha`（0.0 = 停用 balance loss） |
| DeepSeek V3 相容 | `moe_score_func`（"softmax"/"sigmoid"）, `moe_n_groups`, `moe_topk_groups`, `moe_route_scale` |
