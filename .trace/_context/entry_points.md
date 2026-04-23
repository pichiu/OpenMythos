# Stage 2.1 Entry Points (entry_points.md)

## 程式進入點

OpenMythos 是一個 **pure library**，沒有長期運行的 server。進入點分為三類：

---

## 1. Library 使用入口（主要）

### `open_mythos/__init__.py`
所有公開 API 從此匯出。使用者 `import open_mythos` 後可直接存取：

```python
from open_mythos.main import (
    MythosConfig, OpenMythos,
    RMSNorm, GQAttention, MLAttention, Expert, MoEFFN,
    LoRAAdapter, TransformerBlock, LTIInjection, ACTHalting, RecurrentBlock,
    precompute_rope_freqs, apply_rope, loop_index_embedding,
)
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.variants import (
    mythos_1b, mythos_3b, mythos_10b, mythos_50b,
    mythos_100b, mythos_500b, mythos_1t,
)
```

⚠️ `__init__.py:52` 匯出 `load_tokenizer`、`get_vocab_size`，但這兩個函式在任何模組均未定義，呼叫會拋出 `ImportError`。

### 模型建構流程（`open_mythos/main.py:899`）

```
MythosConfig(...)          # 設定 dataclass
    ↓
OpenMythos(cfg)            # nn.Module 子類別
    ↓ __init__:
    1. nn.Embedding(vocab_size, dim)              → self.embed        (line 934)
    2. precompute_rope_freqs(...)                  → self.freqs_cis    (line 937)
    3. precompute_rope_freqs(qk_rope_head_dim,...) → self.freqs_cis_mla (line 941)
    4. [TransformerBlock × prelude_layers]         → self.prelude      (line 946)
    5. RecurrentBlock(cfg)                         → self.recurrent    (line 949)
    6. [TransformerBlock × coda_layers]            → self.coda         (line 950)
    7. RMSNorm(dim)                                → self.norm         (line 954)
    8. nn.Linear(dim, vocab_size, bias=False)      → self.head         (line 955)
    9. self.head.weight = self.embed.weight        # weight tying      (line 956)
   10. _init_weights(): N(0, 0.02)                                     (line 960)
```

---

## 2. 替代模型入口（`open_mythos/moda.py`）

`MoDAModel` 是完整獨立的替代架構，不共用 `OpenMythos` 的 class：

```python
from open_mythos.moda import MoDAConfig, MoDAModel

cfg = MoDAConfig(...)
model = MoDAModel(cfg)
```

`MoDAModel.__init__`（`moda.py:934`）：
1. `nn.Embedding(vocab_size, d_model)` → `self.embed`
2. `RotaryEmbedding(head_dim, max_seq_len, rope_base)` → `self.rope`
3. `[MoDABlock × n_layers]` → `self.blocks`
4. `RMSNorm` → `self.norm_out`
5. `nn.Linear(d_model, vocab_size, bias=False)` → `self.lm_head`
6. weight tying: `lm_head.weight = embed.weight`

---

## 3. 訓練腳本入口（`training/3b_fine_web_edu.py`）

```python
# 單 GPU 啟動
python training/3b_fine_web_edu.py

# 多 GPU 啟動（torchrun）
torchrun --nproc_per_node=N training/3b_fine_web_edu.py
```

`main()` 函式（`training/3b_fine_web_edu.py:311`）初始化順序：

```
1. dist.init_process_group("nccl")    # 若 RANK env var 存在
2. MythosTokenizer()                  # → vocab_size
3. mythos_3b()                        # → MythosConfig
4. OpenMythos(cfg)                    # → model
5. FSDP(model, ...)                   # 若多 GPU
6. AdamW(model.parameters(), ...)     # optimizer（必須在 FSDP 後）
7. load_checkpoint(...)               # 若 checkpoints/ 存在
8. FineWebEduDataset + DataLoader
9. training loop
```

**重要**：optimizer 必須在 FSDP wrap **之後**建立（FSDP 會 re-flatten parameters）。

---

## Initialization 關鍵細節

| 元件 | 初始化方式 | 所在行 |
|------|-----------|--------|
| Linear / Embedding weights | `N(0, 0.02)` | `main.py:965` |
| `LTIInjection.log_A` | `zeros(dim)` → A≈1（訓練前不穩定；需訓練才收斂） | `main.py:710` |
| `LTIInjection.log_dt` | `zeros(1)` | `main.py:711` |
| `LTIInjection.B` | `ones(dim) * 0.1` | `main.py:712` |
| `MoEFFN.router_bias` | `zeros(n_experts)`（buffer，非 parameter） | `main.py:485` |
| RoPE 頻率 | 模型建構時一次性預計算，存為 buffer | `main.py:937` |

---

## 模型規格快速參考（`variants.py`）

| 函式 | dim | Experts | Loop iters | 預估參數 |
|------|-----|---------|-----------|---------|
| `mythos_1b()` | 2048 | 64 | 16 | ~1B |
| `mythos_3b()` | 3072 | 64 | 16 | ~3B |
| `mythos_10b()` | 4096 | 128 | 24 | ~10B |
| `mythos_50b()` | 6144 | 256 | 32 | ~50B |
| `mythos_100b()` | 8192 | 256 | 32 | ~100B |
| `mythos_500b()` | 12288 | 512 | 48 | ~500B |
| `mythos_1t()` | 16384 | 512 | 64 | ~1T |
