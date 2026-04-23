# Stage 2.4 Extension Points (extensions.md)

## 可替換點與擴展機制

OpenMythos 是研究導向的函式庫，擴展點主要透過 **設定注入** 和 **子類別替換**。

---

## 1. 注意力機制替換（最主要的替換點）

**設定點**：`MythosConfig.attn_type`（`main.py:59`）

```python
# 切換注意力實作
cfg = MythosConfig(attn_type="gqa")   # GQA：更少 KV heads
cfg = MythosConfig(attn_type="mla")   # MLA（預設）：壓縮 KV cache
```

**內部選擇邏輯**（`main.py:649`）：
```python
self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
```

**擴展方式**：若要新增第三種 attention（e.g. GQA + Flash + Sliding Window），需：
1. 新增 class 繼承 `nn.Module`，實作與 `GQAttention.forward` 相同的簽名
2. 修改 `TransformerBlock.__init__` 的選擇邏輯

---

## 2. FFN 類型替換

**設定點**：`TransformerBlock.__init__(cfg, use_moe: bool)`（`main.py:640`）

```python
# Dense FFN（Prelude, Coda 使用）
block = TransformerBlock(cfg, use_moe=False)  # → Expert(dim, dim*4//3)

# MoE FFN（RecurrentBlock 使用）
block = TransformerBlock(cfg, use_moe=True)   # → MoEFFN(cfg)
```

**FFN 替換接口**：`TransformerBlock.ffn` 屬性，forward 簽名為 `(x: Tensor) → Tensor`。

---

## 3. 模型規格工廠函式（`variants.py`）

```python
# 現有工廠函式
mythos_1b() / mythos_3b() / mythos_10b() / mythos_50b()
mythos_100b() / mythos_500b() / mythos_1t()

# 新增自定義規格
def my_custom_config() -> MythosConfig:
    return MythosConfig(
        dim=1024,
        n_heads=8,
        ...
    )
```

每個工廠函式返回一個完整的 `MythosConfig`，可傳入 `OpenMythos(cfg)` 建構模型。

---

## 4. 完整替代架構（moda.py）

`MoDAModel` 是獨立的替代模型，可替換 `OpenMythos`：

```python
# 原始 RDT 架構
from open_mythos import OpenMythos, MythosConfig
model = OpenMythos(MythosConfig(...))

# MoDA 替代架構（Depth-aware attention + DeepSeek MoE）
from open_mythos.moda import MoDAModel, MoDAConfig
model = MoDAModel(MoDAConfig(...))
```

兩者均繼承 `nn.Module`，`forward` 簽名略有不同：
- `OpenMythos.forward(input_ids, n_loops, kv_cache, start_pos)` → `(B,T,vocab_size)`
- `MoDAModel.forward(input_ids, labels)` → `(logits, loss)`

---

## 5. Tokenizer 替換

**設定點**：`MythosTokenizer(model_id)` 預設使用 `"openai/gpt-oss-20b"`

```python
# 預設 tokenizer
tok = MythosTokenizer()

# 替換為任何 HuggingFace tokenizer
tok = MythosTokenizer("meta-llama/Llama-3-8B")
tok = MythosTokenizer("/local/path/to/tokenizer")
```

**接口要求**：`tokenizer.py` 封裝的公開接口為 `encode(str) → list[int]`、`decode(list[int]) → str`、`vocab_size: int`。訓練腳本使用 `encoding.encode(sample["text"])` 即此接口。

---

## 6. 訓練設定擴展點（`training/3b_fine_web_edu.py`）

訓練腳本中所有超參數為 **函式內 literal 常數**（`main()` 第 374-386 行），這是有意的設計（減少 CLI/config 層的維護負擔）。需要調整時直接修改對應行：

| 常數 | 位置 | 說明 |
|------|------|------|
| `seq_len = 2048` | line 374 | context 長度 |
| `micro_batch = 4` | line 375 | 每 GPU 的 micro batch |
| `target_tokens = 30_000_000_000` | line 376 | 訓練 token 目標（30B） |
| `lr = 3e-4` | line 381 | peak learning rate |
| `dataset_subset = "sample-10BT"` | line 386 | 切換為 `"sample-100BT"` 或 `"default"` |
| `ckpt_every = 1000` | line 384 | checkpoint 頻率 |

---

## 7. 負載均衡 Hook

`MoEFFN.router_bias` 是 non-gradient buffer（`main.py:485`），訓練時需**外部調整**：

```python
# 訓練迴圈中（非 optimizer 管理）
with torch.no_grad():
    model.recurrent.block.ffn.router_bias[overloaded_experts] -= delta
    model.recurrent.block.ffn.router_bias[underused_experts] += delta
```

目前訓練腳本（`3b_fine_web_edu.py`）**未實作**此外部 bias 調整邏輯，是已知技術債。

---

## 8. Flash Attention 條件啟用

Flash Attention 2 的啟用完全透明，無需程式碼修改（`main.py:8–13`）：

```python
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
```

- 安裝 `pip install open-mythos[flash]` → 自動啟用
- 未安裝 → fallback 到手動 scaled dot-product attention
- 只影響 `GQAttention`；`MLAttention` 不支援 Flash Attn（使用標準 matmul）

---

## 可以新增功能而不改核心的方式

| 需求 | 建議方式 |
|------|---------|
| 新的 attention 類型 | 實作 `nn.Module`，修改 `TransformerBlock.__init__` 的一行 if/else |
| 新的模型規格 | 在 `variants.py` 新增 `def mythos_Xb() -> MythosConfig:` |
| 新的 tokenizer | 換 `MythosTokenizer(model_id)` 的 model_id |
| 新的 expert FFN | 繼承 `Expert` 或替換 `MoEFFN`，保持 `(B,T,dim) → (B,T,dim)` 簽名 |
| 新的訓練資料集 | 實作 `IterableDataset`，保持 `yield (input_ids, target_ids)` 接口 |
| 替代架構實驗 | 使用 `MoDAModel` 或新增同層級的模組 |
