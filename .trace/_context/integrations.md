# Stage 2.5 外部整合 (integrations.md)

## 外部依賴與服務呼叫

OpenMythos 是研究函式庫，外部整合相對單純。

---

## 1. PyTorch（核心）

**依賴**：`torch==2.11.0`（pyproject.toml:43）

所有張量運算、nn.Module、autograd、分散式訓練均透過 PyTorch。

**關鍵用法**：
- `torch.distributed`（NCCL backend）：`training/3b_fine_web_edu.py:19`
- `torch.distributed.fsdp.FullyShardedDataParallel`：多 GPU 訓練
- `torch.amp.autocast`：混合精度（bfloat16 / float16）
- `torch.utils.data.IterableDataset`：串流資料集

---

## 2. Hugging Face Transformers

**依賴**：`transformers>=4.40.0`

**唯一用途**：tokenizer 載入（`tokenizer.py:3`）

```python
from transformers import AutoTokenizer

class MythosTokenizer:
    def __init__(self, model_id="openai/gpt-oss-20b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
```

**網路請求**：`from_pretrained()` 第一次呼叫會從 HuggingFace Hub 下載 tokenizer 檔案，之後 cache 在 `~/.cache/huggingface/`。

**失敗處理**：無 retry/fallback 邏輯；若 Hub 不可達或 model_id 不存在，會直接拋出 `OSError`。

---

## 3. Hugging Face Datasets

**依賴**：`datasets>=2.18.0`

**用途**：串流載入 FineWeb-Edu（`training/3b_fine_web_edu.py:92`）

```python
from datasets import load_dataset

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,           # 不全部下載到 disk
).shard(num_shards=total_shards, index=shard_index)
```

**Shard 機制**：`world_size × num_workers` 個 shard，每個 `(rank, worker_id)` 組合擁有一個不相交的 shard，無需跨 process 協調。

**失敗處理**：無 retry；streaming dataset 中斷會導致 `StopIteration`，訓練迴圈會重新初始化 iterator（`training/3b_fine_web_edu.py:487`）。

---

## 4. Flash Attention 2（Optional）

**依賴**：`flash-attn>=2.8.3`（可選）

**條件啟用**（`main.py:8–13`）：

```python
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
```

**使用位置**：`GQAttention.forward`（`main.py:245–258`）

Flash Attn 2 原生支援 GQA（n_kv_heads < n_heads），輸入 cast 為 bf16，完成後 cast 回原始 dtype。

**Fallback**：手動 `torch.matmul` + KV head `repeat_interleave`（`main.py:261–274`）

---

## 5. PyPI（套件發布）

**套件名**：`open-mythos`（注意有連字號）
**版本**：0.5.0
**PyPI 頁面**：https://pypi.org/project/open-mythos/

---

## 6. loguru（訓練腳本）

**依賴**：`loguru`（`training/requirements.txt` 隱含）

```python
from loguru import logger
logger.info("...")
logger.success("Checkpoint saved")
logger.warning("...")
```

僅用於訓練腳本的 structured logging，不影響 library 核心。

---

## 7. HuggingFace Hub（訓練用 tokenizer 源）

訓練腳本使用 `"openai/gpt-oss-20b"` 作為 tokenizer，這是：
- HuggingFace Hub 上的一個公開 tokenizer
- 需要網路存取或 offline cache
- ⚠️ 未驗證：若此模型 ID 不存在或需要登入，訓練腳本啟動即失敗

---

## 失敗模式彙整

| 整合點 | 可能的失敗 | 目前的處理 |
|--------|---------|----------|
| HF Transformers tokenizer | Hub 不可達、model_id 不存在 | 無；直接 raise OSError |
| HF Datasets streaming | 網路中斷、shard 讀取錯誤 | StopIteration → 重新 iter |
| Flash Attn 2 | CUDA 不可用、未安裝 | try/except → 靜默 fallback |
| FSDP 多 GPU | NCCL 錯誤 | 由 PyTorch 處理（無自定義 retry） |
| Checkpoint 儲存 | 磁碟滿 | `OSError` raise，舊 checkpoint 因 tmp+replace 保持完整 |

---

## 無整合的項目

- ❌ 無 REST API / gRPC
- ❌ 無資料庫連線
- ❌ 無訊息佇列（Kafka/RabbitMQ）
- ❌ 無雲端儲存（S3/GCS）整合
- ❌ 無 wandb / TensorBoard 整合（僅用 loguru stdout logging）
- ❌ 無 ONNX / TorchScript export 支援
