# Stage 2.3 核心領域邏輯 (core_logic.md)

## 專案的「心臟」

OpenMythos 的核心是 **RecurrentBlock**（`main.py:788`）及其穩定機制 **LTIInjection**（`main.py:684`）。其次是 **MoEFFN** 的路由機制。

---

## 核心 1：LTI-Stable Recurrent Update

**所在**：`main.py:684–742`

### 問題
訓練 looped transformer 時，若 hidden state 跨 loop 步驟累積導致 spectral radius ≥ 1，會造成：
- residual explosion（hidden state 無限成長）
- loss spike（梯度爆炸）

### 解法（Parcae, Prairie et al., 2026）
將 loop 的 state update 建模為 LTI 動態系統：

```
h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)
```

**A 的穩定性保證**（`main.py:714–725`）：
```python
def get_A(self) -> torch.Tensor:
    # A_continuous = Diag(-exp(log_A))  ← 永遠為負對角矩陣
    # A_discrete = exp(Δt · A_continuous)
    # 在 log space 計算：A_discrete = exp(-exp(log_dt + log_A))
    # clamp(-20, 20) 防止 float32 數值問題
    return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))
    # 輸出 ∈ (0, 1)，無論 log_A / log_dt 為任何值 → ρ(A) < 1 永遠成立
```

**學習參數**：
- `log_A`：shape `(dim,)`，學習 A_continuous 的量級
- `log_dt`：shape `(1,)`，學習離散化步長 Δt
- `B`：shape `(dim,)`，學習輸入注入比例

**為何在 log space 計算**：避免 `0 * inf = NaN`（當 log_dt → -∞, log_A → +∞ 時）。

---

## 核心 2：ACT Halting（自適應計算時間）

**所在**：`main.py:750–780`, `RecurrentBlock.forward:853–889`

```python
class ACTHalting(nn.Module):
    def __init__(self, dim):
        self.halt = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.halt(h)).squeeze(-1)  # → (B, T) ∈ (0,1)
```

**Loop 中的 ACT 邏輯**（`main.py:865–888`）：

```python
halted = zeros(B, T, bool)
cumulative_p = zeros(B, T)
h_out = zeros_like(h)

for t in range(n_loops):
    # ... 更新 h ...
    p = self.act(h)                           # 本步的 halting prob
    still_running = ~halted

    # ACT remainder trick：跨閾值時用剩餘質量，不超過 1
    remainder = (1.0 - cumulative_p).clamp(min=0)
    weight = where(cumulative_p + p >= threshold, remainder, p)
    weight = weight * still_running.float()   # 已 halt 的位置貢獻 0

    h_out += weight.unsqueeze(-1) * h         # 加權累積
    cumulative_p += p * still_running.float()
    halted |= (cumulative_p >= threshold)

    if halted.all() and kv_cache is None:     # 無 cache 才能 early exit
        break
```

**關鍵設計決策**：有 KV cache 時不可 early exit，因為每個 loop 都必須填充對應的 cache 鍵，否則後續 decode 步驟找不到對應的 K/V。

---

## 核心 3：LoRA Depth Adapter

**所在**：`main.py:578–619`

```
純 weight-tying（每 loop 完全相同）
    ↕  LoRAAdapter（中間地帶）
完全獨立層（無參數共享）
```

**實作**（`main.py:603–619`）：
```python
def forward(self, x, loop_t):
    max_t = self.scale.num_embeddings - 1
    t_idx = min(loop_t, max_t)          # depth extrapolation: clamp 到訓練範圍
    s = self.scale(tensor(t_idx))       # (rank,) — per-loop 縮放向量
    down = self.down(x) * s             # (B,T,rank) — 共享下投影 × loop-specific 縮放
    return down @ self.B                 # (B,T,dim) — 共享上投影
```

`scale` 是 `Embedding(max_loops, rank)`：每個 loop index 有一個獨立的 rank 維縮放向量。

**Depth Extrapolation 的 clamp**：若 inference 使用更多 loop（超過訓練的 max_loop_iters），超出的 loop index 使用最後一個訓練 loop 的 scale，避免 out-of-range index。

---

## 核心 4：Loop-Index Sinusoidal Embedding

**所在**：`main.py:541–570`

```python
def loop_index_embedding(h, loop_t, loop_dim, theta=10000.0):
    freqs = 1.0 / (theta ** (arange(0, loop_dim, 2) / loop_dim))
    angles = loop_t * freqs                          # 每個頻率 × loop index
    emb = cat([angles.sin(), angles.cos()])[:loop_dim]  # 類似 sinusoidal PE
    emb_full = zeros(h.shape[-1])
    emb_full[:loop_dim] = emb
    return h + emb_full                              # 只改變前 loop_dim 個 channel
```

- `loop_dim = cfg.dim // 8`（`main.py:821`）：只有 12.5% 的 channel 受影響
- 與 sequence position 的 RoPE 類比，但作用在 recurrence depth 維度
- 讓相同的 weights 在不同 loop 深度時表現出不同的行為

---

## 核心 5：MoE Routing（Aux-Loss-Free Load Balancing）

**所在**：`main.py:497–532`

```python
logits = self.router(flat)                  # (B*T, n_experts) — 無偏差
scores = F.softmax(logits, dim=-1)          # 用於 gate weights
_, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
                                             # bias 只影響選誰，不影響 weights
topk_scores = scores.gather(-1, topk_idx)
topk_scores /= topk_scores.sum(-1, keepdim=True)  # renorm
```

**DeepSeek-V3 啟發的設計**：
- `router_bias` 是 **buffer**（非 gradient parameter）
- 訓練時外部監控 expert 使用率，調整 bias 讓低使用率 expert 被選中更多
- gradient 只流過 `scores`（softmax），不流過 `router_bias`
- 避免傳統 auxiliary balance loss 對 primary loss 的干擾

**Shared experts 作為 bypass**：
```python
for shared in self.shared_experts:
    out = out + shared(flat)    # 每個 token 都通過，無論 routing 結果
```
Shared expert 的 hidden dim = `expert_dim × n_experts_per_tok`（比單個 routed expert 大 topk 倍）。

---

## 核心 6：MoDA（替代架構）的 Depth Attention

**所在**：`moda.py:740–814`

MoDA 將 attention 擴展至跨 layer 的深度維度：

```
單層標準 attention：
  Q(x_l) · K(x_l)^T    ← 只看當前層的 sequence KV

MoDA attention：
  Q(x_l) · cat([K(x_l), K_depth_0, K_depth_1, ..., K_depth_{l-1}])^T
                          ↑ sequence            ↑ 跨所有前面層的 depth KV

單一 softmax 跨所有 (T + L) 個位置
```

**Depth KV Cache 寫入**（`moda.py:903–913`）：
- 每層 block 輸出 `x_out` 後，用 `k_write`、`v_write` 投影產生此層的 depth KV
- 下一層 attention 時，這些 depth KV 被 append 到 cache list 中

---

## Pattern 總結

| Pattern | 實現方式 | 主要 class/函式 |
|---------|---------|---------------|
| State machine（LTI） | ZOH 離散化負對角矩陣 | `LTIInjection` |
| Adaptive halting | 累積概率 + remainder trick | `ACTHalting` |
| Weight sharing + adaptation | 共享矩陣 + per-loop scale embedding | `LoRAAdapter` |
| Position-aware recurrence | Sinusoidal loop-index signal | `loop_index_embedding` |
| Sparse expert dispatch | Softmax routing + bias load balancing | `MoEFFN` |
| Depth-aware attention | Cross-layer KV cache + unified softmax | `MoDAAttention` |
