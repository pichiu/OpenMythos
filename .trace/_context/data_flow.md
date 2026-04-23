# Stage 2.2 Data Flow (data_flow.md)

## 代表性 Use Case：自回歸文字生成

從 `model.generate(input_ids, max_new_tokens=8, n_loops=4)` 完整 trace。

---

## Forward Pass 完整資料流（訓練 / prefill）

```
input_ids: (B, T)    e.g. (2, 16) — token indices
    │
    ▼ main.py:1019
x = self.embed(input_ids)         → (B, T, dim)
    │
    ├─ 選擇 RoPE buffer:          main.py:1020
    │    attn_type=="mla"  → freqs_cis_mla[start_pos:start_pos+T]  (T, rope_dim//2) complex
    │    attn_type=="gqa"  → freqs_cis[start_pos:start_pos+T]      (T, head_dim//2) complex
    │
    ├─ 建 causal mask (T>1):      main.py:1023
    │    mask = upper-triangular -inf  shape (1,1,T,T)
    │    T==1 → mask = None
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Prelude Loop  (prelude_layers 次，各層各自一套權重)│
│  for i, layer in enumerate(self.prelude):        │
│    x = layer(x, freqs_cis, mask,                │
│               kv_cache, f"prelude_{i}")          │
│    TransformerBlock.forward():    main.py:653    │
│      x = x + dropout(attn(norm(x), ...))        │
│      x = x + dropout(ffn(norm(x)))              │
│       ↑ dense Expert FFN（非 MoE）              │
└────────────────────┬────────────────────────────┘
                     │ x: (B, T, dim)
    ▼ main.py:1028
e = x   ← 凍結為 encoded input，供每個 loop 注入

┌─────────────────────────────────────────────────┐
│ RecurrentBlock.forward()     main.py:825        │
│  初始 h = x (即 e)                               │
│  for t in range(n_loops):                        │
│    ① h_loop = loop_index_embedding(h, t, dim//8)│
│    ② combined = RMSNorm(h_loop + e)             │
│    ③ trans_out = block(combined, ...)            │
│       ← TransformerBlock (use_moe=True)         │
│       ← MoEFFN：routed experts top-K + shared  │
│    ④ trans_out += lora(trans_out, t)            │
│       ← LoRAAdapter (depth-wise delta)          │
│    ⑤ h = injection(h, e, trans_out)             │
│       ← LTIInjection: A·h + B·e + trans_out    │
│    ⑥ p = act(h)   → (B,T)  halting prob        │
│    ⑦ 累積 ACT，更新 h_out 加權和               │
│    ⑧ 若 halted.all() and kv_cache is None → break│
│  return h_out    ← ACT-weighted 跨 loop 平均    │
└────────────────────┬────────────────────────────┘
                     │ x: (B, T, dim)

┌─────────────────────────────────────────────────┐
│ Coda Loop  (coda_layers 次，dense FFN)           │
│  for i, layer in enumerate(self.coda):           │
│    x = layer(x, freqs_cis, mask,                │
│               kv_cache, f"coda_{i}")             │
└────────────────────┬────────────────────────────┘
                     │ x: (B, T, dim)
    ▼
logits = self.head(self.norm(x))   → (B, T, vocab_size)
```

---

## GQAttention 內部資料流（`main.py:212`）

```
x: (B,T,dim)
    │
    ├─ wq(x) → (B,T, n_heads*head_dim) → view → (B,T,H,head_dim)
    ├─ wk(x) → (B,T, n_kv_heads*head_dim) → view → (B,T,Hkv,head_dim)
    └─ wv(x) → (B,T, n_kv_heads*head_dim) → view → (B,T,Hkv,head_dim)
    │
    ├─ apply_rope(q, freqs)  apply_rope(k, freqs)
    │
    ├─ [kv_cache] cat past K,V → grow sequence dim
    │
    ├─ Flash Attn 路徑（_HAS_FLASH_ATTN）：
    │    cast to bf16 → flash_attn_func(q,k,v, causal=...) → cast back
    │
    └─ Fallback 路徑：
         k,v repeat_interleave(groups) → scaled dot-product → softmax → dropout
    │
    └─ wo(out) → (B,T,dim)
```

## MLAttention 內部資料流（`main.py:357`）

```
x: (B,T,dim)
    │
    Q path:
    ├─ q_down(x) → (B,T, q_lora_rank)
    ├─ q_norm(...)
    ├─ q_up_nope → (B,T,H, qk_nope_dim)   [無 RoPE]
    └─ q_up_rope → (B,T,H, qk_rope_dim)   [apply_rope]
    q = cat(q_nope, q_rope)  per head     (B,T,H, nope+rope)

    KV path:
    ├─ kv_down(x) → (B,T, kv_lora_rank + qk_rope_dim)
    ├─ c_kv = [..., :kv_lora_rank]         ← 只有這個被 cache
    └─ k_rope = [..., kv_lora_rank:]       ← expand + apply_rope + cache
    │
    ├─ [kv_cache] cat c_kv, k_rope
    │
    c_kv → kv_norm → kv_up → split → k_nope, v   (每步重建，不 cache)
    k = cat(k_nope, k_rope)
    │
    └─ scaled dot-product attn → wo → (B,T,dim)
```

## MoEFFN 資料流（`main.py:497`）

```
x: (B,T,dim)
    │
    flat = x.view(B*T, dim)
    │
    ├─ router(flat) + router_bias → logits (B*T, n_experts)
    ├─ softmax(logits) → scores（用於 gate weights）
    ├─ topk(logits + bias, k=topk) → topk_idx   ← bias 只影響選擇，不影響 weights
    ├─ scores.gather(topk_idx) → topk_scores → renorm
    │
    ├─ Routed dispatch（per expert scatter）:
    │    for i in range(topk):
    │      for eid in range(n_experts):
    │        mask = expert_ids == eid
    │        out[mask] += scores[mask] * routed_experts[eid](flat[mask])
    │
    └─ Shared experts（全部 token）:
         for shared in shared_experts:
           out += shared(flat)
    │
    out.view(B,T,dim) → (B,T,dim)
```

---

## 自回歸 Generate 流程（`main.py:1036`）

```
input_ids: (B, T_prompt)
kv_cache = {}

step 0:
    cur_ids = input_ids      # (B, T_prompt)
    start_pos = 0
    → forward(cur_ids, n_loops, kv_cache, start_pos=0)
    → logits[:,-1,:] / temperature
    → top_k masking → softmax → multinomial → next_tok
    input_ids = cat([input_ids, next_tok])   # (B, T_prompt+1)

step 1..N:
    cur_ids = input_ids[:, -1:]   # (B, 1) ← 只傳最新 token
    start_pos = T_prompt + step - 1
    → forward(cur_ids, n_loops, kv_cache, start_pos)
      ├─ kv_cache 提供所有歷史 K,V
      └─ mask = None（T=1，無需因果遮罩）
    → next_tok → append

return input_ids   # (B, T_prompt + max_new_tokens)
```

**Cache key 命名規則**（`main.py:1026`）：
- Prelude layer i → `"prelude_{i}"`
- Recurrent loop t → `"recurrent_loop_{t}"`
- Coda layer i → `"coda_{i}"`

---

## 訓練資料流（`training/3b_fine_web_edu.py`）

```
FineWebEduDataset.__iter__():
    HF streaming dataset（shard by rank × worker）
    → encode document text → token id list
    → rolling buffer sliced into (seq_len+1) chunks
    → yield (input_ids[:-1], target_ids[1:])  # next-token prediction

DataLoader (batch_size=micro_batch, num_workers=4, pin_memory=True)
    → (x, y): (micro_batch, seq_len)

Training loop:
    for micro_step in range(grad_accum):
        logits = model(x)                        # (B, T, vocab_size)
        loss = cross_entropy(logits, y) / grad_accum
        loss.backward()
    clip_grad_norm_(1.0)
    optimizer.step()
```
