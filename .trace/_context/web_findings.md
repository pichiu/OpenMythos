# Stage 1 網路搜尋發現 (web_findings.md)

## 搜尋摘要

所有搜尋於 2026-04-23 執行。

---

## 1. OpenMythos 專案本身

**搜尋詞**：`OpenMythos kyegomez recurrent depth transformer Claude Mythos 2026`

| 來源 | 摘要 |
|------|------|
| [GitHub - kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos) | 官方 repo；由 Kye Gomez 維護 |
| [MarkTechPost 介紹](https://www.marktechpost.com/2026/04/19/meet-openmythos-an-open-source-pytorch-reconstruction-of-claude-mythos-where-770m-parameters-match-a-1-3b-transformer/) | 發佈於 2026-04-19；強調 770M 參數 Parcae 模型匹敵 1.3B 標準 Transformer |
| [Awesome Agents 介紹](https://awesomeagents.ai/news/openmythos-recurrent-depth-transformer/) | OpenMythos 作為 Looped MoE Transformer 的說明 |
| [Dataconomy 報導](https://dataconomy.com/2026/04/20/openmythos-project-attempts-to-reconstruct-claude-mythos-design/) | 2026-04-20；定位為推測性重建，非洩漏 |
| [TechBriefly 報導](https://techbriefly.com/2026/04/20/openmythos-project-claims-claude-mythos-is-a-recurrent-depth-transformer/) | Claude Mythos 是 RDT 的假設說明 |
| [36Kr 報導（中文版）](https://eu.36kr.com/en/p/3774953856418309) | 22 歲開發者逆向工程並開源 Mythos 架構 |

**關鍵 Takeaway**：
- OpenMythos 是**推測性重建**，基於公開研究文獻
- Kye Gomez（kyegomez）是 The Swarm Corporation 的成員，並非 Anthropic 相關人員
- 社群對「Claude Mythos 是否真的是 RDT」有爭議

---

## 2. Parcae：穩定循環模型的縮放定律

**搜尋詞**：`Parcae scaling laws stable looped language models Prairie 2026`

| 來源 | 摘要 |
|------|------|
| [arXiv 2604.12946](https://arxiv.org/abs/2604.12946) | 論文頁面 |
| [Sandy Research Blog](https://sandyresearch.github.io/parcae/) | 官方部落格（含互動圖表） |
| [Together AI Blog](https://www.together.ai/blog/parcae) | Together AI 合作研究說明 |
| [MarkTechPost 分析](https://www.marktechpost.com/2026/04/16/ucsd-and-together-ai-research-introduces-parcae-a-stable-architecture-for-looped-language-models-that-achieves-the-quality-of-a-transformer-twice-the-size/) | UCSD + Together AI 合作 |

**關鍵 Takeaway**：
- 作者：Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu（UCSD + Together AI）
- 核心發現：透過 ZOH 離散化將注入參數的 spectral norm 限制在 < 1，保證訓練穩定
- 770M Parcae 模型 = 1.3B 標準 Transformer 的品質（相同訓練資料）
- 建立了 looped model 的第一套縮放定律（power law exponents）
- OpenMythos 的 `LTIInjection` 直接實作此論文

---

## 3. Loop, Think, & Generalize 論文

**搜尋詞**：`loop think generalize implicit reasoning recurrent depth transformers arxiv 2604.07822`

| 來源 | 摘要 |
|------|------|
| [arXiv 2604.07822](https://arxiv.org/abs/2604.07822) | 論文主頁 |

**關鍵 Takeaway**：
- 作者：Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, Yuekun Yao（2026-04-09 投稿）
- 研究焦點：RDT 在 systematic generalization 和 depth extrapolation 兩種組合泛化挑戰的表現
- 結論：增加 inference-time loop 數可以解決訓練時未見過的 N-hop 推理鏈（depth extrapolation）
- 這正是 OpenMythos `generate()` 的 `n_loops` 參數設計依據

---

## 4. Mixture-of-Depths Attention (MoDA)

**搜尋詞**：`Mixture-of-Depths Attention MoDA arxiv 2603.15619`

| 來源 | 摘要 |
|------|------|
| [arXiv 2603.15619](https://arxiv.org/abs/2603.15619) | 論文主頁 |
| [GitHub - hustvl/MoDA](https://github.com/hustvl/MoDA) | 官方硬體優化實作（Triton kernel） |

**關鍵 Takeaway**：
- 作者：Huazhong University of Science & Technology + ByteDance（2026-03-16）
- 核心思想：每個 attention head 同時 attend sequence KV（當前層、因果）+ depth KV（前面所有層的相同 token 位置）
- 性能：比 baseline 降低 0.2 perplexity，下游任務提升 2.11%，FLOPs 僅增加 3.7%
- OpenMythos 在 `open_mythos/moda.py` 完整實作此架構並結合 DeepSeek MoE

---

## 其他相關連結

| 主題 | 連結 |
|------|------|
| Reasoning with Latent Thoughts（Saunshi 2025） | [arXiv 2502.17416](https://arxiv.org/abs/2502.17416) |
| DeepSeekMoE（Dai 2024） | [arXiv 2401.06066](https://arxiv.org/abs/2401.06066) |
| Relaxed Recursive Transformers / LoRA（Bae 2024） | [arXiv 2410.20672](https://arxiv.org/pdf/2410.20672) |
| Anthropic Claude Mythos 外部報導 | [Foreign Policy 2026-04-20](https://foreignpolicy.com/2026/04/20/claude-mythos-preview-anthropic-project-glasswing-cybersecurity-ai-hacking-danger/) |
| COCONUT — 連續潛在空間推理 | [arXiv 2412.06769](https://arxiv.org/abs/2412.06769) |
