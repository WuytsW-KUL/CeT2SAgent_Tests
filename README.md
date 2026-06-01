# CeT2SAgent Tests

Evaluation results for the **Context-Enhanced Text-to-SPARQL Agent (CeT2S-Agent)** on the [QALD-9-Plus](https://github.com/KGQA/QALD_9_plus) benchmark (104 answerable English test questions over DBpedia).

All macro-F1 values are computed per-question and averaged across runs using `calc_avg_macro_f1.py`. Scores are expressed as percentages (×100).

CeT2S-Agent = https://github.com/WuytsW-KUL/CeT2SAgent

---

## English Baseline Comparison

All agents evaluated on the English test set. Each configuration was run **3 times**; the table shows individual run scores and the macro-F1 average.

| Agent | Run 1 | Run 2 | Run 3 | **Avg F1 (%)** |
|---|---|---|---|---|
| **CeT2S-Agent — GPT-4o** | 58.10 | 55.13 | 58.13 | **57.12** |
| **CeT2S-Agent — GPT-4o-mini** | 54.64 | 54.16 | 53.76 | **54.19** |
| mKGQAgent — GPT-4o | 47.74 | 47.04 | 45.45 | 46.74 |
| mKGQAgent — GPT-4o-mini | 33.52 | 35.51 | 32.05 | 33.69 |
| KG-Agent (Data-Shapes) — GPT-4o | 43.35 | 44.54 | 44.13 | 44.00 |
| KG-Agent (Data-Shapes) — GPT-4o-mini | 33.21 | 33.54 | 33.21 | 33.32 |

---

## Ablation Study

All ablation runs use **GPT-4o-mini** on the English test set (**3 runs** each). The full CeT2S-Agent (GPT-4o-mini) score of **54.19%** serves as the reference.

| Configuration | Run 1 | Run 2 | Run 3 | **Avg F1 (%)** | **Δ vs Full** |
|---|---|---|---|---|---|
| Full CeT2S-Agent *(reference)* | 54.64 | 54.16 | 53.76 | **54.19** | — |
| No EAT Detection | 56.86 | 54.04 | 56.04 | **55.65** | +1.46 |
| No Context Generation | 48.08 | 50.38 | 49.48 | **49.31** | −4.88 |
| No Context, No EAT | 46.15 | 46.55 | 50.51 | **47.74** | −6.45 |
| No ICL Retrieval | 40.60 | 44.32 | 41.20 | **42.04** | −12.15 |
| No ICL, No EAT | 37.18 | 36.32 | 39.86 | **37.79** | −16.40 |
| No Context, No ICL | 28.98 | 29.05 | 29.41 | **29.15** | −25.04 |
| No Context, No ICL, No EAT | 28.40 | 26.15 | 28.78 | **27.78** | −26.41 |

---

## Multilingual Evaluation — Translation Impact

All runs use **GPT-4o-mini** (**2 runs** per language). Scores compare CeT2S-Agent with translation to English, CeT2S-Agent without translation, and mKGQAgent (with its own multilingual strategy).

| Language | CeT2S-Agent Run 1 | CeT2S-Agent Run 2 | **Avg (%)** | No-Translate Run 1 | No-Translate Run 2 | **Avg (%)** | mKGQAgent Run 1 | mKGQAgent Run 2 | **Avg (%)** |
|---|---|---|---|---|---|---|---|---|---|
| German (de) | 50.08 | 47.10 | **48.59** | 45.13 | 44.64 | **44.88** | 35.17 | 31.09 | **33.13** |
| Russian (ru) | 49.99 | 50.89 | **50.44** | 45.36 | 44.65 | **45.00** | 28.73 | 28.04 | **28.39** |
| Ukrainian (uk) | 48.69 | 47.43 | **48.06** | 46.88 | 46.66 | **46.77** | 30.52 | 29.17 | **29.84** |
| Lithuanian (lt) | 50.79 | 51.00 | **50.90** | 38.53 | 41.64 | **40.08** | 29.79 | 27.03 | **28.41** |
| Belarusian (be) | 47.74 | 46.41 | **47.08** | 38.90 | 41.78 | **40.34** | 22.22 | 31.44 | **26.83** |
| Bashkir (ba) | 32.21 | 29.54 | **30.87** | 30.81 | 34.41 | **32.61** | 16.09 | 19.18 | **17.64** |
| Spanish (es) | 50.14 | 47.65 | **48.89** | 40.60 | 40.32 | **40.46** | 28.29 | 29.33 | **28.81** |
| **Average (non-English)** | | | **46.40** | | | **41.45** | | | **27.58** |
