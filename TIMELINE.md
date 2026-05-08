# Research Timeline — Supervised SAE Project

Chronological reading order of the project's research summaries. Each entry's headline result links to the full document; the per-version-tag code change log is in [`changes.md`](changes.md).

---

| # | Date | Version | File | Headline |
|---:|---|---|---|---|
| 0 | (pre-dated) | initial | [`summary.md`](summary.md) | First sketch: 7-step pipeline turning a pretrained SAE into a labeled feature dictionary. |
| 1 | 2026-03-21 | v1 | [`summary1.md`](summary1.md) | First end-to-end run with local annotation (gpt-oss-20b) + MSE supervision; 127 features × 500 seqs. Single-circuit (rabbit→habit) proof-of-concept. |
| 2 | 2026-03-30 | v2.0 | [`summary2.md`](summary2.md) | Test catalog (8 prefix-decidable surface features) on GPT-2 Small layer 8. Validates the basic loop. |
| 3 | 2026-03-31 | v3.0 | [`summary3.md`](summary3.md) | Scaling test catalog → full ~127-feature catalog; first reconstruction-vs-classifier trade-offs surface. |
| 4 | 2026-04-07 | v4.0 | [`summary4.md`](summary4.md) | Migration to Gemma-2-2B layer 20 (d_model=2304). Confirms pipeline portability across model families. |
| 5 | 2026-04-09 | v6.0 | [`summary5.md`](summary5.md) | Phase 3: first head-to-head between supervised SAE and pretrained unsupervised baseline. Calibrated F1 = 0.629 on Gemma-2-2B; 33/68 features causally active. |
| 6 | 2026-04-13 | v7.0 | [`summary6.md`](summary6.md) | Frozen decoder + amplification sweep. Proves cosine = 1.0 by construction is a useful contract; ablates against learned-decoder variants. |
| 7 | 2026-04-19 | v7.0 | [`summary7.md`](summary7.md) | First valid supervised-vs-pretrained comparison on **GPT-2 Small layer 9** (the densest semantic band). *Superseded by v8.1 methodology.* |
| 8 | 2026-04-21 | v8.0–v8.10 | [`summary8.md`](summary8.md) | Discovery loop ships its first two real promoted features. Methodology retrenchment: linear-probe baseline now matches sup-SAE on cal F1; pivot from "better features" to "calibration-honest classification at z=0." |
| 9 | 2026-04-29 | v8.11–v8.18.34 | [`summary9.md`](summary9.md) | **Methodology synthesis at scale.** Boundary-discipline contract introduced; literal-hinge formulation (zero-margin, no pos_weight, frozen decoder) becomes the default. Probe-vs-SAE causal targeting asymmetry **1100×** (median targeting ratio 682× sup vs 0.59× probe). 12/46 features pass per-feature causal necessity (KL up to 2.09 nats). 5K-seq production: cal F1 = 0.564. |
| 10 | 2026-05-05 | v8.20.12 | [`summary10.md`](summary10.md) | **10K-seq production at 4B annotator on the curated 104-feature catalog: cal F1 = 0.556.** Real EleutherAI Delphi auto-interp on the same SAE/layer reaches **median F1 = 0.010 / mean F1 = 0.025** — 23× gap. Manual catalog audit shows 71% of Delphi descriptions are non-falsifiable strings. |
| 11 | 2026-05-06 | v8.21 | [`summary11.md`](summary11.md) | **Latest production result.** 8B annotator + boundary-discipline exclusions on the 103-feature catalog → cal F1 = **0.604**. Sequence-split eval (whole documents held out) costs only Δ = −0.004 F1 and tightens the naive L0 ratio to **1.015**. The catalog-quality lever is the lever: three operating points (466 broad / 104 curated 4B / 103 curated 8B+excl) produce 0.481 / 0.556 / 0.604 — all dramatically beating the 0.025 unsupervised auto-interp baseline. |

---

## Reading paths for different audiences

### "I have 5 minutes"
[`summary11.md`](summary11.md) — the latest result, with the four-cell rigor table and the sequence-split / catalog-quality findings.

### "I want the methodology synthesis"
[`summary9.md`](summary9.md) — boundary discipline, literal hinge, frozen decoder, causal-necessity validation. This is the bedrock; everything after is scaling and cross-document validation.

### "I want to understand the unsupervised baseline"
[`summary10.md`](summary10.md), §"Full-scale Delphi result" — the qualitative audit explaining mechanistically why mean F1 = 0.025: 71% of Delphi descriptions are corpus-tautological non-statements.

### "I want to see the failures"
The v8.19 scaling-run debacle (chunked Opus generation produced ~50 duplicate features; throughput collapsed under multi-EngineCore vLLM; ~$30-50 wasted compute) is summarized inline in [`changes.md`](changes.md) under the v8.19 entries and in the paper's "Failed loss configurations" paragraph.

### "I want to read the code"
Start at [`pipeline/run.py`](pipeline/run.py) (CLI dispatcher), then [`pipeline/config.py`](pipeline/config.py) (every flag), then the stage you care about (`inventory.py` → `annotate.py` → `supervised_hinge.py` → `evaluate.py`).

### "I want to read the loss derivation"
[`supervised_saes_hinge_loss.md`](supervised_saes_hinge_loss.md) — mentor's note. Three principled formulations (ReLU + hinge / JumpReLU + hinge / gated + BCE); we ship the first.

### "I want a per-version code change log"
[`changes.md`](changes.md) — every version tag from v1.0 through v8.21 with what changed in each file. Long but searchable.

---

## What changed across the timeline

A reviewer wanting a one-sentence-per-version delta:

- **v1 → v2**: gpt-oss-20b annotator → standardized API/local-annotator split; manual catalog → automated 8-feature test catalog.
- **v2 → v3**: 8 features → ~127; first reconstruction vs classification tension visible.
- **v3 → v4**: GPT-2 Small → Gemma-2-2B (validated cross-model portability).
- **v4 → v6**: phase 3 evaluation framework (sup vs pretrained head-to-head, causal validation à la Makelov).
- **v6 → v7**: frozen decoder default-on; amplification sweep; first cosine = 1.0 contract.
- **v7 → v8**: discovery loop ships; methodology retrenchment from "better features" to "calibration-honest classification."
- **v8 → v8.18**: boundary-discipline contract per leaf; literal hinge replaces hybrid as default; Delphi removed from supervised-arm gate (it was nerfing F1 by source-latent-faithfulness filtering).
- **v8.18 → v8.20**: scaling to 5K → 10K seqs; vLLM throughput tuning after the v8.19 dedup / EngineCore failures; min-support filter; corpus extension path.
- **v8.20 → v8.21**: 4B → 8B annotator; per-leaf exclusions in suffix; sequence-split eval added; HuggingFace release.

The full per-file delta table for any v8.x → v8.y is in [`changes.md`](changes.md), grep for the version tag.
