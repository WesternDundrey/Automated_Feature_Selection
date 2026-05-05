# 10K-seq production run @ 104-feature curated catalog (v8.20.12)

**Date:** 2026-05-05
**Target:** GPT-2 Small @ layer 9, `blocks.9.hook_resid_pre`.
**Annotator:** Qwen3-4B-Base via vLLM v0.20.1, unconstrained decode (v8.20.11).
**Catalog:** 104 leaves curated from a 445-feature Opus-judge run (qwen-decidable filter: lexically-simple POS + surface + position-relative).
**Hardware:** 2× RTX 6000 Pro Blackwell (96 GB each), Granite Rapids host.

## Headline numbers (10000 seqs × 128 positions × 104 features = 33.2M decisions per shard)

| Metric | Supervised SAE | Linear probe | Notes |
|---|---:|---:|---|
| **Mean F1 (t=0)** | **0.554** | 0.474 | sup wins t=0 by **+0.080** (calibration-honesty property) |
| Mean calibrated F1 | 0.556 | **0.607** | probe wins calibrated by +0.051 (capacity asymmetry) |
| Mean oracle F1 | 0.554 | — | natural threshold ≈ optimal threshold |
| Mean AUROC | 0.899 | 0.939 | probe higher AUROC; SAE has tighter F1 at z=0 |
| **R² (full SAE)** | **0.9702** | — | 103+256 = 359 latents |
| Pretrained SAE R² (24,576 latents) | 0.9843 | — | reconstruction cost of supervision = **−0.0141 R²** for **68× fewer latents** |
| ΔR²(supervised slice) | **+0.9351** | — | sup carries 94% of recon |
| ΔR²(unsupervised slice) | +0.1455 | — | U adds 15% headroom |
| Mean cosine to target | **1.0000** | — | frozen-decoder, by construction (103/103 features) |
| Mean FVE (per-feature) | 0.186 | — | heterogeneous: see backbone subset below |
| L0 (sup, naive >0) | 13.30 | — | matches GT L0 = 13.34 |
| L0 (sup, calibrated) | 17.38 | — | calibrated overshoots GT by +30% |
| **Naive L0 ratio** | **0.997** | — | calibration-honest at production scale |

## Per-group F1 (by id prefix)

| Group | n | Mean F1 (t=0) | Mean cal F1 | Notes |
|---|---:|---:|---:|---|
| **days_of_week** | 1 | **0.712** | 0.707 | single-feature, near-perfect AUROC 0.999 |
| **months** | 1 | **0.702** | 0.730 | single-feature, AUROC 0.998 |
| **prepositions** | 5 | **0.652** | 0.674 | small lexical sets (`to/in/for/with/by/at`) |
| **punctuation** | 18 | **0.621** | 0.616 | surface tokens: comma, newline, hyphen, quote variants |
| **position** | 33 | **0.595** | 0.581 | left-context-relative ("after X", "cap after Y") |
| **morphology** | 11 | **0.564** | 0.561 | numeric/ordinal/cardinal + hyphen continuation |
| **articles** | 7 | **0.507** | 0.515 | `the` variants (the/a/an, with-context); the_after_comma/prep weaker |
| **copula** | 2 | **0.502** | 0.526 | `be_form` + contractions |
| **conjunctions** | 3 | **0.475** | 0.480 | and/or/coordinator |
| **discovery** | 22 | **0.425** | 0.434 | weakest group; many low-base-rate or context-richer features |
| **TOTAL** | **103** | **0.554** | **0.556** | |

## High-FVE backbone (FVE > 0.5)

19 features land in the high-FVE backbone — these are the cleanest concept-direction recoveries:

| Feature | F1 | FVE | n_pos |
|---|---:|---:|---:|
| punctuation.newline | 0.958 | 0.987 | 14,963 |
| punctuation.open_quote | 0.891 | 0.999 | 4,079 |
| articles.the | 0.849 | 0.999 | 4,377 |
| punctuation.after_quote_boundary | 0.736 | 0.902 | 152,026 |
| position.after_period_or_para | 0.731 | 0.903 | 144,249 |
| punctuation.period_end_quote | 0.707 | 0.912 | 127,677 |
| position.after_newline | 0.707 | 0.982 | 25,669 |
| punctuation.space_after_emdash | 0.695 | 0.566 | 146,525 |
| position.quoted_speech_start | 0.691 | 0.900 | 179,700 |
| position.after_endoftext | 0.679 | 0.995 | 11,389 |
| position.first_after_doc_start | 0.626 | 0.992 | 16,291 |
| position.after_close_angle | 0.601 | 0.578 | 96,264 |
| punctuation.close_quote | 0.603 | 0.958 | 76,233 |
| punctuation.open_quote_speech | 0.584 | 0.997 | 6,793 |
| position.cap_the_after_period | 0.563 | 0.990 | 23,711 |
| punctuation.period_after_paren | 0.482 | 0.972 | 49,794 |
| punctuation.close_angle | 0.480 | 0.990 | 20,073 |
| articles.the_lowercase | 0.740 | 0.651 | 18,365 |
| punctuation.close_paren_aside | 0.389 | 0.781 | 31,698 |

**Backbone mean F1 = 0.669** (19 features, +0.115 over full-catalog mean). These are surface/position features where the activation at positive positions IS the mean-shift direction.

## Per-feature standouts

| Class | Strongest features | Weakest features |
|---|---|---|
| **Surface** | `discovery.foreign_word` (F1=0.897), `punctuation.newline` (0.958), `articles.the` (0.849), `morphology.numeric_digit` (0.836), `morphology.lower_after_space` (0.805) | `articles.a_after_comma` (0.218), `position.pronoun_after_period` (0.187), `discovery.relative_after_comma` (0.188) |
| **Specific lexical** | `discovery.unit_measurement` (0.783), `morphology.cardinal_number` (0.763), `discovery.cap_after_endoftext` (0.687) | `discovery.or_after_may` (0.149), `discovery.to_after_according` (0.162) |
| **Position-relative** | `position.after_comma` (0.850), `position.lower_after_the` (0.777), `position.after_period_or_para` (0.731) | `position.after_open_bracket` (0.366), `position.connector_after_period` (0.368) |

## Reconstruction story

```
Full SAE (103 sup + 256 unsup):  R² = 0.9702
Without supervised slice:        R² = 0.0351   (drop = 0.9351)
Without unsupervised slice:      R² = 0.8247   (drop = 0.1455)
Pretrained SAE (24,576 latents): R² = 0.9843

→ Cost of supervision: −0.0141 R² for 68× fewer latents
→ Supervised slice carries 94% of reconstruction work
→ Top-3 reconstruction-relevant features (per ΔR²):
    articles.the              ΔR² = +0.135
    punctuation.open_quote    ΔR² = +0.102
    punctuation.newline       ΔR² = +0.016
   These three alone account for ~25% of the supervised slice's R² contribution.

3/103 features have ΔR² ≥ 0.01 (strong recon contribution)
5/103 features have ΔR² ≥ 0.001
Mean per-feature ΔR² = 0.00254
```

The "long-tail features barely participate in reconstruction" finding holds: the supervised SAE's reconstruction quality rests on a very small backbone, while the rest of the named features serve as crisp classifiers without significant recon load. This matches summary9's "FVE and causal effect partially decoupled" finding.

## Calibration honesty

```
Naive L0 ratio (predicted >0 / GT):    0.997   ← essentially perfect
Calibrated L0 ratio:                   1.302   ← overshoots by 30%
49/103 features fire within 0.01 of GT positive rate at the natural zero
Median |r@cal - r@gt|:                 0.011

Mean F1 t=0:    0.554   ← natural threshold
Mean F1 cal:    0.556   ← per-feature tuning gain: +0.002
Mean F1 val-promo (held-out half): 0.554
```

The natural threshold ≈ optimal threshold — calibration buys essentially nothing. This is the load-bearing property of literal hinge + frozen decoder + no pos_weight: per-feature scores are calibrated by construction, no per-feature threshold tuning needed.

## What this run validates (v8.20.x technical wins)

This 10K-seq run completed in ~3 hours wall-clock at ~5000 dec/s aggregate (2 shards × ~2500 dec/s). The throughput required all of v8.20.x's fixes:

* **v8.20.10 — bumped vLLM scheduler defaults**: `max_num_seqs=2048` (was auto-scaled to 512), `max_num_batched_tokens=65536` (was vLLM-default 16384). Smoke test isolated this as the prefill bottleneck.
* **v8.20.11 — dropped `allowed_token_ids` constrained decode**: 4.4× slowdown via vLLM's logits-processor path. Replaced with Python-side parsing of `output.token_ids[0]` against `{tok_0_set, tok_1_set}` (with leading-space variants). Parse-failure fallback rate well under 1% on Qwen3-4B-Base binary-question prompts.
* **v8.20.12 — extend-corpus uses multi-shard**: previously always single-shard, leaving the second GPU idle on a 2-GPU box. Now auto-shards via `_annotate_local_parallel`.
* Smoke test (`tools/vllm_smoke.py`): TEST 3-CON ablation showed 7021 → 1582 p/s when constrained decoding is active. Diagnostic infra that turned a hand-wavy "vLLM is slow" into a concrete fix in <10 minutes.

Without these fixes, the run was projected at 95+ hours wall-clock. The combined throughput multiplier from v8.20.10/11/12 was ~30-100× over the prior defaults.

## What's next

* **train + evaluate already complete** (this artifact). The 0.554 mean F1 at 10K seqs is consistent with summary9's 5K-seq production at U=256 (0.573), modestly lower because the curated 104-feature catalog skews harder toward boundary cases (no scaffold filler).
* **FVE backbone reporting** is the cleanest paper headline: 19 features with FVE > 0.5 hit 0.669 mean F1. This is the equivalent of summary9's "16-feature backbone at 0.677 cal F1" framing.
* **Probe comparison**: probe wins calibrated F1 by +0.051 but loses t=0 F1 by 0.080. Per the locked framing in `project_next_actions.md`: claim "supSAE achieves better description fidelity at the natural threshold," not "better features."
* **Causal validation** (`--step causal`) is the missing third leg — expected per-feature KL ablation analogous to summary9's 12/46 features with measurable causal effect. Not yet run on this artifact.

---

# Pilot result: Delphi auto-interp F1 = 0.034 vs supSAE F1 = 0.547 (v8.19.x)

**Date:** 2026-05-01
**Target:** GPT-2 Small @ layer 9, `blocks.9.hook_resid_pre`. Pilot scale: 500 sequences × (50 Opus features + 30 Delphi features), two-arm Type-1 native-pipeline F1 head-to-head.

## One-line headline

**At pilot scale, the supervised SAE on the Opus-designed catalog reaches median F1 = 0.547; real EleutherAI Delphi auto-interp on the same model's pretrained unsup SAE (`gpt2-small-res-jb`, 24,576 latents) reaches median F1 = 0.034.** Δ median = +0.513, a ~16× gap. This puts into serious question whether the projected ~28-hour full Delphi annotation pass is worth running before we understand what's making auto-interp fail this badly on its own grounded descriptions.

| arm | catalog | n | median F1 | mean F1 | cal mean F1 |
|---|---|---:|---:|---:|---:|
| sup | Opus 4.7 (50 features) | 54 | **0.547** | 0.576 | 0.573 |
| unsup | Delphi (30 latents, 1:1) | 29 | **0.034** | 0.050 | — |

(n = features with ≥ 5 test positives. The 54 sup count includes Opus's group entries + a few symmetry-completing leaves above the requested 50; 29/30 Delphi means one latent had < 5 positives in the held-out test slice.)

## What this is and isn't

**Type-1 native-pipeline definition (per `summary_saes_hinge_loss.md` mentor framing):**
- Sup arm: Opus designs description `d_c`; annotator labels `d_c` → `y_c`; supSAE trained on `y_c`; F1(supSAE feature `S_c` fires vs `y_c`) on held-out tokens.
- Unsup arm: Delphi describes pretrained unsup latent `U_j` as `d_j`; annotator labels `d_j` → `y_j`; F1(`U_j` fires vs `y_j`) on held-out tokens.

**Same held-out flat positions** for both arms (v8.19.3 fixed the previously-broken split alignment). Same annotator (Qwen3-4B-Base via vLLM, prefix-cached). Same model layer. Same corpus.

**What this F1 measures:** "does the latent's firing pattern agree with its own associated natural-language description on tokens it didn't see at training?"

**What this F1 does NOT mean:** "unsup SAEs are useless." It specifically means: real Delphi descriptions are poor predictors of their own latents' firing in the held-out set. That's a failure of the **post-hoc auto-interp pipeline**, not of the unsup latents themselves. A reviewer would correctly point out that:
1. Many `gpt2-small-res-jb` latents are known to be polysemantic; Delphi's description captures one mode and the latent fires in others.
2. Delphi's description was generated from top-activating contexts (high-activation extreme); the held-out F1 is dominated by the bulk of mid-activation tokens where the latent still fires but on different semantic content.
3. The Sonnet-4.6 explainer (cost-bounded) may produce noisier descriptions than Opus would; this is the cheaper variant in our pilot.

## Why the gap is plausible

The supervised arm has two structural advantages that DO show up in F1 even with no pipeline cheating:
- **Opus designs descriptions to be learnable.** Prefix-decidable, ≤ 10-word single-sentence, boundary-discipline contract. Opus is implicitly choosing the description so the supSAE can fit it. Delphi is post-hoc-describing whatever the unsup latent does.
- **The supSAE is trained on the labels.** Of course F1 on a label distribution it was trained against is high. The right comparison is: are the labels themselves coherent? Sup arm gets to choose; unsup arm gets a description fixed by post-hoc constraints.

The mentor's framing was honest about this: the F1 difference is "post-hoc unsup latent explainability vs trained supervised latent label fidelity." Different things, both useful, both honestly named. The gap is expected to be large; ~16× is consistent with the framing.

## What this changes about the plan

**Conservative read (the user's instinct):** if Delphi descriptions are this badly predictive of their own latents at pilot scale, the full ~28-hour Delphi annotation is mostly going to confirm "Delphi descriptions don't predict Delphi latents well at scale either," which is a paper claim but not a strong one. The unsup arm may not be informative enough to justify the compute.

**Counter-read:** the pilot is small (30 features, 500 sequences). Some of those 30 latents may be:
- Genuinely uninterpretable (dense, polysemantic) → can't be saved by scale
- Fine but Delphi's description was off → would tighten with more activations or a better explainer
- Rare features whose held-out positive count is < 5 → noise

We don't know the split without a per-feature breakdown.

## Open questions before scaling Delphi

These would all be cheap (≤ 1 hr) and would sharpen the decision:

1. **Per-feature F1 distribution.** Read `pipeline_data_pilot_unsup/unsup_f1.json` and bucket the 29 evaluated features. Is the 0.034 median pulled down by 25 zeros + 4 strong, or is it 29 features all near 0?
2. **Switch Delphi's explainer Sonnet 4.6 → Opus 4.7.** A pilot-only, ~$3 cost. If median jumps from 0.034 → ~0.20+ with a better explainer, scale matters; if it stays flat, the unsup latents themselves are the bottleneck, not Delphi's descriptions.
3. **Re-run Delphi with `fuzz` scorer instead of `detection`.** Tests whether the per-latent fidelity changes when the scorer prompt is more aggressive. ~1 hr.
4. **Spot-check 5 Delphi descriptions by hand.** Read them and the corresponding latent's top-K contexts. Are the descriptions plausible, or do they look like Delphi guessing? This is 30 minutes of human review and tells you more than another scaling run.

## Recommended decision tree

- **If question 1 is "all features near 0":** unsup auto-interp is broken at this layer/SAE, not at this scale. Don't scale Delphi; the headline becomes "auto-interp on `gpt2-small-res-jb` produces descriptions whose latents do not predict them on held-out tokens at any scale." That's a publishable negative result.
- **If question 1 is "bimodal: 5-10 strong, rest near 0":** there are real interpretable latents but most aren't. Selecting them post-hoc would be cheating, but reporting the bimodality + showing the strong ones would be a more nuanced paper claim.
- **If questions 2 or 3 lift the median materially:** scale matters; the explainer / scorer choice is load-bearing; rerun the full pipeline with the better setup.
- **Default if all four come back unclear:** keep the pilot as the unsup-arm result; ship the headline at "Δ = +0.513 at pilot scale; full-scale Delphi run skipped because pilot showed (the diagnostic finding here)."

## Scaling-run status (separate)

The `pipeline_data_scaling/` 500-feature × 50K-sequence run is on a different track. Not affected by the pilot result; still worth running. Final command sequence locked at:

```bash
git pull   # get dd09c00 (CLI flags) + 68ac070 (annotate.py docstring fix)

rm -rf pipeline_data_scaling/

export FLAGS="--output_dir pipeline_data_scaling \
  --n_sequences 50000 \
  --opus-n-features 500 \
  --shortlist-size 1500 \
  --no-scaffold \
  --min-support 500 \
  --position-subsample-k 64"

python -m pipeline.run $FLAGS --step shortlist
python -m pipeline.run $FLAGS --step opus-catalog
python -m pipeline.run $FLAGS --step annotate    # ~2-3 days on 2×5090s
python -m pipeline.run $FLAGS --step train       # ~6-8 hr
python -m pipeline.run $FLAGS --step evaluate
```

Wall-clock ~3 days; API cost ~$15-20 for the Opus call.

## Process note

This pilot ran on the v8.19.x Delphi-vs-Opus architecture: real EleutherAI Delphi v0.1.3 cloned + installed via `install.sh` Compartment 4; Opus 4.7 catalog generator with auto-downshift on 1M-context overflow; literal-mentor hinge defaults (zero-margin, no pos_weight, frozen decoder); position-subsample disabled at pilot scale; min-support filter not active at pilot scale.

The 5K-seq matched-arm Type-1 vs Type-2 (oracle_unsup) head-to-head was always the rigorous comparison; the pilot was a go/no-go gate. **The gate fired this exact concern: should we scale.** Answer: not before the four diagnostic questions above. Each is 15-60 min and would change the right scaling decision.

---

## Full-scale Delphi result (2026-05-03)

Followed the pilot with the full unsup-arm scoring at 5000 sequences against the Delphi catalog of 91 leaves (`pipeline_data_compare_unsup/unsup_f1.json`). Wall-clock 24.2s — only the F1 readout step, since Delphi's annotations were already produced.

| arm | catalog | n | mean F1 | median F1 | scale |
|---|---|---:|---:|---:|---|
| sup (summary9 production baseline) | Sonnet, 46 features (U=256, hinge, frozen) | 46 | **0.573** (supT0) / **0.564** (supCal) | — | 5000 seq |
| **unsup (this run, real EleutherAI Delphi)** | **gpt2-small-res-jb, 91 latents** | **91** | **0.025** | **0.010** | **5000 seq** |

**Δ mean ≈ +0.55 (~23× gap)** at matched corpus scale. Decisively confirms the pilot: real Delphi auto-interp on `gpt2-small-res-jb` produces descriptions whose latents do not predict them on held-out tokens, and **the gap does not close with more sequences**. The 5000-seq scale with 91 latents is more decisive than the pilot's 500-seq × 30-latent slice.

The natural match for this comparison is the v8.18.39 production run (summary9). For a strict 91-feature sup-arm at this scale, `pipeline_data_compare_sup/evaluation.json` should hold the matched-count number; verify with `mean_f1` / `median_f1` from that artifact when checking apples-to-apples.

### Per-feature distribution (open question)

Mean=0.025 / median=0.010 is highly skewed. Whether this is "all 91 features near zero" or "bimodal: a few strong, rest dead" matters for the paper claim. Bucket the per-feature F1 from `unsup_f1.json` to know which.

If all-near-zero: the unsup-arm headline is "real Delphi descriptions don't predict their own latents at any scale on this SAE/layer" — a publishable negative result.
If bimodal: report the bimodality and the strong-subset; selecting them post-hoc is cheating, but acknowledging the heterogeneity is honest.

### Honest framing for the paper

- "Real EleutherAI Delphi auto-interp on `gpt2-small-res-jb` (24,576 latents, layer 9) produces descriptions whose latents achieve median F1 = 0.010 / mean F1 = 0.025 on held-out tokens, at 5000-sequence scale."
- supSAE wins by training against the labels; the pretrained SAE has known polysemy at this layer (multiple modes per latent; Delphi captures one).
- Frame as "supSAE achieves better description fidelity," NOT "supSAE makes better features." The selection-freedom asymmetry (Opus designs descriptions to be learnable; Delphi describes whatever the unsup latent does post-hoc) IS the methodology being tested; not normalized away.

Cost: ~$5 in Sonnet 4.6 for the original Delphi explainer pass; $0 for the F1 readout (24.2s).

### Qualitative catalog audit — why the F1 is 0.025

Manual audit of the 100-latent Delphi catalog (`delphi_catalog.json`) explains the F1 mechanistically. The catalog overwhelmingly produces descriptions of the form "long passages of text from diverse sources with no consistent pattern" — non-falsifiable strings that apply to virtually every position in OpenWebText. The detection scorer cannot separate positives from negatives because the description is a tautology over the corpus.

| Bucket | Count | Pattern |
|---|---:|---|
| **Explicit non-statements** | **~65/91 (71%)** | Phrases like "no consistent pattern", "no clear unifying", "no specific pattern", "essentially arbitrary", "uniformly activated regardless of content" |
| Vague topical flavor | ~22/91 (24%) | Low-quality web / mid-sentence informational / blog-forum-noise — somewhat predictive but too broad to score |
| **Actually specific & actionable** | **~5/91 (5%)** | The minority where Delphi recovered a learnable concept |
| Explicit parse failure | 1 | `latent_14940`: "Explanation could not be parsed" |
| Explicit "no pattern at all" | 1 | `latent_8357`: "no particular tokens or patterns are activated" |

The 5/91 actionable Delphi descriptions worth preserving as evidence Delphi *can* recover concepts when the underlying latent is monosemantic enough:

- `latent_6141` — "begin with a special or unusual character (Unicode artifact, punctuation, or formatting symbol) at the very start of the passage"
- `latent_16328` — "fully enclosed within special formatting characters such as quotation marks or newline symbols"
- `latent_21566` — "formal or institutional speech contexts, particularly involving legal rights, civil liberties, or religious/political rhetoric"
- `latent_11270` — "multi-line or multi-sentence text spans from structured or technical contexts such as code, academic references, or formal writing"
- `latent_9222` — "mid-sentence or mid-passage, typically following a conjunction or article, representing continuation of a broader context"

This is the **5% of `gpt2-small-res-jb` layer-9 latents that are post-hoc-describable as token-level concepts**. The other 95% are either polysemantic (multiple modes per latent — Sonnet correctly recognizes "no pattern") or activate on diffuse context properties not reducible to a single yes/no question.

**The catalog IS the explanation.** When 71% of features are described with phrases the corpus universally satisfies, F1 ≈ base-rate; mean=0.025 is approximately what you'd get from random predictions weighted by feature base rate. This isn't Delphi failing as a tool — Sonnet is being honest about what it sees. It's the **post-hoc auto-interp paradigm itself** failing on this SAE/layer.

The supSAE methodology sidesteps this entirely: Opus designs descriptions to be **learnable as token-level YES/NO questions** (boundary-discipline contract: positive_examples + negative_examples + exclusions per leaf, prefix-decidable, ≤10-word single-sentence). The descriptions are constrained to be classifiable before the SAE is even trained. The supervised arm's higher F1 is not a property of the trained SAE — it's a property of the description-design constraint.

This is the cleanest available qualitative evidence for the methodological contribution: **catalog quality is the bottleneck**, post-hoc auto-interp produces uncatalogable descriptions for the bulk of unsup latents, and the supSAE pipeline replaces that bottleneck with a constrained-design step where every description is required to be learnable.
