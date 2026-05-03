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
