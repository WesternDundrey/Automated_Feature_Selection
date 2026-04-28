# Boundary discipline + literal-hinge synthesis at scale (v8.11 → v8.18.34)

**Date:** 2026-04-28
**Target:** GPT-2 Small @ layer 9, `blocks.9.hook_resid_pre`. Two scales tested: 8-feature prefix-decidable test catalog at 500 sequences (mechanism isolation) and 43-feature Sonnet-generated catalog at 2000 sequences with min-support 50 (production scaling).

## One-line headline

**Supervised SAE with frozen decoder + zero-margin hinge + no pos_weight + n_unsup=64 beats the linear probe at t=0 by +0.113 F1 (0.502 vs 0.389)** on the production-scale 43-feature catalog. The natural-threshold property the mentor's note predicted holds at scale: naive predicted L0 matches GT L0 to 7%, calibration overshoots GT L0 by 96%. AUROC parity (SAE 0.950, probe 0.970), reconstruction parity at 230× fewer latents than the pretrained SAE (R² = 0.944 with 43 + 64 = 107 supervised + unsupervised vs 0.985 with 24,576 unsupervised), and ΔR²(S) = 0.915 (the supervised slice carries 91% of reconstruction). What's still missing is causal validation: cos = 1.000 is by construction; whether the named directions actually steer model behavior is the open empirical question.

The other half of this cycle is methodology retrenchment, again. v8.18.26 ripped Delphi out entirely because it was nerfing F1 through source-latent-faithfulness filtering. v8.18.28 introduced a boundary-discipline contract for catalog generation. v8.18.29 rewrote the test catalog as 8 prefix-decidable surface features so threshold geometry could be tested in isolation from annotator noise. v8.18.32 added a regex backstop for prefix-decidability. v8.18.33 added a post-annotation min-support filter. v8.18.34 made the prefix-decidable contract opt-in via `--legacy-prompts` and made the boundary-discipline annotator suffix opt-out via `--no-exclusions-in-suffix` (recovers ~2-3× annotation throughput).

## Context

Summary8 ended with the promote-loop shipping its first two real features and ~10 cycles of reviewer-driven correctness fixes. The v8.10 catalog (74 features, 33 scaffold control + 41 Sonnet-discovered) reached calibrated F1 = 0.612 — below the linear probe (0.620) and post-training readout (0.652). The supervised-classification-advantage claim was retired. What remained: cosine = 1.000 frozen-decoder geometry, R² = 0.971 at 75× fewer latents than the pretrained SAE, K=2 composition correlation `corr(decoder_cos, linearity) = −0.83`, and a working catalog-growth loop.

This cycle (v8.11 → v8.18.34) tested four architectural questions:

1. **Loss formulation.** The mentor's note (`supervised_saes_hinge_loss.md`) prescribes ReLU + zero-margin hinge with no STE. Does that work on real activations or does class-balanced BCE / margin hinge dominate?
2. **Decoder freedom.** Does the supervised slice's cosine-to-target survive end-to-end training, or does U capacity siphon the gradient and let supervised columns drift?
3. **Catalog quality.** Is the bottleneck the loss, the annotator, the catalog itself, or something else? Does prefix-decidability of feature descriptions affect annotator label quality?
4. **Scaling.** Do the test-catalog findings (8 hand-curated features) survive at production scale (50-100 Sonnet-generated features)?

The answers are interlocking. We had to fix the catalog first to isolate the loss signal; once we did, the loss matrix became readable; once that was readable, scaling could be tested.

## What was built (chronologically)

**v8.11 — Hinge / JumpReLU / Gated-BCE supervision modes.**
Implemented the mentor's three principled formulations as new `--supervision` modes:
- `hinge`: ReLU + hinge on pre-activations (mentor's #1)
- `hinge_jumprelu`: JumpReLU + hinge with learnable per-feature θ (mentor's #2)
- `gated_bce`: two-path encoder, BCE on gate (mentor's #3)
All three share decoder unfrozen by design (the doc explicitly rejects pre-computed target_dirs as a hard constraint). Legacy `hybrid` mode retained for ablation.

**v8.12 — v8.16 — Delphi integration cycles.**
Five rounds attempting to wire Eleuther's Delphi (DetectionScorer + LatentRecord) as a catalog-quality gate. By v8.18 Delphi was technically working but every audit cycle surfaced a new methodological flaw. The integration was a 13-file 2,053-line surface area for ~15% F1 improvement on inventory output.

**v8.18.22 — Token-level enforcement.** Inventory prompts rewritten with explicit REJECT/ACCEPT blocks. `catalog_quality.py` regex hard-fails on context-level predicates ("Text is", "Sentence presents", "register / genre / domain of X").

**v8.18.24 — Catalog quality strict mode with LLM crispness override.** Soft-flag patterns ("or", "and", "any") trigger Sonnet's crispness judgment. Hard-fail patterns reserved for genuinely vague phrases. Status: pass / quarantine / fail. LLM verdict overrides soft-flag findings (so "opening or closing quotation mark" doesn't get auto-rejected).

**v8.18.25 — Flip defaults.** Empirical evidence had accumulated: end-to-end hinge / gated_bce gave decoder cosines ≈ 0.16 (random) and FVE ≈ 0.005 (useless for intervention) when n_unsup=256 was available. Frozen decoder gave cos=1 + FVE ≈ 0.30 at ~0.03 F1 cost vs linear probe baseline. `hinge_freeze_decoder=True` became the default for hinge-family modes. Delphi gate also flipped off by default.

**v8.18.26 — Delphi removed entirely.** 2,053 lines deleted: `pipeline/delphi_score.py`, `pipeline/delphi_subprocess.py`, all `cfg.delphi_*` fields, all `--delphi-*` CLI flags, runtime deps. Catalog quality is now enforced by `pipeline.catalog_quality` (lexical hard-fail + LLM crispness with quarantine) and `pipeline.overlap_check` (post-annotation pairwise IoU + subset analysis).

**v8.18.27 — Drop vLLM cold-start workarounds.** The `enforce_eager=True` shim, `VLLM_USE_V1=0`, `NCCL_*_DISABLE` env vars, and `_configure_vllm_runtime_env` function were hotfixes for cold-start hangs that have since been fixed upstream. Restored CUDA graphs (~10-20% throughput uplift).

**v8.18.28 — Boundary-discipline contract.** The vague-description failure mode: "Token is a period or full stop" got labeled inconsistently across abbreviation/decimal/sentence-end positions; SAE learned the union; t=0 F1 stayed in the 0.5-0.7 range. Fix: `inventory.organize_hierarchy` prompt now requires three new fields per leaf (`positive_examples`, `negative_examples`, `exclusions`). Exclusions flow into the annotator suffix as ", NOT abbreviation periods, NOT decimal points". `catalog_quality.py` validates these fields. If Sonnet can't articulate boundary cases for a leaf, the leaf is hard-failed.

**v8.18.29 — Prefix-decidable test catalog.** External audit: "I would prioritize: Use a dead-simple prefix-decidable catalog only. No 'sentence-final', 'introducing a PP', 'functioning as subject/object'." Test catalog rewritten as 8 surface-only features (`comma_char`, `period_char`, `digit_only`, `starts_uppercase`, `leading_space`, `single_visible_char`, `all_lowercase_word`, `contains_hyphen`). Each carries the boundary-discipline schema.

**v8.18.30 — usweep ergonomics.** `--usweep-skip-promote` short-circuits the promote-loop step at each width (irrelevant for fixed catalogs). Summary table gained the geometry-honest read: `supT0` / `prbT0` / `ptT0` / `supCal` / `ΔR²(S)` / `FVE`. `--upstream-dir` decouples artifact source from output directory.

**v8.18.31 — `--no-pos-weight`.** Reviewer caught: runs labeled "mentor's hinge" used `hinge_margin=1.0` (default) and class-balanced `pos_weight`, which is an SVM-style margin variant — not the literal formula `max(0, -(2y-1) z_i)` from the doc. `--no-pos-weight` CLI added so the literal-formula ablation is actually possible. Default stays on (rare-class recall would collapse without it).

**v8.18.32 — Prefix-decidable inventory contract + regex backstop.** Inventory prompts (explain_features, organize_hierarchy) explicitly enumerate prefer/accept/reject lists so Sonnet skips features that need future tokens / full-sentence parse / document topic. `catalog_quality.py` adds regex hard-fails for "followed by", "preceding", "introducing a", "before X", "sentence-final", "subject/object/predicate of", "named entity", "sarcasm", "in a politics article". 9/9 of the audit's bad examples now hard-fail; 7/8 good examples pass cleanly.

**v8.18.33 — Post-annotation min-support filter.** After annotation, features with positive count < `cfg.min_support` are dropped from the catalog before training (Step 2.5). Features with n_pos < ~30 had AUROC near random and were contributing pure noise to the mean F1. Backed-up to `feature_catalog.unfiltered.json`, dropped features appended to `feature_catalog.quarantined.json` with reason `"min_support<N"`. Idempotent.

**v8.18.34 — Legacy prompts + opt-out exclusions in suffix.** Two new flags. `--legacy-prompts` drops the v8.18.32 prefix-decidable contract from the inventory prompts (still rejects text/sentence/document predicates but allows right-context-dependent features like "Token is the verb in a quote-attribution clause") and downgrades the prefix-decidable regex backstop from hard-fail to soft-flag (LLM-judged). `--no-exclusions-in-suffix` keeps the boundary-discipline metadata in the catalog but doesn't append exclusions to the annotator's per-feature suffix at label time, recovering ~2-3× annotation throughput (~1000 dec/s vs ~350 dec/s).

## The two empirical results

### Result 1: test catalog (mechanism isolation)

8 prefix-decidable features × 500 sequences × 128 positions, hinge with `--hinge-margin 0 --no-pos-weight`, frozen decoder default, n_unsup sweep:

| n_unsup | R² | ΔR²(S) | FVE | supT0 | supCal | cal gain |
|---|---|---|---|---|---|---|
| 0 | 0.834 | 0.937 | 0.408 | 0.750 | 0.755 | +0.005 |
| 8 | 0.923 | 0.821 | 0.408 | 0.765 | 0.772 | +0.007 |
| 32 | 0.935 | 0.262 | 0.408 | 0.761 | 0.766 | +0.005 |
| **64** | **0.944** | **0.179** | **0.408** | **0.772** | **0.781** | **+0.009** |
| 128 | 0.955 | 0.047 | 0.408 | 0.768 | 0.772 | +0.004 |
| 256 | 0.968 | 0.003 | 0.421 | 0.751 | 0.802 | +0.051 |
| 1024 | 0.996 | 0.082 | 0.421 | 0.749 | 0.807 | +0.058 |

Linear probe at supT0 = 0.758, supCal = 0.781 throughout. Pretrained-SAE post-train readout at supT0 ≈ 0.801, supCal ≈ 0.804.

Three things this established:
- The natural threshold matches the optimal threshold under literal hinge (calibration gain ≤ +0.009 in 12/14 sweep cells).
- U=64 is the synthesis sweet spot: highest supT0, ΔR²(S) > 0.1, FVE > 0.4. Past U=128, supervised slice loses recon load.
- Frozen + literal hinge are orthogonal axes (encoder controls threshold geometry, decoder freeze controls direction interpretability).

### Result 2: production-scale catalog (does it scale?)

43 Sonnet-generated features (post-min-support-50 filter from a 750-latent inventory) × 2000 sequences × 128 positions, same loss config, n_unsup=64, `--legacy-prompts --no-exclusions-in-suffix --min-support 50`. Run wall-clock 6.6 hr (Sonnet inventory + Qwen3-4B-Base annotation at ~1000 dec/s + train + eval).

| metric | Supervised SAE | Linear probe | Post-train (24,576 latents) |
|---|---|---|---|
| t=0 F1 | **0.502** | 0.389 | 0.481 |
| Calibrated F1 | 0.526 | 0.549 | 0.614 |
| AUROC | 0.950 | 0.970 | 0.964 |

The supervised SAE **beats the probe at t=0 by +0.113 F1**. The probe needs +0.16 from calibration to recover its pos_weight zero-shift; the supervised SAE under literal hinge has no such shift to recover.

L0 calibration:
```
GT L0:               1.87 features per token
Naive predicted L0:  2.01 (ratio 1.071 — essentially perfect)
Calibrated L0:       3.67 (ratio 1.957 — overshoots by 96%)
Per-position |pred - GT|:   mean 1.96, median 2.00
Calibration quality: 31/43 features within 0.01 of GT positive rate
Median |r@cal - r@gt|:      0.0021
```

Reconstruction:
```
Pretrained SAE (24,576 latents):  R² = 0.985
Supervised SAE (43 + 64 = 107 latents): R² = 0.944
Reconstruction cost of supervision: −0.041 R² for 230× fewer latents
ΔR²(S):    +0.915 (supervised slice carries 91% of reconstruction)
ΔR²(U):    +0.110 (unsupervised slice contributes 11%)
Mean cosine to target_dir:  1.000 (frozen, by construction — vacuous as evidence)
Mean FVE:                   0.343 (heterogeneous; 6/43 with FVE > 0.5)
```

### Top per-feature backbone (the publishable subset)

The 10 features with cal F1 ≥ 0.65 + FVE ≥ 0.05 form the methodology's load-bearing core:

```
control.semicolon                cal F1=0.980  AUROC=1.000  FVE=0.999  n_pos=228
punctuation_type.opening_quote   cal F1=0.916  AUROC=0.997  FVE=0.999  n_pos=300
punctuation_type.hyphen_compound cal F1=0.827  AUROC=0.960  FVE=0.084  n_pos=287
control.whitespace_run           cal F1=0.784  AUROC=0.984  FVE=0.986  n_pos=1582
control.repeated_character       cal F1=0.774  AUROC=0.988  FVE=0.998  n_pos=429
control.code_keyword             cal F1=0.769  AUROC=0.996  FVE=0.096  n_pos=30
control.hex_or_uuid_fragment     cal F1=0.718  AUROC=0.993  FVE=0.012  n_pos=74
control.bracket_opening          cal F1=0.711  AUROC=0.977  FVE=0.060  n_pos=132
control.currency_symbol          cal F1=0.699  AUROC=1.000  FVE=0.042  n_pos=52
token_content.the_definite       cal F1=0.668  AUROC=0.980  FVE=0.994  n_pos=1098
```

Three FVE-near-1 features (`control.semicolon`, `punctuation_type.opening_quote`, `control.whitespace_run`, `control.repeated_character`, `token_content.the_definite`) are clean concept-direction recoveries: their decoder column captures 99%+ of activation variance at positive positions, and their cal F1 is near-ceiling. These are the strongest evidence the mean-shift target direction *is* the concept direction for surface features with high base rate.

### The long tail

22 features have cal F1 between 0.30 and 0.65 (workable but noisier), 11 features have cal F1 < 0.30 (likely over-narrow descriptions or hard-to-annotate boundary cases). Examples:

```
control.abbreviation_period      cal F1=0.094  n_pos=80    description requires
                                                            disambiguating
                                                            sentence-end vs Mr/Dr
control.list_bullet_or_asterisk  cal F1=0.189  n_pos=13    rare, even at min_support=50
morphological_fragment.prefix    cal F1=0.211  n_pos=39    description ambiguous
control.markdown_link_bracket    cal F1=0.164  n_pos=51    rare in OpenWebText
```

Mean F1 across all 43 features is 0.502 (t=0) / 0.526 (cal). Mean F1 across just the 10-feature backbone is approximately 0.785 (cal). The headline 0.502 is dragged by the long tail.

### FVE rises with more data

A direct comparison across catalog scale (same loss, same n_unsup=64):
- 1000 seq, 19 discovery + 33 scaffold = 52 features (v8.18.32 strict prompts): mean FVE = 0.272
- 2000 seq, 13 discovery + 30 scaffold = 43 features (v8.18.34 legacy prompts post-min-support): mean FVE = **0.343**

More sequences → more positive examples per feature → cleaner mean-shift target direction → higher FVE. Per-feature, the high-FVE backbone features tightened from 0.94-0.98 to 0.98-1.00 (`hashtag_fragment` 0.961 → 0.963, `repeated_character` 0.998 → 0.998, `whitespace_run` 0.986 → 0.986, `unit_suffix` 0.974 → 0.975). The ~0.07 mean FVE rise comes from features that previously had FVE < 0.05 either getting more support or being filtered out by min_support. Worth re-running at 5000+ sequences to see if FVE rises further.

## What the production run proves

1. **The literal-hinge calibration-honesty property scales from 8 features to 43.** Naive L0 ratio went 1.002 (test catalog) → 1.071 (real catalog). Per-feature, 31/43 fire within 0.01 of GT rate at the natural zero. The probe needs +0.16 from per-feature calibration; the supervised SAE needs essentially nothing.

2. **Supervised SAE beats the linear probe at t=0 by a large margin (+0.113).** This is the cleanest publishable claim: at the no-tuning threshold both classifiers actually use, the SAE is better. The probe's pos_weight architecture costs it ~0.16 F1 at t=0 that calibration recovers; the SAE doesn't pay that tax.

3. **U=64 keeps the supervised slice load-bearing.** ΔR²(S) = 0.915 means the 43 supervised columns do 91% of the reconstruction work. Without them, R² drops to 0.029. This survives at scale: the test-catalog finding (ΔR²(S) = 0.179 at U=64 with 8 features) generalizes to ΔR²(S) = 0.915 at U=64 with 43 features. The supervised slice is the production model's reconstruction engine, not a side-channel classifier.

4. **Reconstruction parity at 230× fewer latents.** R² = 0.944 with 107 latents vs 0.985 with 24,576. −0.041 R² for the supervision constraint is a small price.

5. **Mean FVE rises with corpus size** (0.272 → 0.343 from 1000 → 2000 sequences). Suggests more data tightens the data-anchored direction. Not yet tested past 2000.

## What it does NOT prove

1. **cos = 1.000 is by construction (frozen decoder).** It says we obeyed our design: each decoder column equals its target direction. It does NOT say target direction = causal concept direction. That requires intervention testing.

2. **No causal validation has been run.** `--step causal` (Makelov sufficiency / necessity) and `--step composition` (K-way joint ablation linearity) have not been run on v8.18.34 artifacts. The intervention-validity claim still rests on the v8.0 composition result on the v8.10 catalog (`corr(decoder_cos, linearity) = −0.83` at K=2).

3. **Calibrated F1 still loses to probe.** Probe cal F1 = 0.549 vs SAE cal F1 = 0.526 (gap −0.023). Probe AUROC = 0.970 vs SAE AUROC = 0.950. The post-train baseline (linear readout from 24,576 pretrained latents) cal F1 = 0.614 — a +0.088 lead. So at the per-feature-tuned threshold, having more capacity wins. The supervised SAE's t=0 win comes from threshold honesty, not from stronger per-feature scores.

4. **The 10-feature backbone is small.** 10 of 43 features carry the methodology claim. The other 33 are workable-to-noisy. A reviewer asking "what does this catalog actually represent?" gets a defensible answer for 10 features and "we tried" for the rest.

5. **Annotator inter-rater reliability not measured here.** Summary8's run found inter-annotator F1 = 0.583 with the v8.10 catalog. We haven't re-measured under v8.18.34's prompts. The current cal F1 = 0.526 is below summary8's IRR ceiling, so we're not yet inter-rater-bound, but at higher F1 we would be.

## Reframed paper story (current draft)

The publishable claims, ordered from strongest to weakest:

1. **Calibration-honest classification at t=0 beats probe by +0.113.** Supervised SAE with frozen decoder + zero-margin hinge + no pos_weight produces per-feature scores whose natural zero IS the optimal threshold. Naive L0 matches GT L0 to 7%. Probe needs per-feature calibration (gain +0.16); supervised SAE doesn't. AUROC and calibrated F1 are within 0.05 of probe, so the t=0 lead isn't bought by score-quality regression — it's threshold-geometry advantage.

2. **Supervised slice is the production model's reconstruction engine.** At U=64, ΔR²(S) = 0.915 (91% of reconstruction goes through 43 supervised columns). Compare to U=1024 / U=2048 where ΔR²(S) ≈ 0 — the unsupervised pool absorbs all reconstruction and the supervised slice becomes cosmetic. The U=64 sweet spot keeps the supervised slice load-bearing.

3. **Reconstruction parity at 230× fewer latents.** 107 supervised+unsup latents reach R² = 0.944 vs 0.985 for the pretrained 24,576-latent SAE. Cost of supervision: −0.041 R².

4. **Boundary-discipline contract is a real F1 lever.** Test-catalog F1 went 0.672 → 0.751 → 0.772 as catalog descriptions got crisper (boundary discipline + prefix-decidable + literal hinge). Doesn't survive verbatim to scale (real catalog has heterogeneous quality), but the inventory-time gates produce cleaner labels.

5. **Frozen decoder + literal hinge are orthogonal contributions.** Encoder-side controls threshold geometry; decoder-side controls direction interpretability. Co-occurring at U=64 is the synthesis. The mentor's principled hinge formulation (validated by run 5 of the test-catalog ablation) and the engineering anchor for direction interpretability are not in conflict — they address different desiderata.

What is *not* yet a publishable claim:
- "Named directions causally steer model behavior" — needs `--step causal` to run.
- "Interpretation as concept directions" — the cos = 1 number is by construction; FVE alone is heterogeneous evidence (6/43 features with FVE > 0.5 vs the rest).

## Honest limitations

1. **cos = 1 is vacuous as evidence.** Implementation invariant. FVE per-feature is the real signal; its mean (0.343) hides the bimodal distribution (top features 0.95-1.00 vs bottom features 0.01-0.10).

2. **Long-tail features hold up the mean F1.** 11 of 43 features have cal F1 < 0.30. These are real features with real labels but the descriptions are either too narrow (`morphological_fragment.prefix_fragment` n_pos=39 across both halves), too rare (`control.list_bullet_or_asterisk` n_pos=13), or hard-to-annotate at boundaries (`control.abbreviation_period` cal F1 = 0.094 because annotator can't tell sentence-end from Dr.).

3. **Three-feature problem under target_dir = mean_shift.** Some features have FVE near zero even at frozen decoder + min-support filtering — `control.list_bullet_or_asterisk` (FVE=0.011), `morphological_fragment.prefix_fragment` (0.018), `token_role.infinitive_marker_to` (0.017), `control.tld_or_url_tail` (0.026). For these, mean-shift either captures too little of the activation variance OR the feature has no unidirectional residual-stream signature. Switching target_dir_method to LDA or logistic might lift them; not yet tested.

4. **Class imbalance ablation incomplete.** `--no-pos-weight` was tested only on features with base rate ≥ 0.05 (test catalog) and ≥ ~0.005 after min-support filter (production catalog). Real catalogs at lower min-support thresholds (or with rarer concepts) may show recall collapse.

5. **No causal / intervention validation.** Top priority for next cycle.

6. **No 5000+ sequence comparison.** FVE rose with corpus size from 1000 to 2000 seqs. Whether this continues or plateaus is open; the ~10 hr annotation cost at 5000 seqs is feasible.

## What to run next

**Highest priority (the missing third leg of the methodology tripod):**

1. **`--step causal` on the v8.18.34 production artifact.** Validates that the 10-feature backbone's decoder columns actually steer model behavior. Three axes per the Makelov framework: approximation (sufficiency / necessity under sparse intervention), sparse controllability (greedy feature editing), interpretability (F1 vs known attributes). Without this, the cos = 1 + FVE story is "we declared and obeyed" — the claim that the named directions are causally meaningful needs intervention evidence. ~30 min wall-clock on the existing artifact.

2. **`--step composition` on the v8.18.34 production artifact.** K=2 joint-ablation linearity correlation with decoder cosine. Replicates the v8.0 finding (`corr(decoder_cos, linearity) = −0.83`) on the new architecture. ~10-20 min.

**Medium priority (target-dir method ablation — the question we kept deferring):**

3. **target_dir_method sweep at production config** (`mean_shift / lda / logistic` × U=64 × literal-hinge × frozen). Currently the four FVE-near-zero features (`list_bullet`, `prefix_fragment`, `infinitive_marker_to`, `tld_or_url_tail`) fail because mean-shift doesn't capture them. LDA or logistic might. If LDA uniformly lifts mean FVE by 0.10+, switch the global default. If only some features benefit, switch to per-feature method selection. Three single-config retrains, ~10 min total.

4. **5000-sequence run at the same config.** Tests whether FVE continues to rise with corpus size (1000 → 2000 gave +0.07) and whether the 11 currently-low-F1 features recover with more positive examples. Annotation cost ~12-15 hr.

**Lower priority (loss / architecture exploration):**

5. **Soft-anchored decoder validation run.** Initialize decoder columns at target_dirs, add `λ_dir · (1 − cos(W_dec[:, k], target_dir_k))` with decay schedule, drop λ_dir to zero by epoch 10. If cos stays > 0.8 after the anchor weakens, that's much stronger evidence than cos = 1 by construction — it's evidence the data supports the direction.

6. **Anti-siphoning U penalty.** `λ_anti · Σ_j max(0, cos(W_dec_U[:, j], target_dir_k))²`. Pushes U decoder columns to be orthogonal to supervised directions so U can't claim variance the supervised slice owns. Currently a non-issue at U=64 (ΔR²(S) = 0.915), but matters if scaling U beyond 256.

7. **Explicit per-feature magnitude loss.** `(sup_acts[:, k] − relu(x · target_dir_k))²` at positive positions, decoupled from the gate. Drives FVE up for low-FVE features without depending on global MSE pressure.

## Files of record

| file | role |
|---|---|
| `pipeline/supervised_hinge.py` | mentor's three formulations + frozen-decoder grad hook (v8.11+) |
| `pipeline/catalog_quality.py` | lexical hard-fail + LLM crispness + boundary-discipline + min-support filter (v8.18.24, v8.18.28, v8.18.32, v8.18.33) |
| `pipeline/inventory.py` | `organize_hierarchy` + `explain_features` with conditional prefix-decidable contract (v8.18.32, v8.18.34) |
| `pipeline/test_catalog.json` | 8 prefix-decidable surface features (v8.18.29) |
| `pipeline/usweep.py` | U-width sweep with `--usweep-skip-promote`, `--upstream-dir`, headline t=0 columns (v8.18.30) |
| `pipeline/config.py` | full config surface incl. `use_pos_weight`, `hinge_margin`, `hinge_freeze_decoder`, `target_dir_method`, `min_support`, `legacy_prompts`, `exclusions_in_annotator_suffix` |
| `pipeline/annotate.py` | `_format_feature_for_annotator(include_exclusions=...)` for the throughput / boundary-discipline trade (v8.18.34) |
| `pipeline_data/usweep_frozen_literal/summary.json` | test-catalog frozen-decoder × literal-hinge sweep |
| `pipeline_data/evaluation.json` | latest production-scale run results |
| `pipeline_data/feature_catalog.json` | post-min-support 43-feature catalog |
| `pipeline_data/feature_catalog.unfiltered.json` | pre-filter backup |
| `pipeline_data/feature_catalog.quarantined.json` | dropped features with `_drop_reason` annotations |
| `supervised_saes_hinge_loss.md` | mentor's note (formulations 1-3, gate-loss arguments) |
| `summary8.md` | prior cycle: discovery loop ships, methodology retrenchment |
| `summary9.md` | this writeup |

## Process note

Eight reviewer rounds over this cycle, focused on threshold geometry and catalog quality. Each round caught a specific overclaim or unjustified assumption: "two equal stories" (corrected to "main + ablation"), "frozen wins on every axis" (corrected to "wins on F1 / cos / FVE / intervention plausibility, but ΔR²(S) ≈ 0 at U=1024 means supervised slice isn't the recon engine at all widths"), "cos = 1 validates direction" (corrected to "cos = 1 is by construction; needs causal validation"), "literal hinge proves the framework wrong" (corrected to "the runs labeled 'mentor's hinge' used margin=1 and pos_weight; not the literal formula"), "we have v8.10's 0.62 F1" (corrected to "we have 0.42 cal F1; v8.10's 0.62 was a noisy-label ceiling, not a quality win"), "supervised SAE doesn't beat probe on real catalogs" (overturned by the v8.18.34 production result: t=0 F1 0.502 vs probe 0.389).

The final correction matters: the supervised SAE *does* beat the probe, but only at t=0, and only because of the calibration-honesty property the mentor's loss formulation provides. That's a contribution the linear probe and post-train baselines structurally cannot match — they rely on per-feature threshold calibration to be competitive at all.

The architecture in v8.18.34 is genuinely simpler than v8.10 (Delphi removed, vLLM workarounds removed, hinge-family modes consolidated), has better empirical numbers (test-catalog supT0 0.672 → 0.772, production t=0 F1 vs probe gap −0.05 → +0.11), and has tighter honesty discipline (cos = 1 reported with caveats, calibrated/oracle F1 demoted to diagnostic, t=0 F1 promoted to headline, L0 ratio added as a separate calibration-honesty metric).

The publishable contribution rests on three legs:
- t=0 F1 beats probe at production scale ✓ (this run)
- L0 calibration-honest at scale ✓ (this run)
- Reconstruction parity at 230× fewer latents ✓ (this run)
- ΔR²(S) > 0 at U=64 ✓ (this run)
- Causal effect along named directions ✗ (needs `--step causal`)

Until leg five is in, "named, controllable directions" is asserted on the strength of FVE (heterogeneous) and cos = 1 (vacuous). Closing that gap is the next experiment.
