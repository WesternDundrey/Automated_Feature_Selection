# Phase 1: GPT-2 Validation (pre-Gemma)

**Goal:** Harden the supervised SAE's claims before scaling to Gemma-2-2B. Three tests:

1. **Causal validation** — does the SAE causally influence model behavior?
2. **Calibrated thresholds** — honest F1 without oracle test-set tuning
3. **Ablation study** — which loss components actually matter?

---

## 1. Causal Validation

**Command:**
```bash
python -m pipeline.run --step causal
```

**What it runs (4 tests):**

### Test 1: Approximation (IOI)
Replaces layer-8 residual stream with SAE reconstruction on IOI prompts.
- **Sufficiency**: logit_diff(reconstruction) / logit_diff(clean). Target: > 0.9.
- **Necessity**: measures how much removing the reconstruction (keeping only the residual) hurts. Target: > 0.8.

If sufficiency is high, our 320-latent SAE preserves the information needed for the IOI circuit. If necessity is high, the SAE isn't just passing through noise — it's capturing what matters.

### Test 2: Sparse Controllability (IOI)
Greedy feature editing: select k features to add/remove to flip the model's IOI prediction.
- **IIA** (Interchange Intervention Accuracy): fraction of clean→corrupted logit shift achieved by editing k features
- **Edit success rate**: does argmax prediction actually flip?
- Tests k = 1, 2, 4 features

This tests whether individual SAE features carry enough causal weight to steer model behavior — something a linear probe fundamentally cannot do.

### Test 3: Interpretability
Placeholder for IOI-specific attribute matching. Run `--step ioi` for the full IOI validation.

### Test 4: Per-Feature Causal Necessity (NEW)
For each of the 64 supervised features:
1. Replace residual with full SAE reconstruction → baseline logits
2. Replace residual with SAE reconstruction minus feature k → ablated logits  
3. Compute KL(baseline || ablated) at positions where feature k is active

**Metrics per feature:**
- `mean_kl`: How much does removing this feature change the model's output? High KL = causally important.
- `pred_change_rate`: Fraction of active positions where the top-1 prediction changes.

**What to look for:**
- Features with high classification F1 AND high KL → the SAE learned a causally meaningful representation
- Features with high F1 but low KL → the feature classifies correctly but its decoder column isn't load-bearing (unsupervised latents compensate)
- Features with low F1 but high KL → decoder column carries causal information even though classification threshold is miscalibrated
- `Features with KL > 0.01`: count of features that measurably steer the model

**Config:**
```python
causal_n_sequences = 50   # sequences for IOI + feature necessity
causal_batch_size = 4     # (unused currently, processes one sequence at a time)
```

Increase `causal_n_sequences` for more stable estimates (50 is fine for GPT-2).

**Output:** `pipeline_data/causal.json`

---

## 2. Calibrated Threshold Evaluation

**Command:**
```bash
python -m pipeline.run --step evaluate
```

**What changed:** The 20% held-out data is now split into 10% val + 10% test.
- Thresholds are tuned per-feature on the val set using `optimal_threshold_f1()`
- All metrics (F1, AUROC, baselines) are reported on the test set
- Oracle thresholds (per-feature optimum on test) are kept for reference but clearly labeled "NOT honest eval"

**Output reports three F1 variants:**

| Metric | Meaning | Comparable? |
|--------|---------|-------------|
| F1 (t=0) | Naive threshold at pre_act > 0 | Yes — apples-to-apples with probe/post-training |
| **Calibrated F1** | Threshold from val set, applied to test | **Yes — this is the honest number** |
| Oracle F1 | Per-feature optimum on test set | No — overfit to test, reference only |

**Key question:** Does calibrated F1 beat the post-training baseline (0.474 in v3.0)?
- If yes: the claim holds under honest evaluation
- If no: the gap between t=0 (0.394) and oracle (0.502) was mostly overfitting, not calibration

**Note:** Test set is now ~6,400 vectors (was ~12,800). Per-feature positive counts are halved. Data-starved features (<25 positives in test) will be noisier.

---

## 3. Ablation Study

**Command:**
```bash
python -m pipeline.run --step ablation
```

**Variants tested (each trains from scratch):**

| Variant | Override | Tests |
|---------|----------|-------|
| baseline | (none) | Full model |
| no_hierarchy | lambda_hier=0 | Is hierarchy loss helping? |
| no_warmup | warmup_steps=0 | Does gradual supervision ramp-in matter? |
| no_unsupervised | n_unsupervised=0 | Do the 256 unsupervised latents help reconstruction? |
| no_sparsity | lambda_sparse=0 | Does L1 on all latents matter? |
| bce_only | supervision_mode=bce | Is decoder direction alignment worth the complexity? |
| mse_mode | supervision_mode=mse | How does Makelov-style MSE compare to hybrid? |

**Runtime:** ~7 variants x 15 epochs x ~30 sec/epoch = ~50 min on a single GPU.

**Output:** `pipeline_data/ablation.json` + printed comparison table with deltas from baseline.

**What to look for:**
- `no_hierarchy` delta: if F1 drops, hierarchy loss is earning its keep
- `bce_only` vs baseline: if baseline (hybrid) wins, direction alignment helps
- `no_unsupervised`: if R2 craters but F1 holds, unsupervised latents only help reconstruction
- `no_sparsity`: if L0 explodes but F1 is similar, L1 is cosmetic (our unsupervised L0=248 suggests this)

---

## Run Order

```bash
# 1. Re-evaluate with calibrated thresholds (fastest, ~2 min)
python -m pipeline.run --step evaluate

# 2. Causal validation (~5 min for GPT-2)
python -m pipeline.run --step causal

# 3. Ablation study (~50 min)
python -m pipeline.run --step ablation
```

Steps 1 and 2 use the existing trained SAE. Step 3 retrains from scratch for each variant.

**Important:** Step 1 (evaluate) will overwrite `pipeline_data/evaluation.json` with the new val/test split. The old 80/20 results in summary3.md remain as a historical snapshot.

---

## Interpreting Results

### If causal validation passes:
- Sufficiency > 0.9, necessity > 0.7, several features with KL > 0.01
- **Conclusion:** The supervised SAE is a causal model, not just a classifier. Scale to Gemma.

### If causal validation fails:
- Sufficiency < 0.8 or most features have KL < 0.001
- **Diagnosis:** The 320-latent SAE doesn't capture enough of the residual stream (R2=0.97 should prevent this) OR the decoder columns aren't aligned with causally relevant directions (cosine=0.556 might be too low).
- **Fix:** Increase unsupervised latents, or increase direction_loss_weight.

### If calibrated F1 beats post-training:
- **Conclusion:** The v3.0 result is real, not an artifact of oracle threshold tuning.

### If calibrated F1 loses to post-training:
- **Diagnosis:** The 0.394→0.502 gap was mostly overfitting to the test set. The true advantage of supervised SAE over post-training is the decoder structure (causal interventions), not classification.
- **Response:** Lean into the causal results for the narrative. Classification parity is sufficient if causal advantage is real.

---

## Code Changes (this commit)

1. **pipeline/evaluate.py** — Split test set into val (10%) + test (10%). Calibrate thresholds on val, report on test. Oracle thresholds kept but labeled.

2. **pipeline/causal.py** — Added `test_feature_necessity()`: per-feature decoder ablation measuring KL divergence and prediction change rate. Wired into `run()` as Test 4.

3. **pipeline/ablation.py** — Fixed deprecated `use_mse_supervision` flag to use `supervision_mode`. Now tests all three modes (hybrid/mse/bce).

---

## Known Issue: hook_point mismatch

The pretrained SAE baseline uses `sae_id="blocks.8.hook_resid_pre"` but the supervised SAE is trained on `hook_point="blocks.8.hook_resid_post"`. These are different activation spaces (pre vs post layer 8). The pretrained SAE's R2 and post-training F1 are evaluated on activations it wasn't trained on — making the "beats post-training" comparison unfair in our favor. To fix: either switch hook_point to `resid_pre` (requires retrain) or find a pretrained SAE for `resid_post`.

---

## All-In-One (vast.ai)

Everything in a single tmux command. Assumes setup.sh already ran (repo cloned, Qwen downloaded).

```bash
tmux new -s run
cd /workspace/Automated_Feature_Selection && git pull
uv pip install --system --no-deps sae-lens transformer-lens
uv pip install --system -r pipeline/requirements.txt
export OPENROUTER_API_KEY="sk-or-..."

# Full pipeline (inventory → annotate → train → evaluate) + Phase 1 validation
python -m pipeline.run --local-annotator --full-desc \
    --n_latents 100 --n_sequences 500 --epochs 15 \
  && python -m pipeline.run --step causal \
  && python -m pipeline.run --step ablation \
  2>&1 | tee run.log
```

Estimated time: ~3-4 hours total (annotation dominates). Results in `pipeline_data/`.
