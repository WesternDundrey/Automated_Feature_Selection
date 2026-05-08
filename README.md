# Catalog-Driven Supervised Sparse Autoencoders

Specify a catalog of token-level features as prefix-decidable yes/no questions, label every (token, feature) pair with a local LLM, train an SAE whose decoder columns are pinned to the conditional-mean directions of those features. Recovers what unsupervised SAEs miss.

**Headline result:** mean cal F1 = 0.604 on a 103-feature catalog vs 0.025 for real EleutherAI Delphi auto-interp on the unsupervised 24,576-latent `gpt2-small-res-jb` SAE. Same model, same layer, same corpus, same scorer.

---

## For reviewers

If you've landed here from the paper, here is the orientation:

| You want to | Read |
|---|---|
| The paper itself | `paper.tex` (submitted) |
| The chronological research story | [`TIMELINE.md`](TIMELINE.md) |
| The latest production result (8B annotator + sequence-split) | [`summary11.md`](summary11.md) |
| The 10K-seq curated baseline | [`summary10.md`](summary10.md) |
| The methodology synthesis (literal hinge + frozen decoder + boundary discipline) | [`summary9.md`](summary9.md) |
| How to reproduce | [`RUNNING.md`](RUNNING.md) |
| The full per-file change log | [`changes.md`](changes.md) |
| The mentor's loss-design note | [`supervised_saes_hinge_loss.md`](supervised_saes_hinge_loss.md) |
| What went wrong at scale (failure summary) | inline in [`changes.md`](changes.md) under v8.19 + the "Failed loss configurations" paragraph in the paper Discussion |

---

## Repository map

The actual research pipeline lives in **`pipeline/`**. Everything at the repo root is documentation, demos, or external dependencies.

```
.
├── pipeline/                  ← THE CODE. Catalog → annotation → train → eval.
│   ├── run.py                   CLI entry point. Dispatches --step {inventory,
│   │                            annotate, train, evaluate, causal, ...}.
│   ├── config.py                Single Config dataclass; every flag flows here.
│   ├── inventory.py             Stage 1: catalog generation via frontier LLM.
│   ├── annotate.py              Stage 2: local-vLLM token-level labeling.
│   ├── supervised_hinge.py      Stage 3a: literal-hinge trainer (default).
│   ├── train.py                 Stage 3b: legacy hybrid/MSE/BCE trainer.
│   ├── evaluate.py              Stage 4: F1, calibration, R², per-feature FVE.
│   ├── causal.py                Per-feature ablation KL_+ / targeting ratio.
│   ├── extend_corpus.py         Grow N_old → N_new without re-annotating.
│   ├── opus_judge.py            Catalog cascade: Opus quality judge.
│   ├── propose_haiku.py         Catalog cascade: Haiku candidate proposer.
│   ├── filter_candidates.py     Catalog cascade: deterministic filters.
│   ├── validate_catalog.py      Catalog cascade: Sonnet validator.
│   ├── dedup_catalog.py         Post-train target_dir clustering for dedup.
│   ├── catalog_quality.py       Crispness gate + boundary-discipline checks.
│   ├── irr.py                   Annotator self-consistency (κ between passes).
│   ├── pilot.py                 Cheap end-to-end go/no-go gate.
│   ├── promote_loop.py          U→S capacity transfer loop.
│   ├── composition.py           K-way joint-ablation linearity.
│   ├── intervention.py          Steering / activation-patching tooling.
│   ├── feature_catalog.json     Default 103-leaf curated catalog.
│   └── (utility modules)        position_mask, cache_meta, supervised_sae, …
│
├── tools/                     Standalone scripts (vllm_smoke benchmark, etc.)
├── changes.md                 Per-version-tag log of every code change.
├── summary*.md                Chronological research summaries (see TIMELINE.md).
├── pipeline_steps.md          v2.0 implementation documentation.
├── phase1_validation.md       Early validation phase notes.
├── supervised_saes_hinge_loss.md  Mentor's note on loss design.
├── RUNNING.md                 vast.ai setup + run commands.
├── install.sh / setup.sh      vast.ai bootstrap scripts.
├── supervised_sae_demo.ipynb  v1.0 single-circuit (rabbit→habit) proof-of-concept.
├── rabbit_habit_supervised_sae.ipynb  Deprecated v0.x (had circular ground truth).
├── debug_pretrained_sae.py    Quick sanity check for sae_lens loading.
├── test_pretrained.py         Pretrained-SAE reconstruction sanity check.
├── delphi/                    Eleuther Delphi (cloned, used for unsup arm).
├── circuit-tracer/            Circuit-tracer dependency.
└── agentic-delphi*/           Reference Delphi forks (not imported directly).
```

---

## Quickstart

Trained checkpoints are released via an anonymous HuggingFace mirror referenced in the paper's "Software, Data, and Reproducibility" section. Reproducing from scratch:

```bash
# 1. clone + install (vast.ai instructions in RUNNING.md)
git clone <repo>
cd Automated_Feature_Selection
bash install.sh
export OPENROUTER_API_KEY="sk-or-..."   # only needed for catalog generation

# 2. minimal run: 8 prefix-decidable test features, 500 seqs, ~10 min on a 4090
python -m pipeline.run --catalog pipeline/test_catalog.json \
    --local-annotator --n_sequences 500 --epochs 15

# 3. full automated pipeline matching summary11 headline (~6 hr on 2× RTX 6000 Pro)
python -m pipeline.run --local-annotator --full-desc \
    --n_latents 500 --n_sequences 5000 --epochs 15 \
    --output_dir pipeline_data --supervision hinge

# 4. evaluate
python -m pipeline.run --output_dir pipeline_data --step evaluate
```

For full flag documentation, sequence-level held-out evaluation, IRR / causal / ablation steps, and the multi-shard vLLM annotation flow, see [`RUNNING.md`](RUNNING.md).

---

## What's specified vs trained vs measured

| Stage | Decision | Where it's set |
|---|---|---|
| Catalog (feature names + descriptions) | LLM-specified, manually curated | `pipeline/inventory.py`, `pipeline/feature_catalog.json` |
| Token-level labels | LLM-annotated | `pipeline/annotate.py` |
| Decoder direction (supervised slice) | **Frozen** at conditional-mean target | `pipeline/supervised_hinge.py` (Eq. 2 in paper) |
| Encoder, biases, U-slice decoder | Trained | hinge loss + reconstruction MSE |
| F1 / calibration / R² | Measured on held-out positions | `pipeline/evaluate.py` |
| Per-feature causal effect | Measured via ablation KL | `pipeline/causal.py` |

---

## Key empirical headlines

(From `summary11.md` and `summary10.md`. Full per-feature tables in the JSON artifacts shipped with the HuggingFace release.)

| metric | sup.\ SAE (this work) | linear probe | Delphi auto-interp |
|---|---:|---:|---:|
| mean F1 (z=0)              | **0.613** | 0.474 | 0.025 |
| mean cal F1                | 0.604     | 0.607 | --- |
| mean cal F1 (sequence-split) | **0.600** | --- | --- |
| naive L0 ratio             | **1.022** (token) / **1.015** (seq) | --- | --- |
| R² (full SAE)              | 0.969     | --- | 0.984 (pretrained) |
| #latents                   | 103+256   | ---  | 24,576 |
| 12 of 46 features pass causal-necessity (predecessor artifact) | ✓ | --- | --- |

---

## Citing this work

```
Catalog-Driven Supervised Sparse Autoencoders: Specifying Token-Level
Features Before Training Recovers What Unsupervised Dictionaries Miss.
Anonymous. ICML 2026 Mech Interp Workshop submission.
```

The non-anonymous URL and citation will appear in the camera-ready version.

---

## License

To be added. Code under MIT / Apache-2.0 (TBD); model weights under the corresponding HuggingFace license.

---

## Anonymity statement

This repository is shared in support of an ICML 2026 double-blind submission. References to author names, email addresses, GitHub or HuggingFace usernames, institutional affiliations, and grants are intentionally absent. The on-start script, raw GitHub URLs, and HuggingFace model name in the public copy of this repo are scrubbed; reviewers should reach the anonymized artifact via the paper's "Software, Data, and Reproducibility" section.
