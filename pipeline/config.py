"""
Pipeline configuration.

Edit defaults below or override at runtime:
    cfg = Config(target_layer=16, n_sequences=10000)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── Base model ───────────────────────────────────────────────────
    model_name: str = "gpt2"
    # Layer 9 is the densest semantic band in GPT-2 Small. Validated in
    # summary7: 64 leaves from Sonnet vs 31 at layer 6; cal_F1=0.625 vs 0.484;
    # pretrained SAE R²=0.985 (with the standard from_pretrained loader fix).
    target_layer: int = 9
    hook_point: str = ""  # auto-set in __post_init__
    device: str = "cuda"
    model_dtype: str = "float32"  # GPT-2 Small runs in fp32

    # ── Pretrained SAE (sae_lens format) ─────────────────────────────
    sae_release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.9.hook_resid_pre"

    # ── Feature selection from pretrained SAE ────────────────────────
    n_latents_to_explain: int = 500
    # v8.19.0: bumped 30→50 for Opus 4.7 1M-context (more activating
    # contexts per latent strengthens descriptions). v8.18.26 raised this
    # from 20→30 when Delphi was removed; v8.19 raises again now that
    # Opus 4.7 replaces Sonnet for the sup-arm catalog design pass.
    top_k_examples: int = 50
    # v8.19.0: 2M→3M to give shortlist_latents stable freq + concentration
    # estimates over the 24576 gpt2-small-res-jb latents (Engels-style
    # dense-latent detection threshold needs ≥3M for ±0.01 freq precision
    # at the 0.10 cutoff).
    n_tokens_for_activation_collection: int = 3_000_000
    activation_collection_batch_size: int = 8
    activation_collection_seq_len: int = 128
    min_firing_rate: float = 0.0005
    max_firing_rate: float = 0.10

    # ── LLM models (OpenRouter model IDs) ────────────────────────────
    explanation_model: str = "anthropic/claude-sonnet-4.6"
    organization_model: str = "anthropic/claude-sonnet-4.6"
    annotation_model: str = "anthropic/claude-haiku-4.5"
    # v8.19.0: 10→30 for Opus 4.7 1M context (used by opus_catalog.py
    # design pass; legacy Sonnet inventory still uses 10 via local override).
    features_per_explanation_batch: int = 30
    # v8.19.6: 50→80 for the 500-feature scaling run. The annotator's
    # accuracy holds at ≤100 features per call when descriptions are
    # short (≤10 words, single sentence — enforced by the v8.19.6 Opus
    # prompt edits). Pass --features-per-call N to override.
    features_per_annotation_call: int = 80
    max_annotation_concurrency: int = 20

    # ── Supervision corpus ──────────────────────────────────────────
    corpus_dataset: str = "Skylion007/openwebtext"
    corpus_split: str = "train"
    n_sequences: int = 5000
    seq_len: int = 128
    corpus_batch_size: int = 8
    # v8.19.6 lever-7 position subsampling: when > 0, annotate K of T
    # positions per sequence (deterministic per-sequence RNG seeded by
    # cfg.seed). 0 disables (full-sequence annotation, the default).
    # Saved as `position_mask.pt` sidecar; train.py / evaluate.py /
    # unsup_f1.py / oracle_unsup.py respect it. Compensate the lower
    # per-feature positive count by raising cfg.min_support — at
    # 50K seqs × 64 of 128 positions, min_support=500 gives a 0.0156%
    # base-rate floor.
    position_subsample_k: int = 0

    # ── Training ────────────────────────────────────────────────────
    n_unsupervised: int = 256
    epochs: int = 15
    batch_size: int = 512
    lr: float = 3e-4
    lambda_sup: float = 2.0
    # Bumped 0.01 → 0.05 after the no_sparsity ablation showed zero F1/L0
    # change at 0.01 (sparsity wasn't binding) and unsupervised L0 reached
    # 243/256 latents — basically dense. 0.05 tightens unsup L0 toward a
    # target SAE's typical 30–80 range without hurting supervised F1.
    lambda_sparse: float = 5e-2
    lambda_hier: float = 0.5
    # v8.18: flat catalog by default (per user preference — "i just don't
    # like grouping"). Sonnet still emits groups in organize_hierarchy
    # because the clustering step benefits from grouping during
    # generation, but the saved feature_catalog.json is post-flattened to
    # leaves only. With flat catalog, the hierarchy loss has nothing to
    # do (no parent-child relations), so lambda_hier is auto-zeroed at
    # train time when flatten_catalog=True. Pass --keep-groups to opt out.
    flatten_catalog: bool = True
    # v8.19.0: 500→700 for the 300-feature catalog at 5K seqs (~15K total
    # steps; 5% warmup matches summary8/9 production runs). For pilot
    # (500 seqs × 80 features) the trainer auto-scales warmup to 100.
    warmup_steps: int = 700
    train_fraction: float = 0.8
    seed: int = 42
    n_lista_steps: int = 0  # LISTA refinement iterations (0 = disabled)

    # ── Supervision mode ───────────────────────────────────────────
    # DEFAULT: "hinge" (per mentor's methodology note, `supervised_saes_hinge_loss.md`).
    # The R² regression observed in v8.11.2's end-to-end test was NOT
    # caused by the loss function — it was BOS masking (v8.6) dropping
    # baseline_mse 10× because position 0 in the residual stream has
    # ~10× higher variance than other positions. SAE reconstruction MSE
    # was essentially unchanged (3.24 → 3.31); the R² denominator was
    # what collapsed. See v8.11.3 changelog for the full diagnostic.
    #
    # Hinge-family (end-to-end training, no frozen decoder):
    # "hinge"           = ReLU + hinge on pre-activations (default).
    # "hinge_jumprelu"  = JumpReLU + hinge on (z - θ), per-feature θ learnable.
    # "gated_bce"       = Two-path encoder (gate + magnitude), BCE on gate.
    #
    # Frozen-decoder family (for reproducing summary6/7 or A/B):
    # "hybrid"          = BCE selectivity + cosine direction alignment.
    # "mse"             = MSE magnitude + cosine direction (Makelov-inspired).
    # "bce"             = legacy BCE only, no decoder alignment.
    supervision_mode: str = "hinge"
    use_mse_supervision: bool = True    # DEPRECATED: use supervision_mode instead
    direction_loss_weight: float = 1.0  # α: decoder direction alignment (hybrid/mse)
    magnitude_loss_weight: float = 0.5  # β: activation magnitude alignment (mse only)
    selectivity_loss: str = "bce"       # "bce", "hinge", or "none"
    use_pos_weight: bool = False        # v8.19.2: DEFAULT FLIPPED to False.
                                        # The literal mentor formula in
                                        # supervised_saes_hinge_loss.md is
                                        # `max(0, -(2y-1) z_i)` with no
                                        # class-imbalance reweighting; this
                                        # is what produced summary9's
                                        # calibration-honest L0=1.015 and
                                        # t=0 F1 vs probe gap +0.187 at
                                        # production scale. Pass
                                        # --use-pos-weight to opt back into
                                        # the SVM-style class-balanced
                                        # variant for ablation.
    hinge_margin: float = 0.0           # v8.19.2: DEFAULT FLIPPED to 0.0
                                        # (zero-margin hinge per mentor
                                        # formula). Sign-correctness only,
                                        # no score shaping past z=0. Pass
                                        # --hinge-margin 1.0 (or higher)
                                        # for SVM-style margin hinge that
                                        # continues shaping scores until a
                                        # configurable confidence buffer.
    hinge_squared: bool = False         # squared hinge: violation² instead
                                        # of raw violation. Smoother gradient
                                        # near boundary; more gradient on
                                        # large violations. Useful if linear
                                        # hinge is under-training.

    # Knobs for the new (v8.11) hinge / gated modes:
    # - gated_tie_weights: if True, GatedBCESAE ties W_mag = exp(r) · W_gate
    #   (per-feature scale r), halving supervised-slice encoder params at
    #   the cost of some expressiveness.
    # - jumprelu_theta_init: initial value of the per-feature θ threshold
    #   in JumpReLUHingeSAE. Small positive (0.1) gives the hinge something
    #   to push against on step 1.
    gated_tie_weights: bool = False
    jumprelu_theta_init: float = 0.1

    # ── v3: Frozen decoder ──────────────────────────────────────────
    # Fix supervised decoder columns to target_dirs before training.
    # Only encoder + unsupervised decoder train. Direction loss is
    # skipped (decoder is already aligned by construction).
    # Ablation proved frozen = learned on F1/R², but frozen gives
    # cosine=1.0 and 5.7× FVE — strictly better for interventions.
    freeze_supervised_decoder: bool = True
    # v8.18.19: ABLATION-ONLY for hinge family modes (hinge,
    # hinge_jumprelu, gated_bce). Opt-in via --freeze-decoder. Off by
    # default so the mentor's "no hacks" design holds for the standard
    # `--supervision hinge` / `--supervision gated_bce` runs. Set True
    # to pin supervised decoder columns at target_dirs and zero their
    # gradient — useful for isolating which hack in hybrid mode (BCE +
    # cosine direction loss + frozen decoder) is doing the F1 work.
    # v8.18.25: DEFAULT FLIPPED to True. End-to-end hinge / gated_bce
    # gave decoder cosines ~0.16 (random) and FVE ~0.005 (useless for
    # intervention). With frozen decoder you get cos=1.0 by construction
    # and FVE ~0.30 — the supervised slice gives clean intervention-
    # ready directions. F1 cost is ~0.03 vs the linear probe baseline,
    # well worth it for interpretability + causal validity. The mentor's
    # "no hacks" framework empirically loses 0.13+ F1 to this single
    # hack at our scale; we accept the hack for the working architecture.
    # Override with --no-freeze-decoder if you want to test the pure
    # principled formulation.
    hinge_freeze_decoder: bool = True

    # ── v8.18.20 catalog quality gates + direction ablation ──
    # Catalog quality validator. "report" = compute + write report,
    # don't drop. "quarantine" (default) = drop hard-fail leaves only
    # (operationally-undefined phrases like "sometimes" / "various" /
    # "associated with"; missing source_latents). "hard" = also drop
    # quarantine-flagged leaves (soft flags + over-long descriptions).
    # The user's design: lexical flags trigger Sonnet crispness check
    # (not auto-drop), so surface variants like "opening or closing
    # bracket" survive while "noun or verb" doesn't.
    catalog_gate_mode: str = "quarantine"
    # Whether the validator calls Sonnet's crispness judgment on
    # soft-flagged descriptions. Off = lexical-only (cheap but noisy).
    catalog_gate_use_llm: bool = True
    # Strict mode: a validator crash/import-failure/exception RAISES
    # instead of falling through with an unfiltered catalog. Off by
    # default (research code, often want to keep going). Per audit:
    # "If this is meant to be a defensive contract, failing the
    # validator should fail the run, not silently proceed." Set to
    # True when you want hard guarantees.
    catalog_gate_strict: bool = False
    # Auto-run pairwise overlap check after annotation. Reports
    # redundant pairs (IoU >= iou_threshold) and subset pairs
    # (max(P(A|B), P(B|A)) >= subset_threshold). Doesn't drop by
    # default — emits a report the user reviews.
    overlap_check_auto: bool = True
    overlap_iou_threshold: float = 0.8
    overlap_subset_threshold: float = 0.95
    overlap_min_support: int = 30

    # Post-annotation min-support filter (v8.18.33). After annotation,
    # any feature with positive count strictly less than this threshold
    # gets dropped from the catalog before training. n_pos < min_support
    # features have AUROC near random and contribute pure noise to the
    # mean F1; they're below the statistical floor where supervised
    # learning can extract a clean classifier. Dropped features are
    # appended to feature_catalog.quarantined.json with reason
    # "min_support<N" so the audit trail is preserved. Set to 0 (default)
    # to disable. Pass --min-support N to enable.
    min_support: int = 0

    # v8.10-style relaxed prompts (v8.18.34). The strict
    # prefix-decidable contract (v8.18.32) plus boundary-discipline
    # schema (v8.18.28) plus exclusions-in-annotator-suffix (also
    # v8.18.28) collectively narrowed the catalog and roughly tripled
    # annotation time per decision. For workflows where the strict
    # contract isn't load-bearing (broader catalogs, faster iteration),
    # legacy_prompts=True drops the prefix-decidable enforcement from
    # the inventory prompts and turns its regex backstop into a soft
    # flag (LLM-judged) instead of a hard fail. exclusions_in_annotator_
    # suffix=False additionally keeps Sonnet's exclusions metadata for
    # human audit but doesn't append them to the annotator's suffix,
    # recovering ~2-3x annotation throughput. Pass --legacy-prompts to
    # set both at once.
    legacy_prompts: bool = False
    # v8.19.6: DEFAULT FLIPPED to False (was True). At 500-feature scale,
    # rendering exclusions into the annotator suffix triples per-decision
    # token count without measurable label-quality gain — boundary-
    # discipline is enforced at catalog-generation time (positive_examples,
    # negative_examples, exclusions are stored in catalog metadata for
    # audit) and the annotator follows the description directly. Pass
    # --exclusions-in-suffix to opt back in for small catalogs.
    exclusions_in_annotator_suffix: bool = False

    # When true, --step extend-corpus saves a permanent immutable
    # snapshot of the pre-extension annotations.pt + tokens.pt at
    # `_pre_extend_<n_old>seqs/` before extending. Lets you compare
    # extended-vs-old or restore manually if anything's wrong post-hoc.
    # Costs ~one annotations.pt of disk space per extension.
    extend_clone_pre: bool = False

    # Target direction method for the freeze-decoder pin and the
    # in-loss direction supervision (hybrid/mse modes).
    #   "mean_shift": current default. d = normalize(μ_pos - μ_all).
    #                 Crude, robust, cleanest interpretation.
    #   "logistic":   ridge logistic regression weight per feature.
    #                 Optimal classification direction; uses confounds
    #                 if they help separation. NOT a "feature direction"
    #                 in the interpretation sense.
    #   "lda":        whitened mean-shift, (Σ + λI)^-1 (μ_pos - μ_all),
    #                 with shrinkage λ. Suppresses high-variance junk.
    #                 Needs strong shrinkage at our 768d / rare-positive
    #                 sample size.
    target_dir_method: str = "mean_shift"
    target_dir_logistic_lambda: float = 1.0
    target_dir_lda_shrinkage: float = 0.1

    # ── v2: Local model annotation ────────────────────────────────
    use_local_annotator: bool = True    # v8.19.4: DEFAULT FLIPPED to True.
                                        # Local Qwen3-4B-Base via vLLM is
                                        # the user's production setup
                                        # (free, ~700 dec/sec/GPU, multi-
                                        # GPU shard). Pass --no-use-local-
                                        # annotator (or set False
                                        # explicitly) to fall back to
                                        # OpenRouter Haiku API for
                                        # tiny-scale debug runs only.
    local_annotator_model: str = "Qwen/Qwen3-4B-Base"  # base model, no thinking, pure transformer
    local_annotation_batch_size: int = 64
    batch_positions: bool = False  # True = full-sequence JSON, False = per-token
    # Run local vLLM annotation in a fresh Python process. This is more
    # robust than trying to keep the parent process CUDA-clean after
    # inventory / Delphi / transformer-lens work.
    local_annotation_subprocess: bool = True
    # v8.18.16: data-parallel local annotation. When local_annotation_parallel
    # is True AND >= 2 GPUs are visible (via CUDA_VISIBLE_DEVICES or
    # torch.cuda.device_count), the annotator splits the corpus into
    # N shards and runs one vLLM instance per GPU concurrently. Each
    # subprocess loads its own ~7.5 GB Qwen3-4B-Base copy; both 5090s
    # have 32 GB each so this fits comfortably. Roughly linear speedup.
    # Set to False to force single-GPU annotation even when 2+ GPUs
    # are present (e.g., to leave one free for another job).
    local_annotation_parallel: bool = True
    # 0 = auto-detect (CUDA_VISIBLE_DEVICES count, falling back to
    # torch.cuda.device_count); 1 = force single-GPU; 2+ = use exactly
    # this many shards (must be ≤ available GPUs).
    n_annotation_gpus: int = 0
    # Sequences per vLLM batch in the per-token annotator. Tuning history:
    # v8.19.6 bumped 2 → 8: too aggressive, Python overhead.
    # Reverted to 2.
    # v8.19.8 dmon showed GPUs at 0% under chunk=2 — vLLM was starving.
    # Tried 4: throughput ~313 dec/s/shard. Better but still patchy.
    # User experiment with --annotation-seq-chunk 32 + max_num_seqs=1024:
    # 654 dec/s/shard with prefix cache warm (~2244× cold→warm speedup
    # observed). 4× the chunk=2 baseline, sustained GPU work. Settling
    # the default at 32 on 5090s with 32 GB. Pass --annotation-seq-chunk
    # to override; 16 might suffice on memory-tight GPUs, 64 if
    # construction overhead is fine.
    local_annotation_seq_chunk: int = 32

    # Positions to mask out at analysis time (starting from position 0).
    #
    # DEFAULT: 0 (v8.11.3). Masking position 0 was added in v8.6 to fix
    # the target_dir collapse where sequence-level features all
    # converged to the same direction due to BOS dominating their
    # positive sets. The side effect: position 0 in the residual stream
    # has ~10× the variance of other positions (attention-sink behavior
    # + no prior context), so masking it drops baseline_mse 10× and
    # correspondingly drops reconstruction R² from ~0.97 to ~0.70 even
    # though the SAE's absolute reconstruction is identical. That
    # regression invalidates summary7's headline R² claim.
    #
    # Downstream BOS handling now routes through v8.10 infrastructure
    # instead of masking:
    #   - scaffold_catalog.json has `control.document_boundary` to name
    #     the BOS token as a control feature (role="control"), so it's
    #     trained but excluded from discovery-only paper stats.
    #   - promote_denylist rejects descriptions naming BOS / endoftext /
    #     padding before the crispness gate, so the discovery loop
    #     never promotes BOS-detector U latents as headline features.
    #   - evaluate reports discovery-only means alongside full-catalog
    #     means so paper stats can cite the cleaner number.
    #
    # Set to 1 to restore the v8.6-v8.11.2 behavior (BOS mask on) if a
    # specific analysis requires BOS-free data (e.g., promote_loop U
    # ranking tends to produce BOS detectors without this).
    mask_first_n_positions: int = 0

    # Maximum fraction of annotation chunks that may end up zero-labeled
    # (after max retries) before the run aborts. A few stragglers are
    # tolerable; >10% means the prompt or the annotator is broken, and
    # training on those labels would silently corrupt the catalog.
    annotation_max_failure_rate: float = 0.10

    # ── Scaffold / control catalog ─────────────────────────────────
    # Path to an optional scaffold catalog (see `pipeline/scaffold_catalog.json`)
    # merged into feature_catalog.json before training. Scaffold features
    # are tagged `role="control"` so downstream evaluation stats can
    # report "headline / discovery-only" numbers separately from "all
    # features (including scaffold)". Empty string = no scaffold merge.
    #
    # v8.15: default flipped from "" to the bundled scaffold path. Pre-v8.15
    # the v8.14 control features were inert by default (only merged when
    # the user explicitly passed --scaffold-catalog). Default-on means new
    # runs absorb surface-artifact directions automatically; pass
    # --no-scaffold to opt out.
    scaffold_catalog: str = "pipeline/scaffold_catalog.json"

    # v8.18.26: Delphi REMOVED entirely. The fields delphi_*,
    # promote_use_delphi_gate, delphi_gate_in_inventory, etc. are
    # gone. Delphi was nerfing supervised-SAE F1 by source-latent
    # faithfulness filtering — the user's audit + experimental
    # evidence both pointed at removal. Catalog quality is enforced
    # via pipeline.catalog_quality (lexical + LLM crispness) and
    # post-annotation overlap analysis instead.

    # Denylist of description patterns. During promote-loop triage, any
    # candidate description (from Sonnet) that contains ANY of these
    # substrings (case-insensitive) is auto-rejected as `denylist`
    # without further gates. Used to prevent the loop from promoting
    # things like BOS tokens, padding, known artifacts — directions the
    # catalog should *audit* but not promote as discoveries.
    promote_denylist: tuple = (
        "endoftext", "<|endoftext|>",
        "beginning of sequence", "bos token",
        "document boundary marker", "padding token",
        "start-of-sequence", "sequence start marker",
    )

    # ── Feature filtering ──────────────────────────────────────
    min_feature_positive_rate: float = 0.0  # disabled by default (rare features are intentional)

    # ── Annotation robustness ────────────────────────────────────
    annotation_max_retries: int = 3
    annotation_retry_base_delay: float = 1.0

    # ── Inter-annotator agreement ────────────────────────────────
    agreement_n_sequences: int = 100
    agreement_n_reruns: int = 2  # number of independent annotation passes

    # ── Causal validation ────────────────────────────────────────
    causal_n_sequences: int = 50  # small — each needs n_features forward passes
    causal_batch_size: int = 4

    # ── Explain-the-residual ─────────────────────────────────────
    residual_n_samples: int = 500
    residual_top_k_positions: int = 100
    residual_model: str = "anthropic/claude-sonnet-4.6"

    # ── Manual catalog (skips inventory step) ────────────────────
    manual_catalog: str = ""  # path to JSON catalog; empty = use inventory

    # ── v8.19 Delphi-vs-Opus comparison architecture ─────────────
    # Symmetric Type-1 native-pipeline F1 head-to-head:
    #   sup arm:   Opus 4.7 designs N features → annotate → train supSAE
    #              → F1(supSAE feature firing vs labels(Opus desc))
    #   unsup arm: real EleutherAI Delphi describes N latents 1:1
    #              → annotate → F1(unsup latent firing vs labels(Delphi desc))
    # Both arms consume the same shortlist of candidate latents from
    # gpt2-small-res-jb (selection-freedom asymmetry IS the methodology).
    shortlist_size: int = 1000
    delphi_n_features: int = 300
    opus_n_features: int = 300
    # v8.19.7: Opus 4.7 output is capped (max_tokens=64K = ~150 features
    # at ~400 JSON tokens per feature with full boundary-discipline). To
    # design more features, split into multiple API calls; each call
    # designs `opus_features_per_call` features from its slice of the
    # shortlist. Chunked design auto-triggers when opus_n_features >
    # opus_features_per_call.
    opus_features_per_call: int = 100
    # Opus 4.7 OpenRouter model id. Verify exact slug at first call;
    # update if Anthropic re-tags. 1M context variant required so Opus can
    # ingest top contexts of 1000 latents in a single design pass.
    opus_explanation_model: str = "anthropic/claude-opus-4.7"
    # Real EleutherAI Delphi: explainer/scorer model on OpenRouter. Sonnet
    # 4.6 is cost-bounded for 300 latents (~$10 vs ~$200 for Opus). Switch
    # to Opus only if ablation shows Sonnet's descriptions limit unsup F1.
    delphi_explainer_model: str = "anthropic/claude-sonnet-4.6"
    delphi_explainer_provider: str = "openrouter"
    delphi_scorer: str = "detection"   # "detection", "fuzz", or both
    # Empty = arm-local default (resolved to <output_dir>/delphi_run in
    # __post_init__). Pass an explicit path to override; pilot.py sets
    # this per-arm so the unsup arm's cache lives under
    # pipeline_data_unsup/delphi_run rather than colliding with the sup
    # arm's pipeline_data/delphi_run.
    delphi_run_dir: str = ""
    # Pre-Delphi/Opus latent shortlist: candidate-pool selection from the
    # 24576-latent gpt2-small-res-jb. Frequency window keeps the
    # describable middle of the distribution; current implementation
    # ranks by descending firing rate inside the window. Concentration-
    # based ranking is documented as a future enhancement in
    # shortlist_latents.py.
    shortlist_freq_min: float = 0.0005   # drop dead latents
    shortlist_freq_max: float = 0.10     # drop ultra-dense (Engels territory)
    shortlist_concentration_topk: int = 1000  # documented; not yet used
    shortlist_calibration_tokens: int = 3_000_000
    # 4-GPU annotation shard: one CUDA-isolated subprocess per GPU.
    n_annotation_shards: int = 4
    # Pilot gate (HARD requirement before full 38hr run):
    pilot_n_sequences: int = 500
    pilot_opus_n_features: int = 50
    pilot_delphi_n_features: int = 30
    # Per-catalog inter-rater reliability sample size (double-annotated).
    irr_sample_size: int = 30
    # Project-out preprocessing (Engels dense latents). PARKED for now;
    # separate ablation, not headline. Detector threshold matches Engels.
    project_out_dense: bool = False
    dense_freq_threshold: float = 0.10

    # v8.19.2 two-arm flow: the unsup arm runs in its own output_dir
    # (typically `pipeline_data_unsup/`). The compare step reads both
    # `cfg.output_dir` (sup arm) and `cfg.unsup_output_dir` (unsup arm)
    # to produce the headline table. Empty string falls back to
    # `<output_dir>_unsup` then `pipeline_data_unsup`.
    unsup_output_dir: str = ""

    # v8.19.5 resume/checkpointing. Each step's run() checks for its
    # primary output artifact at start; if present and valid, prints
    # [resume] and returns the loaded result. Set True to FORCE
    # regeneration (bypass all skip-if-exists checks). Pass --force on
    # the CLI to enable.
    force: bool = False

    # ── Output ──────────────────────────────────────────────────────
    output_dir: str = "pipeline_data"

    def __post_init__(self):
        if not self.hook_point:
            # Derive from sae_id if it contains a hook point hint
            if "hook_resid_pre" in self.sae_id:
                self.hook_point = f"blocks.{self.target_layer}.hook_resid_pre"
            else:
                self.hook_point = f"blocks.{self.target_layer}.hook_resid_post"
        self.output_dir = Path(self.output_dir)
        # v8.19.2 two-arm flow: delphi_run_dir is arm-local by default so
        # --output_dir pipeline_data_unsup automatically gets its own
        # delphi_run cache instead of colliding with the sup arm's.
        if not self.delphi_run_dir:
            self.delphi_run_dir = str(self.output_dir / "delphi_run")
        # Fallback to CPU if CUDA not available.
        # Use environment check to avoid initializing CUDA (which breaks vLLM fork).
        if self.device == "cuda":
            import os
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
                print("WARNING: CUDA disabled via env, falling back to CPU")
                self.device = "cpu"

    @property
    def catalog_path(self) -> Path:
        return self.output_dir / "feature_catalog.json"

    @property
    def top_activations_path(self) -> Path:
        return self.output_dir / "top_activations.json"

    @property
    def raw_descriptions_path(self) -> Path:
        return self.output_dir / "raw_descriptions.json"

    @property
    def activations_path(self) -> Path:
        return self.output_dir / "activations.pt"

    @property
    def tokens_path(self) -> Path:
        return self.output_dir / "tokens.pt"

    @property
    def annotations_path(self) -> Path:
        return self.output_dir / "annotations.pt"

    @property
    def annotations_meta_path(self) -> Path:
        """Sidecar JSON recording the feature-id sequence of annotations.pt's
        last axis. Used to detect catalog reorderings / partial overlaps so
        cached labels can be remapped by ID instead of silently misbinding to
        whatever happens to sit at the same column index."""
        return self.output_dir / "annotations_meta.json"

    @property
    def position_mask_path(self) -> Path:
        """Sidecar (N, T) bool tensor written by annotate.py when
        position_subsample_k > 0. Downstream consumers (train.py,
        evaluate.py, unsup_f1.py, oracle_unsup.py) load this and treat
        unsampled positions as `not labeled` rather than `feature did
        not fire`."""
        return self.output_dir / "position_mask.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "supervised_sae.pt"

    @property
    def checkpoint_config_path(self) -> Path:
        return self.output_dir / "supervised_sae_config.pt"

    @property
    def eval_path(self) -> Path:
        return self.output_dir / "evaluation.json"

    @property
    def agreement_path(self) -> Path:
        return self.output_dir / "agreement.json"

    @property
    def ablation_path(self) -> Path:
        return self.output_dir / "ablation.json"

    @property
    def residual_path(self) -> Path:
        return self.output_dir / "residual_features.json"

    @property
    def causal_path(self) -> Path:
        return self.output_dir / "causal.json"

    @property
    def target_dirs_path(self) -> Path:
        return self.output_dir / "target_directions.pt"

    @property
    def split_path(self) -> Path:
        return self.output_dir / "split_indices.pt"

    @property
    def weaknesses_path(self) -> Path:
        return self.output_dir / "weaknesses.json"

    @property
    def siphoning_path(self) -> Path:
        return self.output_dir / "siphoning.json"
