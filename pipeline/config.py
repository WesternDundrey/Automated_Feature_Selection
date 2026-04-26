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
    # v8.16: bumped 20 → 30 so the Delphi gate has held-out positives
    # (audit fix #3). The describer LLM sees `top_k_examples -
    # delphi_held_out_n` = 20; the scorer reserves the next
    # `delphi_held_out_n` = 10 as never-shown-to-the-describer test
    # positives. The pre-fix gate scored on the same examples Sonnet
    # had used to write the description, which made detection accuracy
    # optimistically high (Sonnet's description literally references
    # those examples).
    top_k_examples: int = 30
    delphi_held_out_n: int = 10
    n_tokens_for_activation_collection: int = 2_000_000
    activation_collection_batch_size: int = 8
    activation_collection_seq_len: int = 128
    min_firing_rate: float = 0.0005
    max_firing_rate: float = 0.10

    # ── LLM models (OpenRouter model IDs) ────────────────────────────
    explanation_model: str = "anthropic/claude-sonnet-4.6"
    organization_model: str = "anthropic/claude-sonnet-4.6"
    annotation_model: str = "anthropic/claude-haiku-4.5"
    features_per_explanation_batch: int = 10
    features_per_annotation_call: int = 50
    max_annotation_concurrency: int = 20

    # ── Supervision corpus ──────────────────────────────────────────
    corpus_dataset: str = "Skylion007/openwebtext"
    corpus_split: str = "train"
    n_sequences: int = 5000
    seq_len: int = 128
    corpus_batch_size: int = 8

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
    warmup_steps: int = 500
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
    hinge_margin: float = 1.0           # margin for hinge selectivity loss.
                                        # Also used by the free-decoder
                                        # hinge-family modes (v8.11+): set to
                                        # 0 for zero-margin hinge (sign-
                                        # correctness only, no score shaping),
                                        # 1+ for SVM-style margin hinge that
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

    # ── v2: Local model annotation ────────────────────────────────
    use_local_annotator: bool = False   # True = local model, False = API
    local_annotator_model: str = "Qwen/Qwen3-4B-Base"  # base model, no thinking, pure transformer
    local_annotation_batch_size: int = 64
    batch_positions: bool = False  # True = full-sequence JSON, False = per-token
    # Full description is the safer default: F-index mode ("F3? ") has
    # hardcoded few-shot exemplars in annotate.py that assume F0=comma and
    # F5=capitalized, which breaks for any catalog that doesn't match those
    # positions (in particular, any catalog with fewer than 6 features, or
    # any incremental discovery round). Flipped to False in v8.3; pass
    # `--use-findex` to opt back in if you know your catalog matches.
    use_findex_suffix: bool = False  # True = "F3? " (~3 tok), False = full description (~15 tok)

    # Sequences per vLLM batch in the per-token annotator. Previously
    # hardcoded to 2; the actual throughput optimum depends on feature count
    # × seq_len × KV-cache budget. Sweep {2,4,8,16,32} on your GPU and pick
    # the knee. A knob here instead of a constant lets the user tune without
    # code changes. Too high wastes KV-cache (exceeds `max_model_len *
    # max_num_seqs` budget in vLLM); too low underfeeds the engine.
    local_annotation_seq_chunk: int = 2

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

    # ── Delphi detection gate (v8.15) ────────────────────────────
    # After Sonnet generates a candidate description, run Delphi's
    # DetectionScorer on held-out activating + non-activating contexts
    # to measure whether the description actually predicts the latent's
    # firing pattern. Descriptions whose detection accuracy is below
    # `delphi_score_threshold` are dropped before annotation. Default
    # 0.7: a coin-flip judge would score 0.5 on a balanced split, so
    # 0.7 is "clearly above chance, not yet rigorous." Raise to 0.8 if
    # you want stricter gating; lower to 0.6 if too many drop.
    delphi_score_threshold: float = 0.7
    # The judge model the DetectionScorer asks. Falls back to
    # `organization_model` (Sonnet) when None.
    delphi_judge_model: str = ""
    # Whether `--step promote-loop` runs the Delphi gate between
    # crispness and the mini-prefilter. Default-on as of v8.15.
    promote_use_delphi_gate: bool = True

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
