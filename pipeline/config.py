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
    top_k_examples: int = 20
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

    # ── v2: Supervision mode ────────────────────────────────────────
    # "hybrid" = BCE selectivity + cosine direction alignment (recommended)
    # "mse"    = MSE magnitude + cosine direction (Makelov-inspired)
    # "bce"    = legacy BCE only (no decoder alignment)
    supervision_mode: str = "hybrid"
    use_mse_supervision: bool = True    # DEPRECATED: use supervision_mode instead
    direction_loss_weight: float = 1.0  # α: decoder direction alignment (hybrid/mse)
    magnitude_loss_weight: float = 0.5  # β: activation magnitude alignment (mse only)
    selectivity_loss: str = "bce"       # "bce", "hinge", or "none"
    hinge_margin: float = 1.0           # margin for hinge selectivity loss

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
    # Position 0 in a transformer has degenerate attention (only self), no
    # prior context, anomalous residual-stream magnitude/direction, and
    # acts as an attention sink. Including it in supervised SAE analysis
    # corrupts:
    #   - target_dirs: sequence-level features collapse to the same
    #     "position-0 vs rest" direction if many positives fall at pos 0.
    #   - reconstruction + R²: dominated by the easy-to-reconstruct BOS
    #     direction rather than the semantic content.
    #   - promote-loop: the top-ΔR² U latents are often BOS detectors,
    #     wasting a round on a causally-useless token.
    # Standard mech-interp practice: mask position 0 from all analysis.
    # Applied at load time (activations = activations[:, n:]); cached
    # tensors don't need re-extraction.
    mask_first_n_positions: int = 1

    # Maximum fraction of annotation chunks that may end up zero-labeled
    # (after max retries) before the run aborts. A few stragglers are
    # tolerable; >10% means the prompt or the annotator is broken, and
    # training on those labels would silently corrupt the catalog.
    annotation_max_failure_rate: float = 0.10

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
