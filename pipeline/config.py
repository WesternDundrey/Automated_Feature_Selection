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
    target_layer: int = 8
    hook_point: str = ""  # auto-set in __post_init__
    device: str = "cuda"
    model_dtype: str = "float32"  # GPT-2 Small runs in fp32

    # ── Pretrained SAE (sae_lens format) ─────────────────────────────
    sae_release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.8.hook_resid_pre"

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
    lambda_sparse: float = 1e-3
    lambda_hier: float = 0.5
    warmup_steps: int = 500
    train_fraction: float = 0.8
    seed: int = 42
    n_lista_steps: int = 0  # LISTA refinement iterations (0 = disabled)

    # ── v2: MSE feature dictionary supervision (Makelov et al. 2024) ──
    use_mse_supervision: bool = True    # False = legacy BCE mode
    direction_loss_weight: float = 1.0  # α: decoder direction alignment
    magnitude_loss_weight: float = 0.5  # β: activation magnitude alignment

    # ── v2: Local model annotation ────────────────────────────────
    use_local_annotator: bool = False   # True = local model, False = API
    local_annotator_model: str = "Qwen/Qwen3-8B"  # HuggingFace model ID
    local_annotation_batch_size: int = 64

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

    # ── Output ──────────────────────────────────────────────────────
    output_dir: str = "pipeline_data"

    def __post_init__(self):
        if not self.hook_point:
            self.hook_point = f"blocks.{self.target_layer}.hook_resid_post"
        self.output_dir = Path(self.output_dir)
        # Fallback to CPU if CUDA not available
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("WARNING: CUDA not available, falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                print("WARNING: torch not installed, defaulting to CPU")
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
