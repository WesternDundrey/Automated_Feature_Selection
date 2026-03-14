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
    model_name: str = "google/gemma-2-2b"
    target_layer: int = 20
    hook_point: str = ""  # auto-set in __post_init__
    device: str = "cuda"
    model_dtype: str = "bfloat16"

    # ── Pretrained SAE (sae_lens format) ─────────────────────────────
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_id: str = "layer_20/width_16k/canonical"

    # ── Feature selection from pretrained SAE ────────────────────────
    n_latents_to_explain: int = 500
    top_k_examples: int = 20
    n_tokens_for_activation_collection: int = 2_000_000
    activation_collection_batch_size: int = 8
    activation_collection_seq_len: int = 128
    min_firing_rate: float = 0.0005
    max_firing_rate: float = 0.10

    # ── LLM models ──────────────────────────────────────────────────
    explanation_model: str = "claude-sonnet-4-6"
    organization_model: str = "claude-sonnet-4-6"
    annotation_model: str = "claude-haiku-4-5-20251001"
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

    # ── Annotation robustness ────────────────────────────────────
    annotation_max_retries: int = 3
    annotation_retry_base_delay: float = 1.0

    # ── Output ──────────────────────────────────────────────────────
    output_dir: str = "pipeline_data"

    def __post_init__(self):
        if not self.hook_point:
            self.hook_point = f"blocks.{self.target_layer}.hook_resid_post"
        self.output_dir = Path(self.output_dir)

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
