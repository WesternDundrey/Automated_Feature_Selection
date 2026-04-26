"""
Per-feature human-audit dump.

For a single feature_id, reads `tokens.pt`, `annotations.pt`,
`activations.pt`, `supervised_sae.pt`, and `evaluation.json`, runs the
SAE forward pass on a deterministic test-split sample, and writes a
markdown file that lists:

  • TP examples — annotator says yes AND the SAE fires above the
    feature's calibrated threshold. These should LOOK like the
    feature description.
  • FP examples — SAE fires but the annotator says no. The SAE thinks
    these match; if they don't actually match the description, the
    SAE has a precision problem.
  • FN examples — annotator says yes but the SAE doesn't fire. The
    annotator thinks these match; if they don't actually look like the
    description, the annotator has a precision problem (description
    too broad, fuzzy boundary).

Each example shows the highlighted target token plus ~20 surrounding
tokens of context, the SAE pre-activation score, and a TRUE/FALSE
checkbox the auditor flips by hand.

Closes the reviewer's #4 recommendation: "human-audit 10 positives
and 10 negatives for each newly promoted feature." We dump pos / FP /
FN at the same time so a single audit pass measures BOTH the SAE's
precision/recall AND the annotator's precision (a noisy annotator and
a confused SAE are easy to confuse without separating these buckets).

Usage:
    python -m pipeline.run --step audit-feature \\
        --feature-id syntactic_construction.relative_clause \\
        --audit-n 10

    # Audit every catalog feature whose cal_F1 < 0.4
    python -m pipeline.run --step audit-feature \\
        --feature-id "" --audit-cal-f1-below 0.4

The output is a markdown file under `pipeline_data/audits/` named
`audit_<feature_id>.md` (or `audit_batch.md` for the batch mode). It
re-runs the SAE forward pass each call (no caching) but only on the
test split, so it's cheap (~30s on a single GPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _decode_context(tokenizer, token_ids: list[int], target_pos: int,
                    window: int = 20) -> str:
    """Render a window of tokens around `target_pos` with the target wrapped
    in **bold**. Each token is decoded individually so subword boundaries are
    visible to the auditor."""
    start = max(0, target_pos - window)
    end = min(len(token_ids), target_pos + window + 1)
    parts: list[str] = []
    for i in range(start, end):
        tok = tokenizer.decode([token_ids[i]])
        # Escape markdown specials in the rendered tokens so the auditor
        # sees the actual subword strings, not interpreted markdown.
        tok = tok.replace("|", "\\|").replace("`", "\\`").replace("*", "\\*")
        if i == target_pos:
            parts.append(f"**>>{tok}<<**")
        else:
            parts.append(tok)
    return "".join(parts).replace("\n", "\\n")


def _resolve_threshold(eval_data: dict, feature_id: str) -> float | None:
    """Look up the calibrated threshold from evaluation.json for a feature.
    Returns None if the feature is missing or has no threshold (rare/skipped
    features)."""
    for f in eval_data.get("features", []):
        if f.get("id") == feature_id:
            t = f.get("cal_threshold")
            return float(t) if t is not None else None
    return None


def _sample_indices(
    mask: np.ndarray, n: int, seed: int,
) -> np.ndarray:
    """Sample up to `n` indices from positions where `mask` is True.
    Sampling is deterministic across runs for the same seed."""
    pos = np.flatnonzero(mask)
    if len(pos) == 0:
        return pos
    if len(pos) <= n:
        return pos
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(pos, size=n, replace=False))


def _audit_one_feature(
    feature: dict,
    feature_idx: int,
    sup_pre_test: np.ndarray,        # (N_test, n_features)
    annotations_test: np.ndarray,    # (N_test, n_features) — bool
    test_position_to_seq: np.ndarray,  # (N_test,) original sequence idx
    test_position_to_pos: np.ndarray,  # (N_test,) original within-seq pos
    threshold: float,
    tokens: torch.Tensor,            # (N, T)
    tokenizer,
    n_per_bucket: int,
    seed: int,
) -> str:
    """Build the per-feature markdown body for a single audit. Picks the
    top-scoring positions in each bucket so the human reviewer sees the
    SAE's "best efforts" rather than uniformly-random examples (a uniform
    sample of FPs is dominated by tokens that just barely cleared the
    threshold and gives the SAE less benefit of the doubt)."""
    feature_id = feature["id"]
    description = feature.get("description", "(no description)")

    scores = sup_pre_test[:, feature_idx]
    gt = annotations_test[:, feature_idx].astype(bool)
    pred = scores > threshold

    tp_mask = pred & gt
    fp_mask = pred & ~gt
    fn_mask = ~pred & gt

    # Rank within each bucket by SCORE so the auditor sees the SAE's
    # most-confident calls first. For FN, "highest score" means closest
    # to threshold — those are the borderline misses, which are the most
    # informative for moving the threshold or fixing the description.
    def _top_n(mask: np.ndarray, n: int, by_score: bool, descending: bool) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return idx
        if not by_score:
            return _sample_indices(mask, n, seed=seed + feature_idx)
        # Sort by score
        bucket_scores = scores[idx]
        order = np.argsort(-bucket_scores if descending else bucket_scores)
        return idx[order[:n]]

    tp_idx = _top_n(tp_mask, n_per_bucket, by_score=True,  descending=True)
    fp_idx = _top_n(fp_mask, n_per_bucket, by_score=True,  descending=True)
    # FN: pick the highest-scoring misses (closest to threshold from below).
    fn_idx = _top_n(fn_mask, n_per_bucket, by_score=True,  descending=True)

    n_pos_total = int(gt.sum())
    n_pred_total = int(pred.sum())
    n_tp = int(tp_mask.sum())
    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    prec = n_tp / max(n_pred_total, 1)
    rec = n_tp / max(n_pos_total, 1)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    lines: list[str] = []
    lines.append(f"## `{feature_id}`\n")
    lines.append(f"**Description:** {description}\n")
    lines.append(f"**Type:** {feature.get('type', '?')}  ")
    lines.append(f"**Role:** {feature.get('role', 'discovery')}  ")
    lines.append(f"**cal_threshold:** {threshold:.4f}  ")
    lines.append(f"**Test split:** N={len(scores):,}  "
                 f"positives={n_pos_total}  predictions={n_pred_total}\n")
    lines.append(f"**Test cal_F1:** {f1:.4f}  "
                 f"(precision={prec:.3f}, recall={rec:.3f})\n")

    def _fmt_bucket(name: str, idx_arr: np.ndarray, total: int,
                    instructions: str) -> None:
        lines.append(f"\n### {name}  ({len(idx_arr)} of {total} shown)\n")
        lines.append(f"_{instructions}_\n")
        if len(idx_arr) == 0:
            lines.append("\n*(none)*\n")
            return
        for rank, ti in enumerate(idx_arr.tolist(), start=1):
            n = int(test_position_to_seq[ti])
            t = int(test_position_to_pos[ti])
            score = float(scores[ti])
            ctx = _decode_context(
                tokenizer,
                tokens[n].tolist(),
                t,
                window=20,
            )
            lines.append(
                f"\n{rank}. `score={score:+.4f}`  "
                f"`(seq={n}, pos={t})`  "
                f"AUDITOR: [ ] match  [ ] not_match\n"
            )
            lines.append(f"   > {ctx}\n")

    _fmt_bucket(
        "True positives (TP) — SAE fires AND annotator says yes",
        tp_idx, n_tp,
        "These should clearly match the description. If many do not, "
        "the description is fuzzy / annotator is wrong on the 'yes' side."
    )
    _fmt_bucket(
        "False positives (FP) — SAE fires but annotator says no",
        fp_idx, n_fp,
        "If these LOOK like the description, the annotator missed positives "
        "(annotator-recall problem). If they don't, the SAE is over-firing."
    )
    _fmt_bucket(
        "False negatives (FN) — annotator says yes, SAE doesn't fire",
        fn_idx, n_fn,
        "If these LOOK like the description, the SAE missed real positives "
        "(SAE-recall problem). If they don't, the annotator over-fired "
        "(annotator-precision problem)."
    )

    return "\n".join(lines)


def run(
    cfg: Config = None,
    feature_id: str | None = None,
    audit_n: int = 10,
    cal_f1_below: float | None = None,
) -> dict:
    """Audit one or many features.

    Args:
        feature_id: specific feature to audit. Empty string + cal_f1_below
            audits every feature whose test cal_F1 is below the threshold.
        audit_n: examples per bucket (TP / FP / FN), default 10.
        cal_f1_below: if set, batch-audit features below this cal_F1.
    """
    if cfg is None:
        cfg = Config()

    audit_dir = cfg.output_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # ── Required inputs ──
    for path, label in [
        (cfg.catalog_path,           "feature_catalog.json"),
        (cfg.tokens_path,            "tokens.pt"),
        (cfg.activations_path,       "activations.pt"),
        (cfg.annotations_path,       "annotations.pt"),
        (cfg.checkpoint_path,        "supervised_sae.pt"),
        (cfg.checkpoint_config_path, "supervised_sae_config.pt"),
        (cfg.eval_path,              "evaluation.json"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"audit-feature requires {label} at {path}. Run train + "
                f"evaluate first."
            )

    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]
    eval_data = json.loads(cfg.eval_path.read_text())

    # Decide which features to audit.
    if feature_id:
        target_ids = [feature_id]
    elif cal_f1_below is not None:
        below: list[tuple[float, str]] = []
        for f in eval_data.get("features", []):
            cal_f1 = f.get("cal_f1")
            if cal_f1 is None:
                continue
            if cal_f1 < cal_f1_below:
                below.append((float(cal_f1), f["id"]))
        below.sort()  # worst first
        target_ids = [fid for _, fid in below]
        print(f"  Batch audit: {len(target_ids)} features have cal_F1 < {cal_f1_below}")
    else:
        raise ValueError(
            "audit-feature needs either --feature-id <id> or "
            "--audit-cal-f1-below <threshold>."
        )

    if not target_ids:
        print("  Nothing to audit.")
        return {"n_audited": 0, "audit_paths": []}

    # ── Load shared resources once (heavy loads cached across the batch) ──
    print("Loading SAE checkpoint...")
    from .train import load_trained_sae
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
    )
    sae = load_trained_sae(model_cfg)
    sae.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae.eval().to(cfg.device)

    print("Loading activations + tokens + annotations...")
    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)
    tokens_full = torch.load(cfg.tokens_path, weights_only=True)

    # Apply the SAME position mask train.py / evaluate.py applies, so the
    # split indices align with what the SAE actually saw.
    from .position_mask import mask_leading
    activations, annotations = mask_leading(activations, annotations, cfg=cfg)
    mask_n = int(getattr(cfg, "mask_first_n_positions", 0))

    N, T, _d = activations.shape
    n_features = annotations.shape[-1]
    if len(features) != n_features:
        # annotations_meta.json should have caught this, but defend in depth.
        raise RuntimeError(
            f"Catalog has {len(features)} features but annotations.pt has "
            f"{n_features}. Re-run --step annotate to regenerate."
        )

    # Recover the test split exactly as evaluate.py does.
    if cfg.split_path.exists():
        perm = torch.load(cfg.split_path, weights_only=True)
    else:
        print("WARNING: split_indices.pt not found, regenerating via RNG")
        from .train import set_seed
        set_seed(cfg.seed)
        perm = torch.randperm(N * T)

    n_total = N * T
    split_idx = int(cfg.train_fraction * n_total)
    remaining = n_total - split_idx
    val_size = remaining // 2
    test_idx = perm[split_idx + val_size:]

    # Map every flattened test position back to (seq, pos) so the markdown
    # dump can show real corpus context.
    seq_of = (test_idx // T).numpy()
    pos_of = (test_idx % T).numpy()
    if mask_n > 0:
        # Restore the offset we sliced off so the printed positions match
        # the on-disk tokens.pt.
        pos_of = pos_of + mask_n

    # SAE forward on test split, recording supervised pre-activations only.
    print(f"Running SAE forward on {len(test_idx):,} test positions...")
    x_flat = activations.reshape(-1, _d)
    y_flat = annotations.reshape(-1, n_features)
    x_test = x_flat[test_idx]
    y_test = y_flat[test_idx]

    sup_pre_chunks = []
    bs = cfg.batch_size
    with torch.no_grad():
        for i in range(0, x_test.shape[0], bs):
            xb = x_test[i:i + bs].to(cfg.device)
            _, sp, _, _ = sae(xb)
            sup_pre_chunks.append(sp.cpu())
    sup_pre_test = torch.cat(sup_pre_chunks).numpy()
    annotations_test = y_test.numpy()

    # Tokenizer for decoding context.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Per-feature dump ──
    audit_paths: list[str] = []
    by_id_idx: dict[str, int] = {f["id"]: i for i, f in enumerate(features)}

    body_chunks: list[str] = []
    for fid in target_ids:
        if fid not in by_id_idx:
            print(f"  WARNING: '{fid}' not in catalog — skipping")
            continue
        feat_idx = by_id_idx[fid]
        feat = features[feat_idx]
        threshold = _resolve_threshold(eval_data, fid)
        if threshold is None:
            print(f"  WARNING: '{fid}' has no cal_threshold in "
                  f"evaluation.json — skipping (rare feature?)")
            continue

        body = _audit_one_feature(
            feature=feat,
            feature_idx=feat_idx,
            sup_pre_test=sup_pre_test,
            annotations_test=annotations_test,
            test_position_to_seq=seq_of,
            test_position_to_pos=pos_of,
            threshold=threshold,
            tokens=tokens_full,
            tokenizer=tokenizer,
            n_per_bucket=audit_n,
            seed=cfg.seed,
        )
        body_chunks.append(body)

        # Single-feature mode writes one file per call. Batch mode collates
        # everything into one big file for one audit pass.
        if feature_id and not cal_f1_below:
            out_path = audit_dir / f"audit_{fid.replace('.', '_')}.md"
            cal_f1_recorded = next(
                (f.get("cal_f1") for f in eval_data.get("features", [])
                 if f.get("id") == fid),
                None,
            )
            cal_f1_str = (
                f"{cal_f1_recorded:.4f}" if cal_f1_recorded is not None
                else "(not in evaluation.json)"
            )
            header = (
                f"# Feature audit: `{fid}`\n\n"
                f"_Per-bucket sample size: {audit_n}._\n\n"
                f"_Test cal_F1 from evaluation.json: {cal_f1_str}._\n\n"
                "---\n\n"
            )
            out_path.write_text(header + body)
            print(f"  Wrote {out_path}")
            audit_paths.append(str(out_path))
            body_chunks.pop()  # already flushed

    # Batch mode: collate all into one file (sorted by cal_F1 ascending —
    # worst first, so the auditor reads the most suspicious features first).
    if cal_f1_below is not None and body_chunks:
        out_path = audit_dir / "audit_batch.md"
        header = (
            f"# Batch feature audit  (cal_F1 < {cal_f1_below})\n\n"
            f"_Audited {len(body_chunks)} features, "
            f"{audit_n} examples per bucket._\n\n"
            "Order: worst cal_F1 first. Go down the list and flip the "
            "[ ] checkboxes to [x] for true matches. Then count: a feature "
            "with mostly true matches under TP and false matches under FP "
            "is correctly captured. A feature with false matches under TP "
            "is incorrectly captured (drop or rewrite). A feature with "
            "many false matches under FN is fine; the annotator hallucinated.\n\n"
            "---\n\n"
        )
        out_path.write_text(header + "\n\n---\n\n".join(body_chunks))
        print(f"  Wrote {out_path}")
        audit_paths.append(str(out_path))

    print(f"\n  Audit complete. {len(audit_paths)} file(s) written under "
          f"{audit_dir}.")
    return {
        "n_audited": len(audit_paths),
        "audit_paths": audit_paths,
    }
