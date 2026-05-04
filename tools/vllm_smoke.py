"""
vLLM throughput smoke test — diagnose pod-vs-pipeline bottleneck.

Runs isolated benchmarks on a single GPU using vLLM directly,
bypassing our entire annotation pipeline:

  TEST 1  PREFIX-CACHE: 1000 prompts sharing a prefix
  TEST 2  NO-CACHE:     1000 prompts with unique suffixes
  TEST 3  REAL-SHAPE:   13,312 prompts (128 prefixes × ~104 features),
                        UNCONSTRAINED sampling (no allowed_token_ids)
  TEST 3-CON  REAL-SHAPE + CONSTRAINED:
                        same as Test 3 but with allowed_token_ids=[0,1]
                        — the smoking-gun A/B for our production path
  TEST 4  MAX-NUM-SEQS ABLATION: 256/512/1024/2048 (skipped with --skip-ablation)

If Test 3-CON is much slower than Test 3, allowed_token_ids logits-
processor is the bottleneck (the actual fix is to drop it and parse
the output token in Python).

Run with:
  CUDA_VISIBLE_DEVICES=0 python tools/vllm_smoke.py
"""

from __future__ import annotations

import argparse
import time


def _bench(llm, prompts, params, label: str) -> dict:
    """Run llm.generate(prompts, params), report throughput."""
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"  n_prompts = {len(prompts):,}")
    print(f"{'─' * 70}")

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0

    n = len(prompts)
    rate = n / elapsed if elapsed > 0 else 0
    out_tok_total = sum(len(o.outputs[0].token_ids) for o in outputs)
    out_rate = out_tok_total / elapsed if elapsed > 0 else 0

    print(f"  RESULT: {rate:.0f} prompts/sec  "
          f"({elapsed:.1f}s, {out_rate:.0f} output toks/sec)")
    return {
        "label": label,
        "n_prompts": n,
        "elapsed_s": elapsed,
        "prompts_per_sec": rate,
        "output_toks_per_sec": out_rate,
    }


def _make_token_ids_prompts(tokenizer, prefix_text: str, suffix_text: str,
                            n: int, vary_suffix: bool) -> list[dict]:
    """Build n prompts with token IDs (matches our pipeline's call shape)."""
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
    prompts = []
    for i in range(n):
        if vary_suffix:
            sfx = f"{suffix_text} (variant {i % 100})"
        else:
            sfx = suffix_text
        suffix_ids = tokenizer.encode(sfx, add_special_tokens=False)
        prompts.append({"prompt_token_ids": list(prefix_ids + suffix_ids)})
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--n", type=int, default=1000,
                        help="Prompts per benchmark (1, 2, 4)")
    parser.add_argument("--n-real", type=int, default=13312,
                        help="Prompts for real-shape test (#3)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip the max-num-seqs ablation (saves ~5 min)")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=" * 70)
    print("vLLM smoke test")
    print("=" * 70)
    print(f"  model:                    {args.model}")
    print(f"  prompts per bench (1,2,4): {args.n}")
    print(f"  prompts for real-shape (3): {args.n_real}")
    print(f"  max_num_batched_tokens:   {args.max_num_batched_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ─── Build LLM once (max_num_seqs=2048 to test true continuous batching) ───
    print(f"\nLoading vLLM (max_num_seqs=2048, max_num_batched_tokens={args.max_num_batched_tokens})...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_num_seqs=2048,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=0.9,
        max_model_len=512,
        disable_log_stats=False,
    )
    params = SamplingParams(max_tokens=1, temperature=0)

    # v8.20.10.1: also build a CONSTRAINED params object using
    # allowed_token_ids — the smoking-gun A/B for our production path.
    tok_0_id = tokenizer.encode("0", add_special_tokens=False)[0]
    tok_1_id = tokenizer.encode("1", add_special_tokens=False)[0]
    params_constrained = SamplingParams(
        max_tokens=1,
        temperature=0,
        allowed_token_ids=[tok_0_id, tok_1_id],
    )
    print(f"  tok_0_id = {tok_0_id}, tok_1_id = {tok_1_id}")

    results = []

    # ─── Test 1: prefix cache ───
    SHARED_PREFIX = (
        "You are reading text and answering yes/no questions about the "
        "highlighted token. The text is below. "
        "Text: 'The quick brown fox jumps over the lazy dog and runs to "
        "the river where it stops to drink water from the cool stream.' "
        "Token: 'fox'."
    )
    QUESTION = " Is the token a noun?"
    prompts1 = _make_token_ids_prompts(
        tokenizer, SHARED_PREFIX, QUESTION, args.n, vary_suffix=False,
    )
    results.append(_bench(
        llm, prompts1, params,
        "TEST 1 — PREFIX CACHE: 1000 prompts sharing identical prefix",
    ))

    # ─── Test 2: no cache ───
    prompts2 = _make_token_ids_prompts(
        tokenizer, SHARED_PREFIX, QUESTION, args.n, vary_suffix=True,
    )
    results.append(_bench(
        llm, prompts2, params,
        "TEST 2 — NO CACHE: 1000 prompts with unique suffixes",
    ))

    # ─── Test 3: real-shape (mimics our annotation workload) ───
    # 128 prefixes × (n_real / 128) features each — looks like our
    # prefix-block batching at prefix_block=128.
    n_prefix_groups = 128
    features_per_group = max(1, args.n_real // n_prefix_groups)
    real_prompts = []
    for pi in range(n_prefix_groups):
        prefix_text = (
            f"You are reading text. Position {pi}. "
            f"The full text reads: 'Once upon a time in a {pi % 50}-letter "
            f"village far away there lived a curious child who explored.' "
            f"Token: 'village'."
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
        for fi in range(features_per_group):
            sfx = f" Is the token feature_{fi}?"
            sfx_ids = tokenizer.encode(sfx, add_special_tokens=False)
            real_prompts.append({"prompt_token_ids": list(prefix_ids + sfx_ids)})
    real_prompts = real_prompts[:args.n_real]
    results.append(_bench(
        llm, real_prompts, params,
        f"TEST 3 — REAL SHAPE (UNCONSTRAINED): {len(real_prompts):,} prompts",
    ))

    # ─── Test 3-CON: SAME prompts WITH allowed_token_ids constraint ───
    # This is the A/B that isolates whether constrained decoding is
    # the bottleneck. If 3-CON is much slower than 3, that's the bug
    # in our production path.
    results.append(_bench(
        llm, real_prompts, params_constrained,
        f"TEST 3-CON — REAL SHAPE + CONSTRAINED (allowed_token_ids=[0,1])",
    ))

    # ─── Test 4: max_num_seqs ablation (skipped with --skip-ablation) ───
    if not args.skip_ablation:
        print(f"\n{'═' * 70}")
        print(f"  TEST 4 — max_num_seqs ABLATION")
        print(f"{'═' * 70}")
        print("  Note: this rebuilds LLM 4 times (slow startup each ~30s).")
        print("  Skip with --skip-ablation if you don't need this.")

        del llm
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for mns in [256, 512, 1024, 2048]:
            print(f"\n  Building LLM with max_num_seqs={mns}...")
            llm_abl = LLM(
                model=args.model,
                dtype="bfloat16",
                enable_prefix_caching=True,
                max_num_seqs=mns,
                max_num_batched_tokens=args.max_num_batched_tokens,
                gpu_memory_utilization=0.9,
                max_model_len=512,
                disable_log_stats=False,
            )
            r = _bench(
                llm_abl, prompts1, params,
                f"max_num_seqs={mns} | shared-prefix prompts",
            )
            r["max_num_seqs"] = mns
            results.append(r)
            del llm_abl
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ─── Final summary ───
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Test':<60}  {'p/s':>8}  {'tok/s':>8}")
    for r in results:
        print(f"  {r['label'][:60]:<60}  "
              f"{r['prompts_per_sec']:>8.0f}  "
              f"{r['output_toks_per_sec']:>8.0f}")

    print(f"\n  Diagnosis hints:")
    t1 = results[0]['prompts_per_sec']
    t2 = results[1]['prompts_per_sec']
    t3 = results[2]['prompts_per_sec']
    t3con = results[3]['prompts_per_sec']
    print(f"    Test 1 (cache hit):                  {t1:>7.0f} p/s")
    print(f"    Test 2 (no cache):                   {t2:>7.0f} p/s")
    print(f"    Test 3 (real shape, unconstrained):  {t3:>7.0f} p/s")
    print(f"    Test 3-CON (real + allowed_tokens):  {t3con:>7.0f} p/s")
    if t3 > 0:
        slowdown = t3 / max(t3con, 1)
        print(f"    Constraint slowdown (3 / 3-CON):     {slowdown:>7.1f}×")

    print(f"\n  Interpretation:")
    if t3con < t3 * 0.3 and t3 > 1000:
        print(f"    >>> SMOKING GUN: allowed_token_ids constrained decoding")
        print(f"        is {t3/max(t3con,1):.0f}× SLOWER than unconstrained.")
        print(f"        Production uses allowed_token_ids → that's the bug.")
        print(f"        FIX: drop allowed_token_ids, parse the generated")
        print(f"        token's first char (it's almost always '0' or '1'")
        print(f"        on a base model with binary-question prompts).")
    elif t3 > 1000 and t3con > 1000:
        print(f"    Both unconstrained and constrained are fast. The")
        print(f"    production slowdown is elsewhere — check max_model_len,")
        print(f"    subprocess config, or actual prompt lengths in production.")
    elif t3 > 100:
        print(f"    vLLM works but is slower than expected even unconstrained.")
        print(f"    Tune max_num_batched_tokens higher; check max_num_seqs.")
    else:
        print(f"    vLLM broken on this pod (CUDA/driver compat issue).")


if __name__ == "__main__":
    main()
