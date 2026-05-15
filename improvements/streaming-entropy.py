"""Single-pass entropy scoring: fuse the entropy pre-pass into the loss pass.

Implements Implementation/Code Improvements §2 of 05-improvements.tex.

ReasonMaxxer computes H_t in a pre-pass that throws logits away, then redoes
a forward pass for the contrastive loss. Both passes need the same softmax
distribution, so they can be fused. Tradeoff: O(|V|*T) cache vs 2x compute.
CPU-runnable, <60s. Uses Qwen2.5-0.5B-Instruct; falls back to distilgpt2.
"""

import time
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK = "distilgpt2"
PROMPT = "The capital of France is Paris, and the capital of Japan is"


def _load():
    for name in (MODEL_NAME, FALLBACK):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
            return name, tok, mdl
        except Exception as e:
            print(f"[warn] {name}: {e}", file=sys.stderr)
    raise RuntimeError("No model available")


@torch.no_grad()
def two_pass_entropy(model, tokens) -> dict:
    """Paper's approach: pass-1 throws logits away after H_t; pass-2 redoes them."""
    t0 = time.perf_counter()
    logits1 = model(tokens).logits          # [1, T, V]
    probs = F.softmax(logits1, dim=-1)
    H_t = -(probs * probs.clamp_min(1e-12).log()).sum(-1).squeeze(0)  # [T]
    del logits1, probs                       # explicitly drop the tensor
    t1 = time.perf_counter()

    logits2 = model(tokens).logits          # second forward pass
    labels = tokens[0, 1:]
    loss = F.cross_entropy(logits2[0, :-1], labels)  # contrastive-loss surrogate
    t2 = time.perf_counter()
    return {"H_t": H_t, "loss": loss.item(), "t_pass1": t1 - t0, "t_pass2": t2 - t1, "total": t2 - t0}


@torch.no_grad()
def single_pass_entropy(model, tokens) -> dict:
    """Fused approach: one forward, cache logits, derive H_t and loss from cache."""
    t0 = time.perf_counter()
    logits = model(tokens).logits           # [1, T, V] — KEPT in memory
    probs = F.softmax(logits, dim=-1)
    H_t = -(probs * probs.clamp_min(1e-12).log()).sum(-1).squeeze(0)
    labels = tokens[0, 1:]
    loss = F.cross_entropy(logits[0, :-1], labels)  # reuse same logits tensor
    t1 = time.perf_counter()
    cache_bytes = logits.element_size() * logits.numel()
    return {"H_t": H_t, "loss": loss.item(), "t_combined": t1 - t0,
            "cache_MB": cache_bytes / (1024 ** 2), "vocab": logits.shape[-1], "T": logits.shape[1]}


def measure() -> dict:
    name, tok, mdl = _load()
    tokens = tok(PROMPT, return_tensors="pt").input_ids[:, :20]
    tp = two_pass_entropy(mdl, tokens)
    sp = single_pass_entropy(mdl, tokens)
    speedup_pct = 100.0 * (tp["total"] - sp["t_combined"]) / tp["total"]
    # Per-position cache cost in bytes; "breakeven" = T at which cache hits 1 GB.
    bytes_per_pos = 4 * sp["vocab"]
    breakeven_T_1GB = int((1024 ** 3) / bytes_per_pos)
    return {
        "model": name, "T": sp["T"], "vocab": sp["vocab"],
        "two_pass_total_s": tp["total"], "single_pass_s": sp["t_combined"],
        "speedup_pct": speedup_pct, "cache_MB": sp["cache_MB"],
        "bytes_per_position": bytes_per_pos, "breakeven_T_at_1GB": breakeven_T_1GB,
        "loss_match": abs(tp["loss"] - sp["loss"]) < 1e-4,
    }


def main():
    m = measure()
    print(f"=== Two-pass (paper) ===\nTotal: {m['two_pass_total_s']:.3f}s  loss_match={m['loss_match']}")
    print(f"=== Single-pass (cached logits) ===")
    print(f"Combined: {m['single_pass_s']:.3f}s   Cache: {m['cache_MB']:.2f} MB  (T={m['T']}, |V|={m['vocab']})")
    print(f"=== Comparison ===")
    print(f"Speedup: {m['speedup_pct']:.1f}%  ({m['two_pass_total_s']:.3f} -> {m['single_pass_s']:.3f} s)")
    print(f"Per-pos: {m['bytes_per_position']/1024:.1f} KB; 1 GB fits ~{m['breakeven_T_at_1GB']} positions   model={m['model']}")
    return m


if __name__ == "__main__":
    main()
