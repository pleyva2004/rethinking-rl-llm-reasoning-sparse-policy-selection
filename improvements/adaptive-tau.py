"""Hyperparameter-free quantile-based gating for entropy-localised SFT.

Implements §"Hyperparameter-free quantile-based gating" of 05-improvements.tex.

Compares fixed absolute threshold tau (the paper's default) vs quantile-based
gating across two prompts whose entropy distributions differ.

The claim: fixed tau is brittle to entropy-distribution shift between inputs
(or between model scales), while quantile gating produces a stable fraction
of gated positions across inputs.

CPU-runnable, <60s. Uses Qwen2.5-0.5B-Instruct; falls back to distilgpt2.
"""

import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FIXED_TAU = 1.4           # the paper's default
QUANTILE_Q = 0.95         # gate top 5% of positions per rollout
MAX_NEW_TOKENS = 96

# Two prompts chosen to elicit different entropy distributions:
# 1. A math word problem — structured reasoning, expected to have a few
#    high-entropy "decision" points (which operation? which substitution?).
# 2. Open-ended creative prompt — diffuse uncertainty everywhere.
PROMPTS = [
    (
        "math",
        "Solve step by step. A train leaves station A at 60 km/h. Two hours "
        "later, a second train leaves the same station in the same direction "
        "at 90 km/h. How many hours after the first train departs will the "
        "second train catch up?\nAnswer:",
    ),
    (
        "creative",
        "Write a short opening paragraph for a mystery novel set in a "
        "lighthouse during a storm.\nAnswer:",
    ),
]


def load_model(name: str):
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        model.eval()
        return tok, model
    except Exception as e:
        print(f"Failed to load {name}: {e}; falling back to distilgpt2.")
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model.eval()
        return tok, model


def entropy_from_logits(logits: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return float(-(probs * log_probs).sum().item())


def generate_with_entropies(tok, model, prompt: str, max_new: int):
    input_ids = tok(prompt, return_tensors="pt").input_ids
    generated = input_ids.clone()
    entropies = []
    with torch.no_grad():
        for _ in range(max_new):
            out = model(generated)
            next_logits = out.logits[0, -1, :]
            entropies.append(entropy_from_logits(next_logits))
            next_id = int(torch.argmax(next_logits).item())
            generated = torch.cat(
                [generated, torch.tensor([[next_id]])], dim=1
            )
            if hasattr(tok, "eos_token_id") and next_id == tok.eos_token_id:
                break
    return entropies


def fixed_tau_gate(entropies, tau):
    return [i for i, h in enumerate(entropies) if h > tau]


def quantile_gate(entropies, q):
    if not entropies:
        return []
    sorted_e = sorted(entropies)
    threshold = sorted_e[int(q * len(sorted_e))]
    return [i for i, h in enumerate(entropies) if h > threshold]


def main():
    print(f"Model: {MODEL_NAME}")
    print()
    tok, model = load_model(MODEL_NAME)

    rows = []
    for label, prompt in PROMPTS:
        entropies = generate_with_entropies(tok, model, prompt, MAX_NEW_TOKENS)
        n = len(entropies)
        fixed = fixed_tau_gate(entropies, FIXED_TAU)
        quant = quantile_gate(entropies, QUANTILE_Q)
        mean_h = sum(entropies) / n if n else 0.0
        max_h = max(entropies) if entropies else 0.0
        rows.append({
            "label": label,
            "n": n,
            "mean_h": mean_h,
            "max_h": max_h,
            "fixed_count": len(fixed),
            "fixed_frac": len(fixed) / n if n else 0.0,
            "quant_count": len(quant),
            "quant_frac": len(quant) / n if n else 0.0,
        })

    print(f"=== fixed tau = {FIXED_TAU} ===")
    for r in rows:
        print(
            f"  {r['label']:>10}: gated {r['fixed_count']}/{r['n']} "
            f"({r['fixed_frac']*100:.1f}%)   "
            f"mean H={r['mean_h']:.2f}  max H={r['max_h']:.2f}"
        )

    print()
    print(f"=== quantile q = {QUANTILE_Q} ===")
    for r in rows:
        print(
            f"  {r['label']:>10}: gated {r['quant_count']}/{r['n']} "
            f"({r['quant_frac']*100:.1f}%)   (stable target = "
            f"{(1-QUANTILE_Q)*100:.0f}%)"
        )

    print()
    fixed_fracs = [r["fixed_frac"] for r in rows]
    quant_fracs = [r["quant_frac"] for r in rows]
    fixed_spread = max(fixed_fracs) - min(fixed_fracs)
    quant_spread = max(quant_fracs) - min(quant_fracs)
    print(
        f"Fixed-tau gated-fraction spread across prompts: "
        f"{fixed_spread*100:.1f} percentage points"
    )
    print(
        f"Quantile-q  gated-fraction spread across prompts: "
        f"{quant_spread*100:.1f} percentage points"
    )
    print()
    if quant_spread < fixed_spread:
        ratio = fixed_spread / quant_spread if quant_spread > 0 else float("inf")
        print(
            f"Quantile gating is {ratio:.1f}x more stable across prompts. "
            "Claim supported."
        )
    else:
        print(
            "Quantile gating was NOT more stable on this run — would invalidate "
            "the claim on this model scale."
        )


if __name__ == "__main__":
    sys.exit(main())
