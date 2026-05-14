"""Probe of Akgül et al. 2026 (arxiv:2605.06241) — entropy-gated decision points.

Generates a CoT for a math problem under greedy decoding, computes per-token
predictive entropy from the model's own logits, and reports which positions
exceed the decision-point threshold tau.

If the paper's central locator claim holds, the high-entropy positions should
cluster at semantically meaningful "branching" tokens, not at surface noise.
"""

import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TAU = 1.4  # nats; paper's empirically optimal threshold band is [1.2, 1.8]
MAX_NEW_TOKENS = 96
PROMPT = (
    "Solve step by step. A train leaves station A at 60 km/h. "
    "Two hours later, a second train leaves the same station in the same "
    "direction at 90 km/h. How many hours after the first train departs "
    "will the second train catch up?\nAnswer:"
)


def load_model(name: str):
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        model.eval()
        return tok, model
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        print("Falling back to distilgpt2.")
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model.eval()
        return tok, model


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Shannon entropy in nats of a single next-token distribution."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return float(-(probs * log_probs).sum().item())


def generate_with_entropies(tok, model, prompt: str, max_new: int):
    input_ids = tok(prompt, return_tensors="pt").input_ids
    generated = input_ids.clone()
    entropies = []
    chosen_tokens = []

    with torch.no_grad():
        for _ in range(max_new):
            out = model(generated)
            next_logits = out.logits[0, -1, :]
            H = entropy_from_logits(next_logits)
            next_tok_id = int(torch.argmax(next_logits).item())

            entropies.append(H)
            chosen_tokens.append(next_tok_id)
            generated = torch.cat(
                [generated, torch.tensor([[next_tok_id]])], dim=1
            )

            if hasattr(tok, "eos_token_id") and next_tok_id == tok.eos_token_id:
                break

    decoded = [tok.decode([tid]) for tid in chosen_tokens]
    return decoded, entropies


def main():
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold tau: {TAU} nats\n")

    tok, model = load_model(MODEL_NAME)
    tokens, entropies = generate_with_entropies(tok, model, PROMPT, MAX_NEW_TOKENS)

    n = len(tokens)
    above = [i for i, h in enumerate(entropies) if h > TAU]
    frac = len(above) / max(n, 1)

    print("=== Generated trace (token | entropy | DP marker) ===")
    for i, (t, h) in enumerate(zip(tokens, entropies)):
        marker = "  <-- DP" if h > TAU else ""
        # Strip newlines so each token is one line
        t_show = repr(t)
        print(f"{i:3d}  {t_show:>22}  H={h:5.3f}{marker}")

    print()
    print("=== Summary ===")
    print(f"Tokens generated: {n}")
    print(f"Decision points (H > {TAU}): {len(above)}  ({frac*100:.1f}%)")
    print(
        f"Paper expects ~1-8% across tau in [1.2, 2.2]; "
        f"{'WITHIN' if 0.005 <= frac <= 0.20 else 'OUTSIDE'} the expected band."
    )
    if above:
        dp_tokens = [tokens[i] for i in above]
        print(f"DP tokens: {dp_tokens}")
    else:
        print("No decision points found — try lowering tau.")


if __name__ == "__main__":
    sys.exit(main())
