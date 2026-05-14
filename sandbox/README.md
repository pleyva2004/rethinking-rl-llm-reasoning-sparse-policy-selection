# ai-study-rethinking-rl-llm-reasoning

> Sandbox study of **Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning** (Akgül et al., 2026, [arxiv:2605.06241](https://arxiv.org/abs/2605.06241)).

This repo isolates one claim from the paper and probes it with a minimal, self-contained experiment.

---

## The claim being probed

> *Base-model token-level entropy alone identifies the positions where RL would intervene.*
> Specifically: high-entropy positions cluster at "decision points" — moments in a reasoning trace where the model is genuinely uncertain which branch to take — and those are exactly the positions whose tokens RL reranks.

If this is true on a small base model, then a tiny entropy histogram over a single greedy decode should show a clear bimodal-ish structure: most positions near zero entropy (confident next token), a sparse minority above threshold $\tau \approx 1.4$ (decision points), and the high-entropy positions should land at *semantically meaningful* moments — choosing an operator, branching on a case, naming a variable.

## Experiment design

1. Load a small open base model (`Qwen/Qwen2.5-0.5B-Instruct` by default; falls back to `distilgpt2` for fully CPU-only environments).
2. Greedy-decode a short chain-of-thought for a math word problem.
3. At every generated position $t$, compute
   $$H_t = -\sum_v \pi(v \mid q, o_{<t}) \log \pi(v \mid q, o_{<t})$$
   from the model's actual logits.
4. Print the token-by-token sequence with each token annotated by its entropy. Mark positions above $\tau$ as decision points.
5. Report:
   - Total tokens generated
   - Fraction above threshold (paper observes 1–8% across $\tau \in [1.2, 2.2]$)
   - The specific tokens at decision points (qualitative — do they look like real branching moments?)

## Expected output

- Decision-point fraction in the 2–10% range. If this comes out >50% the claim has failed for this model size; if it comes out <0.5% the threshold is mis-calibrated for the model.
- Decision-point tokens visually concentrated at structural moments (operators, conjunctions, choices), not at obvious tokens like punctuation or function words.

## What would falsify the claim

- High-entropy positions land at **un-meaningful** tokens (random punctuation, BPE-fragments mid-word, formatting whitespace). That would imply entropy is tracking surface noise, not reasoning branching, and the paper's "entropy = decision point" identity needs to be base-model-conditional.
- Decision-point fraction is essentially uniform across the trace (no spikiness). That would imply there are no true "branches" in the trace and the paper's mental model doesn't apply.

## Run

```bash
pip install -r requirements.txt
python experiment.py
```

Runs on CPU in ~30–90s depending on model. Default model is small enough to fit in <2GB RAM.

To run with a larger model: edit `MODEL_NAME` at the top of `experiment.py`.

## Files

- `experiment.py` — main script (loads model, generates, computes entropy, annotates tokens)
- `requirements.txt` — pinned deps
- `.gitignore` — standard Python ignores

## Notes

This is sandbox-grade code: optimized for clarity over performance, single-purpose, not production-ready. It probes one prediction of the paper on a much smaller model than the paper studied. Negative results here would be informative — they'd suggest the entropy = decision point identity is a property of the larger pre-trained models the paper uses, not a universal property of autoregressive LMs.

Full study notes (math deep dive + case study) live in private research notes; this repo intentionally focuses on the runnable probe.
