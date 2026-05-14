# improvements/

> Runnable Python prototypes for the proposals in [`../05-improvements.tex`](../05-improvements.tex).

Each file implements one specific subsection of `05-improvements.tex` (Implementation / Code Improvements). Cross-references below.

---

## Prototypes

| File | Implements (§ in 05-improvements.tex) | One-line description |
|------|---------------------------------------|----------------------|
| [`adaptive-tau.py`](./adaptive-tau.py) | Implementation / Code Improvements §1 | Compares fixed-τ entropy gating vs quantile-based gating across two prompts; shows quantile gating is stable across inputs while fixed-τ is brittle to entropy-distribution shift |

## Run

```bash
pip install -r requirements.txt
python adaptive-tau.py
```

Loads `Qwen2.5-0.5B-Instruct` (~1GB, CPU-runnable in <60s). Generates a CoT for two prompts (math word problem + creative writing), computes per-token entropy under greedy decoding, applies both gating schemes, and prints the comparison.

## Expected output (illustrative)

```
=== fixed tau = 1.4 ===
       math: gated  6/96 ( 6.2%)   mean H=0.31  max H=2.84
   creative: gated 41/96 (42.7%)   mean H=1.18  max H=4.62

=== quantile q = 0.95 ===
       math: gated  5/96 ( 5.2%)   (stable target = 5%)
   creative: gated  5/96 ( 5.2%)   (stable target = 5%)

Fixed-tau gated-fraction spread across prompts: 36.5 percentage points
Quantile-q  gated-fraction spread across prompts:  0.0 percentage points
Quantile gating is much more stable across prompts. Claim supported.
```

## Notes

These are prototypes, not production code. They demonstrate the *direction* of the proposed improvement on a small, transparent example — not the SOTA result. A production-grade quantile-gated ReasonMaxxer would belong in a separate repo.

Per-prompt entropy distributions are model- and scale-dependent. The "spread" headline number will vary across runs and across base models; the trend (quantile is more stable than fixed) should hold.
