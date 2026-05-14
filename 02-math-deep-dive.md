# Math Deep Dive — Rethinking RL for LLM Reasoning

**Arxiv:** https://arxiv.org/abs/2605.06241
**Studied:** 2026-05-13

> Mathematician-grade walk-through. Definitions, derivations, load-bearing assumptions. No paraphrase.

---

## Setup & Notation

Let $\pi_\theta(\cdot \mid q, o_{<t})$ denote an autoregressive language model parameterized by $\theta$, conditioned on a prompt $q$ and partial output $o_{<t} = (o_1, \dots, o_{t-1})$. The vocabulary is $\mathcal{V}$.

Two models are central:
- $\pi_{\text{base}}$ — a pre-trained base model (e.g. Qwen2.5-7B).
- $\pi_{\text{RL}}$ — the same base after RL post-training (GRPO, PPO, etc.) with a verifiable scalar reward $r(o, q) \in \{0,1\}$ indicating answer correctness.

A **rollout** for problem $q$ is a sampled trajectory $o = (o_1, \dots, o_T) \sim \pi(\cdot \mid q)$. Greedy decoding from $\pi_{\text{base}}$ and $\pi_{\text{RL}}$ on the same $q$ gives token sequences that may diverge.

## Central Object — Token-Level Entropy and Decision Points

The paper's primary diagnostic is the **per-position predictive entropy** under the base model:

$$
H_t \;=\; -\sum_{v \in \mathcal{V}} \pi_{\text{base}}(v \mid q, o_{<t}) \, \log \pi_{\text{base}}(v \mid q, o_{<t}). \tag{1}
$$

A position $t$ is called a **decision point** at threshold $\tau$ iff $H_t > \tau$. The paper's empirically optimal $\tau \in [1.2, 1.8]$ (in nats), corresponding to roughly the top 2–8% of positions.

### Disagreement statistics (Table 2)

For a base/RL pair, define on each position $t$:
- *Reranked* iff $\arg\max_v \pi_{\text{RL}}(v \mid \cdot) \neq \arg\max_v \pi_{\text{base}}(v \mid \cdot)$ but the RL-preferred token is in the base top-5.
- *Shifted* iff the RL-preferred token is **outside** the base top-5.

The paper measures across 4 base/RL pairs:

| Quantity | Range observed |
|---|---|
| Reranked fraction | $1.03\%$ – $3.96\%$ |
| Shifted fraction | $0.01\%$ – $0.12\%$ (essentially zero) |
| Mean rank (under base) of RL's chosen token | $2.14$ – $2.39$ |
| Entropy ratio: $H_t$ at reranked / $H_t$ at unchanged | $5\times$ – $12\times$ |

**Implication.** $\pi_{\text{RL}}$'s greedy trajectory differs from $\pi_{\text{base}}$'s only at a sparse set of high-entropy positions, and even there, the chosen token is "the second-most-likely thing the base would have said." Formally: the support of disagreement is a measure-zero (in token positions) deviation, and the deviation lives inside the base top-5.

### Oracle intervention (causal claim)

Define the **oracle-corrected** trajectory $o^{\text{oracle}}$ as: greedy-decode from $\pi_{\text{base}}$, but at each reranked position $t \in R$ (set $R$ identified by comparison to $\pi_{\text{RL}}$), replace the base's top-1 with $\arg\max_v \pi_{\text{RL}}(v \mid \cdot)$.

Define a **random-control** trajectory $o^{\text{rand}}$: same procedure, but $|R|$ random positions are picked instead, each replaced with a random base top-5 token.

Empirical claim (their Figure 2): pass@1 of $o^{\text{oracle}}$ matches pass@1 of $\pi_{\text{RL}}$ exactly across all four pairs; pass@1 of $o^{\text{rand}}$ is ≤ base. The control rules out "perturbation alone helps."

### Entropy-only locator

Replace the rule "intervene at $R$" with "intervene at $\{t : H_t > \tau\}$" — i.e., locate decision points without consulting $\pi_{\text{RL}}$. With $\tau = 1.2$, the entropy-gated correction:
- Matches the oracle on Qwen2.5-7B → GRPO.
- Closely approaches it on Qwen2.5-7B → PPO.
- Touches only $1.2\%$–$8.3\%$ of tokens.

This is the load-bearing observation that licenses an **RL-free** training procedure.

---

## ReasonMaxxer — Entropy-Gated Contrastive Fine-Tuning

### Per-rollout normalised advantage

For problem $q$, sample $N$ rollouts $\{o^{(i)}\}_{i=1}^N$ from $\pi_{\text{base}}$ and label each by correctness $r_i \in \{0,1\}$. Compute:

$$
A_i \;=\; \frac{r_i - \bar{r}}{\sigma_r + \epsilon}, \tag{5}
$$

where $\bar{r}$ and $\sigma_r$ are the per-problem mean and standard deviation of $\{r_i\}$, and $\epsilon$ is a numerical floor.

Note: this is the same normalisation used in GRPO (Shao et al., 2024) — the paper inherits the per-group normalisation but **discards the rest of GRPO** (no online rollouts, no KL-to-reference penalty in the standard form, no clipped ratio).

### Decision-point set per rollout

For each rollout $o^{(i)}$, compute base-model entropy at every position and define
$$
D^{(i)} \;=\; \{ t \in \{1, \dots, T_i\} : H_t > \tau \}.
$$

### Loss

The training loss has two terms (paper Section 5.3):

1. **Advantage-weighted CE at decision points only:**
$$
\mathcal{L}_{\text{contrast}}(\theta) = -\sum_i A_i \sum_{t \in D^{(i)}} \log \pi_\theta(o_t^{(i)} \mid q, o_{<t}^{(i)}).
$$
Correct rollouts ($A_i > 0$) push probability mass *toward* the tokens used at decision points. Incorrect rollouts ($A_i < 0$) push *away*. Crucially, the gradient is supported only on $D^{(i)}$ — the bulk of tokens (low entropy, "obvious next token") get no gradient.

2. **Base-anchoring KL** at non-decision positions to prevent drift on tokens RL would never touch:
$$
\mathcal{L}_{\text{anchor}}(\theta) = \sum_i \sum_{t \notin D^{(i)}} \mathrm{KL}\big( \pi_{\text{base}}(\cdot \mid q, o_{<t}^{(i)}) \;\big\|\; \pi_\theta(\cdot \mid q, o_{<t}^{(i)}) \big).
$$

Total: $\mathcal{L} = \mathcal{L}_{\text{contrast}} + \lambda \, \mathcal{L}_{\text{anchor}}$.

### Why this works (mechanistic argument)

Three ingredients combine:

1. **Sparse support** — gradient only flows where it matters; eliminates noise from bulk tokens whose distribution is already correct.
2. **Within-top-5 promotion** — the loss never has to invent a new token; it just has to reweight among existing high-probability candidates, which is a much easier optimisation surface than open-vocabulary RL.
3. **Anchoring KL** — preserves the base distribution outside $D^{(i)}$, so the model retains general fluency and doesn't collapse onto reasoning-only behaviours.

---

## Low-Dimensional Correction (Section 4)

A separate experimental thread shows the entire RL→base delta is captured by a **rank-32 LoRA on QKVO attention projections** trained via KL distillation from $\pi_{\text{RL}}$:

$$
\min_\phi \, \mathbb{E}_{q, o \sim \pi_{\text{base}}} \Big[ \, \mathrm{KL}\big( \pi_{\text{RL}}(\cdot \mid q, o_{<t}) \,\big\|\, \pi_{\text{base}+\Delta_\phi}(\cdot \mid q, o_{<t}) \big) \Big]
$$

with $\|\Delta_\phi\|$ constrained to 0.27%–0.49% of total parameters. Adapter matches RL teacher's accuracy on MATH-500 and GSM8K. Even rank-8 on $W_O$ alone (output projection) approaches the rank-32 QKVO result.

**Inference.** RL's behavioural delta lives in a subspace whose dimension is a tiny fraction of the model's rank, and which is *concentrated in the output projection*. Combined with the sparse-decision-point story, this paints a coherent picture: RL is a **low-rank reranking operator**, not a capability transformer.

---

## Alternative Formulations

This work can be re-derived as:

- **Sparsified DPO at the position level.** DPO (Rafailov et al., 2023) does sequence-level preference learning with reference KL anchoring. ReasonMaxxer is "the same idea but the unit of preference is a *token* at an entropy-gated position, not a full sequence."
- **Best-of-N distillation, restricted.** STaR / rejection sampling FT trains on full correct sequences with uniform token loss. ReasonMaxxer trains on the same sequences but with non-uniform per-position weighting — zero weight outside decision points, advantage-weighted inside.
- **Active-learning supervised fine-tuning.** Treat $H_t$ as the acquisition function. The whole method is "active SFT where the active criterion is the model's own next-token entropy."

---

## Load-Bearing Assumptions

| Assumption | Used in | Failure mode if violated |
|---|---|---|
| RL's chosen tokens lie in base top-5 | Eqs. and Tab. 2; entire framing | If RL reaches outside top-5 (e.g. tool-use or novel proof tactics), entropy-gated CE cannot match RL because the target token isn't a candidate |
| Base entropy correlates with "where RL intervenes" | §3.3, ReasonMaxxer locator | Tasks where the right intervention point is *low-entropy* (e.g. confidently wrong) defeat the locator |
| Verifiable scalar reward $r \in \{0,1\}$ | Advantage computation, Eq. 5 | Open-ended generation has no clean $r$; advantage normalisation breaks |
| Decision-point density is moderate (1–8%) | Sparsity of gradient, efficiency claim | Tasks with diffuse uncertainty (e.g. dialogue) would push every position into $D^{(i)}$, collapsing to standard SFT |
| Anchoring KL preserves general capability | $\mathcal{L}_{\text{anchor}}$ term | Insufficient $\lambda$ → catastrophic forgetting outside reasoning; too high $\lambda$ → no learning signal |

---

## Gaps Flagged

- **The "rank ~2" finding has no theoretical explanation.** The paper observes mean rank 2.14–2.39 across all base/RL pairs. This is suspiciously stable. I cannot derive from the paper why RL's promoted token specifically lives at rank 2 rather than rank 5 or rank 1.5. There may be a fixed-point / Lipschitz argument hiding in the optimisation dynamics of GRPO/PPO with bounded learning rate.
- **The base-anchoring KL is described in the paper text but I could not extract a clean expression for $\lambda$ tuning.** The ablations (App. A) study LoRA rank but not $\lambda$ as a free knob; possible the paper folds it into the contrastive scale.
- **No formal proof that entropy-gated correction is *necessary* for the recovery — only that it's sufficient.** A position outside $D^{(i)}$ might also influence pass@1 if the trajectory branches; the paper's controls handle this empirically but not analytically.

---

## Connections

- Direct lineage: GRPO (Shao et al., 2024) for advantage normalisation; DPO (Rafailov et al., 2023) for offline preference; STaR (Zelikman et al., 2022) for self-training on correct rollouts.
- Mechanistic-RL adjacent: Yue et al. (2025) on pass@k showing RL traces are within base sampling distribution; Davis & Recht (2025) on RL-with-binary-rewards reducing to SGA on the probability of correctness — these prior results constrain *what* RL could possibly be doing; this paper localizes *where* in the trajectory.
- Active learning ancestry: per-position entropy as acquisition criterion echoes Lewis & Gale (1994) uncertainty sampling, lifted to token-level.
