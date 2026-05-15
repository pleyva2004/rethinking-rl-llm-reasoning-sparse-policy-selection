# Tour — Akgül et al. 2026: Sparse Policy Selection for LLM Reasoning

> A guided learning path through the paper, the foundations it leans on, and the constructive extensions worth building.
>
> arXiv: [2605.06241](https://arxiv.org/abs/2605.06241)

---

## Section 1: Reader's Contract

**Who this is for.** Mathematicians, ML researchers in adjacent subfields (optimisation, probability, statistical learning), and engineers comfortable reading per-token training equations. The paper is empirical-mechanistic with a thin but sharp probabilistic spine; you will get more out of it if you are comfortable manipulating expectations over discrete distributions and reading KL terms without flinching.

**Background assumed.** Linear algebra at the level of eigenvalues / linear maps; multivariate calculus through the Jacobian; probability through random variables, expectation, and variance; Shannon-information at the level of entropy and KL. No measure theory required for the body; the proof of the proposed Lipschitz bound (improvement #1) leans on operator norms and a one-line perturbation argument that an analyst will read in two minutes.

For each background piece we link the corresponding `math-foundations` concept page. If you want a guided ramp instead of self-assembly, four pre-built foundation tours exist:

| Audience | Tour |
|---|---|
| Novice | [math-foundations/tours/novice.md](https://github.com/pleyva2004/math-foundations/blob/main/tours/novice.md) |
| CS undergrad | [math-foundations/tours/cs-undergrad.md](https://github.com/pleyva2004/math-foundations/blob/main/tours/cs-undergrad.md) |
| **Math grad (recommended for this paper)** | [math-foundations/tours/math-grad.md](https://github.com/pleyva2004/math-foundations/blob/main/tours/math-grad.md) |
| Researcher | [math-foundations/tours/researcher.md](https://github.com/pleyva2004/math-foundations/blob/main/tours/researcher.md) |

**Recommended entry tour: math-grad.** The paper's machinery sits exactly at "knows what KL is, can read a Jacobian, doesn't need measure theory restated." The novice tour will feel slow; the researcher tour skips foundations you actually use here (Shannon entropy in particular).

**Expected reading time.** 3–4 hours for a focused first pass that includes the foundations skim, the paper proper, and the deep-dive walkthrough. Add 2 hours if you also work the proof for improvement #1.

**One-sentence elevator.** Akgül et al. show that RL post-training on verifiable rewards (RLVR) for math reasoning is *not* learning new capabilities — it is performing a sparse, entropy-gated re-selection at roughly the top 2–8% of token positions, where the RL policy promotes a token that the base model already had at mean rank ~2.2, and the entire effect can be reproduced by an offline contrastive fine-tune (`ReasonMaxxer`) with no rollouts.

---

## Section 2: Foundations Walk

These are read in topological order: probability before information theory before linear algebra applied to gradients. Levels come from `~/.claude/math-foundations/manifest.json`. "Why this paper needs it" annotations are sourced from `02-math-deep-dive.md`.

| # | Concept | Level | Pacing | Why this paper needs it |
|---|---|---|---|---|
| 06 | [Linear Maps and Matrices](https://github.com/pleyva2004/math-foundations/blob/main/concepts/06-linear-maps/README.md) | intermediate | skim | Attention projections $W_Q, W_K, W_V, W_O$ are linear maps; the LoRA delta $\Delta W = BA$ is a low-rank linear-map decomposition. Required to read the LoRA-on-QKVO ablation. |
| 07 | [Eigenvalues and Eigenvectors](https://github.com/pleyva2004/math-foundations/blob/main/concepts/07-eigenvalues/README.md) | intermediate | skim | Used implicitly when reasoning about the spectral structure of the rank-32 LoRA delta and the proposed Lipschitz bound on rank displacement (operator-norm of the GRPO update). |
| 14 | [Gradient and Jacobian](https://github.com/pleyva2004/math-foundations/blob/main/concepts/14-gradient-jacobian/README.md) | advanced | read | The contrastive loss $\mathcal{L}_{\text{contrast}}$ and base-anchoring KL $\mathcal{L}_{\text{anchor}}$ are differentiated through the softmax Jacobian; the Lipschitz bound argument is a Jacobian-norm argument on the policy update map. |
| 22 | [Random Variables](https://github.com/pleyva2004/math-foundations/blob/main/concepts/22-random-variables/README.md) | advanced | skim | Each rollout $o = (o_1, \ldots, o_T) \sim \pi(\cdot \mid q)$ is a discrete-time random sequence; the per-position entropy $H_t$ is a deterministic function of a categorical RV. |
| 23 | [Probability Distributions](https://github.com/pleyva2004/math-foundations/blob/main/concepts/23-distributions/README.md) | advanced | read | $\pi_\theta(\cdot \mid q, o_{<t})$ is a categorical distribution over $\mathcal{V}$ at every position. Everything in the paper — entropy, KL, contrastive loss — is an operation on these per-position distributions. |
| 25 | [Expectation](https://github.com/pleyva2004/math-foundations/blob/main/concepts/25-expectation/README.md) | advanced | read | Shannon entropy is $H = \mathbb{E}[-\log p]$; the GRPO advantage normalisation is a per-rollout sample-mean / sample-std centering of the reward. |
| 26 | [Variance and Covariance](https://github.com/pleyva2004/math-foundations/blob/main/concepts/26-variance-covariance/README.md) | advanced | skim | The per-rollout normalised advantage $\hat A_i = (r_i - \mu)/\sigma$ inherits its variance-stabilising property from this; understanding why this normalisation reduces gradient variance is essential to following the GRPO inheritance discussion. |
| 36 | [Shannon Entropy](https://github.com/pleyva2004/math-foundations/blob/main/concepts/36-shannon-entropy/README.md) | advanced | **drill into proof** | The paper's *central object* is $H_t = -\sum_v \pi_{\text{base}}(v \mid q, o_{<t}) \log \pi_{\text{base}}(v \mid q, o_{<t})$, eq. (1) of the deep-dive. Every result in the paper is conditioned on this scalar exceeding a threshold $\tau$. If you don't internalise entropy, nothing in the paper localises. |
| 37 | [Cross-Entropy](https://github.com/pleyva2004/math-foundations/blob/main/concepts/37-cross-entropy/README.md) | advanced | read | The contrastive loss is structurally a cross-entropy between the empirical advantage-weighted target distribution and $\pi_\theta$. |
| 38 | [KL Divergence](https://github.com/pleyva2004/math-foundations/blob/main/concepts/38-kl-divergence/README.md) | advanced | **drill into proof** | The base-anchoring loss $\mathcal{L}_{\text{anchor}} = \sum_i \sum_{t \notin D^{(i)}} \mathrm{KL}\!\left(\pi_{\text{base}} \,\|\, \pi_\theta\right)$ is what prevents drift on the 92–98% of tokens that ReasonMaxxer is *not* supposed to touch. The asymmetry of KL — base in the first slot, $\theta$ in the second — is load-bearing and worth thinking about. |

Total: 10 nodes. Pacing budget for a math-grad: ~45 minutes total if you already have the prerequisites; the only two that earn "drill into proof" treatment are entropy (because it's the paper's organising principle) and KL (because the asymmetry choice is non-obvious).

---

## Section 3: Paper Concepts Walk

These are paper-specific objects pulled directly from `02-math-deep-dive.md`. They presume you have done the foundations walk above.

| # | Concept | What it does in the paper | Where it appears |
|---|---|---|---|
| P1 | **Token-level entropy $H_t$** | Per-position predictive entropy under $\pi_{\text{base}}$. The single scalar that decides whether a position is a "decision point". | eq. (1), §"Central Object — Token-Level Entropy and Decision Points" |
| P2 | **Decision-point set $D^{(i)} = \{t : H_t > \tau\}$** | The sparse subset of positions on which RL actually changes anything; empirically 2–8% of tokens. | §"Central Object", and the gating rule used in $\mathcal{L}_{\text{contrast}}$ |
| P3 | **GRPO-style advantage normalisation $\hat A_i = (r_i - \mu)/\sigma$** | Per-rollout (per-question group) variance-stabilising centering of the binary reward. Inherited from Shao et al. 2024. | §"Per-rollout normalised advantage" |
| P4 | **Reranked vs shifted classification** | A position is *reranked* if RL's argmax differs from base's but lies in base's top-5; *shifted* if outside top-5. The headline result is that shifted ≈ 0.01–0.12% — RL never goes off-distribution. | Table 2 / §"Disagreement statistics" |
| P5 | **Contrastive loss $\mathcal{L}_{\text{contrast}}$ on decision points** | Advantage-weighted log-likelihood, restricted to $t \in D^{(i)}$. The "do-something" half of ReasonMaxxer. | §"ReasonMaxxer — Entropy-Gated Contrastive Fine-Tuning" |
| P6 | **Base-anchoring KL $\mathcal{L}_{\text{anchor}}$** | $\sum_i \sum_{t \notin D^{(i)}} \mathrm{KL}(\pi_{\text{base}} \,\|\, \pi_\theta)$. The "do-no-harm" half — keeps $\pi_\theta$ pinned to $\pi_{\text{base}}$ outside decision points. | §"Base-anchoring KL" |
| P7 | **KL-clipped policy ratio (inherited PPO/GRPO machinery, mostly *discarded*)** | The paper inherits group-normalisation from GRPO but explicitly **discards** the clipped ratio and KL-to-reference penalty. Worth understanding what is *not* used to appreciate how minimal ReasonMaxxer is. | §"Note: this is the same normalisation used in GRPO..." |
| P8 | **Rank-32 LoRA on QKVO** | The full RL→base behavioural delta is reproduced by a rank-32 LoRA on attention QKVO projections distilled from $\pi_{\text{RL}}$ via KL — i.e., the policy change has very low intrinsic dimension. | §"A separate experimental thread shows..." |
| P9 | **Weight tying with input embedding (LoRA-on-output ablation)** | Most of the LoRA contribution concentrates in $W_O$; combined with the standard LM-head weight-tying to input embeddings, this makes the change interpretable as a small direction in embedding-space. | §"LoRA QKVO" |

---

## Section 4: Improvements Walk

Parsed from `05-improvements.tex`. Four `\section*` blocks: Mathematical, Code, Experimental, Theoretical. Each proposal gets a one-paragraph statement and a validation mode. Math/Theoretical proposals are validated by **PROOF** (LaTeX in `proofs/`); Code/Experimental by **MEASUREMENT** (Python in `improvements/` with a `measure()` function whose output is checked).

### Mathematical

**M1 — Lipschitz bound on rank displacement.** Across all four base/RL pairs Akgül report, the RL-promoted token sits at base-rank 2.14 to 2.39 — a suspiciously narrow band. Conjecture: under bounded GRPO learning rate $\eta$ and per-step KL-budget $\beta$, the rank displacement at decision points is Lipschitz in $(\eta, \beta)$. A perturbation argument on the softmax Jacobian, combined with the known operator-norm bound on the GRPO update, should yield a quantitative bound that recovers the empirical rank-2 regularity from first principles.
- **Validation: PROOF** — `proofs/lipschitz-rank-displacement.tex`

**M2 — Information-theoretic tightening of "top-5 suffices".** The paper observes that 99.88–99.99% of RL choices stay inside the base top-5, but offers no bound. An information-theoretic argument, possibly via Fano's inequality on the reward channel, should produce a $k$-dependent bound of the form *"with high probability under bounded KL budget, the RL choice lies in base top-$k$ for $k \lesssim f(H_t, \beta)$."*
- **Validation: PROOF** — `proofs/info-theoretic-top-k.tex` *(deferred — not building this proof in v1.9)*

### Code (Implementation)

**C1 — Hyperparameter-free quantile-based gating $\tau$.** ReasonMaxxer's $\tau$ is grid-searched per model/scale, partially undoing the savings of eliminating the RL loop. Replace with $D^{(i)} = \{t : H_t^{(i)} > Q_q(\{H_t\}_{t=1}^{T_i})\}$ where $q$ is a single scale-invariant percentile (e.g. $q = 0.95$). The fixed absolute $\tau = 1.4$ collapses (gates 6.2% of one prompt and 42.7% of another); the quantile gate gates exactly $1-q$ of every prompt, by construction.
- **Validation: MEASUREMENT** — `improvements/adaptive-tau.py` (extended with `measure()`). Expected output: a dictionary of the form `{"fixed_tau_gated_pct": [6.2, 42.7], "quantile_gated_pct": [5.0, 5.0], "stability_ratio": ~7x}` showing that the quantile gate is rollout-invariant where the absolute threshold is not.

**C2 — Single-pass entropy scoring.** The naive implementation computes $H_t$ from full softmax distributions stored per-position, which is memory-heavy for long rollouts. A streaming implementation that fuses the entropy computation into the forward pass (logit reuse + online running sum) is $\sim 2\times$ faster and constant-memory.
- **Validation: MEASUREMENT** — `improvements/streaming-entropy.py` (NEW). Expected output: `{"naive_ms": ..., "streaming_ms": ..., "speedup": ~2.0, "max_abs_diff": <1e-6}` confirming numerical equivalence to the naive baseline.

### Experimental

**E1 — Out-of-distribution stress probe.** The paper trains on math problems at the model's competence frontier where base success is non-trivial. Probing on far-OOD tasks (where base success is near zero) would test whether ReasonMaxxer's "no new capabilities" claim survives outside the regime studied. *(Deferred to follow-on work.)*
- **Validation: MEASUREMENT** *(deferred)*

**E2 — Non-verifiable task probe.** The paper's reward $r \in \{0,1\}$ assumes a verifier. On non-verifiable tasks (open-ended writing, soft constraints) the entropy-gating story may not transfer. *(Deferred to follow-on work.)*
- **Validation: MEASUREMENT** *(deferred)*

### Theoretical

**T1 — Active learning as the right framing.** The entropy threshold $\tau$ is exactly the acquisition function of uncertainty-sampling active learning (Lewis & Gale 1994), lifted to the token level. Reframing ReasonMaxxer as "active SFT with $H_t$ as the acquisition criterion and rollout correctness as the label" imports label-complexity bounds, query-efficient regret bounds, and known failure modes (acquisition-function sampling bias) from the AL literature. *(Deferred — proof of equivalence not in v1.9.)*
- **Validation: PROOF** — `proofs/active-learning-equivalence.tex` *(deferred)*

**T2 — Low-rank LoRA $\Delta W$ as inference-time activation steering.** The rank-32 LoRA on QKVO — concentrated in $W_O$ — is structurally identical to representation-engineering / activation-steering interventions. A formal equivalence (showing the LoRA delta acts at decision points as a steering vector added to the residual stream) would unify two literatures and suggest RL is redundant with steering at decision points.
- **Validation: PROOF** — `proofs/lora-equiv-steering.tex`

---

## Section 5: What to Do Next

Three concrete action items, in order of personal-leverage.

**(a) Most personally promising proposal: M1 (Lipschitz bound on rank displacement).** This is the cleanest mathematician-flavoured contribution available. The empirical regularity (rank ~2.2 across four independently trained pairs) is striking, the conjecture is testable, and a one-page perturbation argument on the GRPO update map plus the softmax Jacobian likely suffices. It is also the proposal most likely to *survive* peer review as a stand-alone short note, because it explains an existing observation rather than proposing a new method.

**(b) Single experiment to settle the most ambiguous claim: necessity of entropy-gated correction.** The paper shows entropy-gated correction is *sufficient* to recover the RL→base delta (the oracle intervention experiment), but not *necessary* — a non-decision-point intervention might also work via trajectory branching. The clean experiment: replicate ReasonMaxxer with the gating set $D^{(i)}$ replaced by (i) a *random* same-size subset, (ii) the *complement* $\{t : H_t < \tau\}$. If pass@1 collapses on (i) and (ii), the necessity claim is established. This is one training run per ablation on a 1.5B model — a weekend's compute.

**(c) Where to take this as a follow-on paper: bridge to mechanistic interpretability via the LoRA-as-steering equivalence (T2).** If the LoRA delta really is a steering vector at decision points, then (1) one should be able to *extract* the steering vector directly from $\pi_{\text{RL}} - \pi_{\text{base}}$ without the LoRA scaffolding, (2) the steering vector should compose linearly across reasoning skills (arithmetic + algebra ≈ both), and (3) the same vector should be extractable via supervised contrast on a small held-out set, fully eliminating RL. The follow-on paper writes itself: "RL for Reasoning is Activation Steering in Disguise — and Here's the Vector."

---

*Tour artifact for Akgül et al. 2026 — generated 2026-05-15. See companion `tour.tex` and `tour.ipynb` for LaTeX and notebook versions.*
