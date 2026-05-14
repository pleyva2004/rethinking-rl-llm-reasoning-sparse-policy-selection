# Interview Prep — Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning

**Authors:** Akgül, Kannan, Neiswanger, Prasanna (USC + DEVCOM ARL)
**Arxiv:** https://arxiv.org/abs/2605.06241 (May 2026, v2)
**Studied:** 2026-05-13

---

## One-sentence elevator
RL post-training for reasoning doesn't teach the model anything new — it nudges 1–3% of tokens at high-entropy decision points toward alternatives the base model already had in its top-5, and you can match it without RL by gating a contrastive loss on base-model entropy.

## What's novel
- A **causal mechanistic claim**: it's not "RL helps reasoning"; it's "RL reranks ~2% of tokens, the promoted token is always already in the base top-5 (mean rank ~2.2), and those tokens sit at exactly the positions where base entropy is highest."
- An **oracle ablation that recovers RL's full pass@1** by overwriting only the disagreement positions — random substitutions at the same density don't help (often hurt). This makes the claim falsifiable, not just observational.
- A **teacher-free locator**: base entropy alone (threshold $\tau \approx 1.4$) finds the right positions without ever needing the RL model. This collapses the entire "where to intervene" question.
- **ReasonMaxxer**: applies advantage-weighted contrastive loss only at entropy-gated positions, using a few hundred base rollouts and *no online generation*. Matches GRPO/PPO across 3 model families × 6 scales × 6 math benchmarks at ~1000× lower cost.

## What's mathematically clever
- The setup reframes RL post-training as a **selection problem** on a fixed support: the base model's top-5 at each position. So the policy class is finite and tiny per token, and the only learning signal needed is "which of these 5 is right *here*."
- Entropy $H_t = -\sum_v \pi_\theta(v \mid q, o_{<t}) \log \pi_\theta(v \mid q, o_{<t})$ becomes a **diagnostic for where supervision is informative** — directly connects to the active-learning view that you want gradients where the model is uncertain.
- The contrastive loss reduces to advantage-weighted CE only at gated positions, which is essentially a sparsified DPO with *positional* (not just sequence-level) preferences. Sequence-level preference RL is doing way more work than necessary.

## What I'd push back on
- "Within base top-5" is a load-bearing claim — but it's measured on already RL-trained checkpoints. If RL produces base distributions whose top-5 has been pre-shaped by RL itself, the claim is partially circular. Need pass@k on a held-out *truly base* model.
- Entropy threshold $\tau$ is tuned per setup (1.0–2.2). If you have to grid-search $\tau$ on each new model, "no RL needed" is partially traded for "now you grid-search a different scalar."
- All experiments are math reasoning with verifiable rewards. The whole story may evaporate on open-ended tasks (writing, agentic tool use, multi-turn) where "decision point" isn't well-defined and there's no clean reward.
- Cost comparison ($1000× cheaper) compares to full RL training cost. But ReasonMaxxer still needs *base-model rollouts on labeled correctness* — it's not free, just cheap. The accounting could be sharper.

## Open questions
- Is the "low-dimensional, rank-32 LoRA captures all of RL" finding a property of the math domain, or a structural property of transformers under any RLVR? If the latter, that's a much bigger claim than the paper states.
- Why does the promoted token live at rank ~2 specifically? Is there a theoretical reason RL's correction operator has bounded "step size" in rank-of-promoted-token, or is it an artifact of small RL learning rates?
- Does this collapse if reasoning requires *exploring* outside the base model's distribution (e.g., out-of-distribution proof techniques, novel tool sequences)? The framework predicts: yes, it should collapse — and that's the cleanest test.
