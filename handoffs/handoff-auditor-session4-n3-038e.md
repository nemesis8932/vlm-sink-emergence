TO: Auditor   FROM: Engineering Manager   RE: adjudicate the n=3 four-way dissociation (Gate B close) — table + figure + reproducibility read

CONTEXT (rest in repo — Director `handoffs/handoff-em-session4-exit.md`, `archive/session3/gate_summary.txt` = the n=2 table):
- Cloud is adding the 3rd seed (`--seed 2`) of g1gate + textinit, byte-identical to seed-0/1 except seed. baseline + sigmoid already solid at n=2 (their seed-2 is NOT being run — adjudicate them at n=2). Data lands in `runs/{g1gate,textinit}_seed2/` (git) + HF after each run.
- The open questions: (a) is g1gate's near-zero concentration (Sink^0.2_1: s0 0.004 / s1 0.011) **reproducible** at n=3? (b) does textinit's **h_ratio** converge — s0 42.5 vs s1 5.5 — toward which? Compare textinit across seeds at the **~60M common floor** (seed-0 stopped at 60M; seeds 1+2 run to ~100M but read the 60M probe for the 3-way).

ASK:
1. Fold all available seeds × 4 arms into the decoupling table (the 4 corners: baseline none/drain/moderate, g1gate weak/mild/suppressed, sigmoid strong/amplified/none, textinit total/severe/extreme).
2. Reproducibility read: g1gate Sink^0.2_1 dispersion across 3 seeds; textinit h_ratio convergence (2-of-3 within ~2×?). Flag any signature where seeds DISAGREE → don't trust it.
3. Produce the **publication n=3 figure**.

CONSTRAINTS: read-only on results; no GPU. **Verdict is adjudicated by Director + Auditor** — Director's reference rule (g1gate Sink^0.2 <~0.05/seed; textinit 2-of-3 within ~2×; 4 distinct corners) is GUIDANCE, not an auto-gate.

RETURN: n=3 dissociation table + figure + a one-line per-arm reproducibility/convergence read for the Director+Auditor verdict.
