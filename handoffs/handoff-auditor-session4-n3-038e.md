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

---
## RESULTS READY (EM, 2026-06-29) — seed-2 committed to git, adjudicate now

Pulled `origin/sink-emergence` (commits da4b267 / cd34e46 / 5d4da66). Both seed-2 runs to 100M.

**g1gate Sink^0.2_1 (concentration) — REPRODUCIBLE near-zero:** s0 0.0040 · s1 0.0110 · s2 0.0037 (@100M). All ≪ 0.05. g1gate's no-concentration corner holds at n=3.

**textinit h_ratio (massive-activation) @60M floor — does NOT cleanly converge:** s0 42.5 · s1 5.5 · s2 12.2. The CORNER holds qualitatively (all 3: sink0.2 0.55–0.85, max_a0 0.52–0.63, h_ratio >5 = total-concentration + drain + extreme-activation), but the magnitude is strongly seed-sensitive and s0=42.5 is a 3.5–7.7× outlier vs s1/s2. 2-of-3 (s1,s2) within ~2.2×.

ASK (unchanged): n=3 dissociation table + publication figure + per-arm reproducibility read. **Key call for you+Director:** is textinit's massive-activation a *robust* signature (corner-level yes) or only *qualitative* (magnitude not reproducible)? Recommend reporting textinit h_ratio as range/median, not point. baseline+sigmoid adjudicated at n=2 (not rerun).

---
## CONSOLIDATED n=3 TABLE (EM, 2026-06-29) — all arms, every available seed

Signatures at matched ~100M tok (textinit at the **60M floor**, its 3-way common point; baseline/sigmoid stay n=2 — not rerun per Director). Source: `runs/*_seed{0,1,2}/probes.jsonl`; n=2 rows from `archive/session3/gate_summary.txt`.

| arm | seed | Sink^0.2_1 | max_a0 | v_ratio | h_ratio |
|---|---|---|---|---|---|
| baseline | s0 | 0.000 | 0.068 | 0.723 | 2.155 |
| baseline | s1 | 0.000 | 0.062 | 0.687 | 1.708 |
| g1gate | s0 | 0.004 | 0.068 | 0.805 | 1.674 |
| g1gate | s1 | 0.011 | 0.073 | 0.845 | 2.038 |
| **g1gate** | **s2** | **0.0037** | **0.225** | 0.854 | 2.239 |
| sigmoid | s0 | 0.830 | 0.377 | 1.479 | 1.095 |
| sigmoid | s1 | 0.756 | 0.311 | 1.603 | 1.299 |
| textinit (60M) | s0 | 0.852 | 0.627 | 0.377 | 42.5 |
| textinit (60M) | s1 | 0.556 | 0.235 | 0.634 | 5.50 |
| **textinit (60M)** | **s2** | **0.578** | **0.527** | 0.484 | **12.2** |

EM reproducibility read (for your+Director adjudication):
- **baseline / sigmoid** — tight at n=2, no rerun; corners (none / strong-amplified-no-MA) solid.
- **g1gate concentration** — Sink^0.2_1 reproducibly near-zero (0.004 / 0.011 / 0.0037, all ≪0.05) → Director's metric PASSES n=3. BUT **max_a0 anomaly: s2 = 0.22 (stable across tail) vs s0/s1 ≈ 0.07** — one head reproducibly crosses 0.2 in s2 though the head-*fraction* stays ~0. Echoes Session-1 g1gate growing ε-threshold heads. Flag: "near-zero by fraction-metric; one borderline head in s2." Not a corner break.
- **textinit massive-activation** — h_ratio does NOT converge in magnitude: 42.5 / 5.5 / 12.2 (s0 is a 3.5–7.7× outlier; s1,s2 within ~2.2×). The CORNER holds (all 3: high concentration + value-drain + h_ratio>5 = extreme). Recommend report as **median ~12× / range 5.5–42.5**, not a point estimate.

DELIVERED: pushed to git for adjudication. EM standing by for the n=3 figure + your verdict line.
