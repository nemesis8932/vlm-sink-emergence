TO: Auditor   FROM: Engineering Manager   RE: per-position open-Q #1 CLOSED — fold into v1 limitations

CLOSED on Mac, zero GPU (local-agent return `handoffs/handoff-em-perposition-return-4c9a.md`; data `analysis/per_position_attention.json`, commit 158cd5f).

VERDICT: pos0 IS max-attention-mass for baseline s0/s1, g1gate s0/s1/s2, sigmoid s0/s1, textinit s0/s1 → pos0-anchoring of the headline VALIDATED by mass (not just argmax). TWO flags for v1 limitations:

1. **textinit position is seed-variable.** s2 sink migrates pos0→**pos1** (mass 0.317 vs 0.305; corroborated by live-probe argmax pos1 60% / pos0 9%). Two independent signals agree → real, not noise. Implication: s0 h_ratio 42.5 measured @pos0 is valid; **s1/s2 pos0-anchored h_ratio UNDER-measures** (sink mass sits off pos0). This compounds the textinit h_ratio non-convergence (42.5/5.5/12.2) already in the n=3 brief — part of the magnitude spread is anchor-mislocation, not pure seed variance. Worth a sentence.
2. **RF "hidden pos1 sink" = noise, ruled OUT.** argmax-VOTE put 76% of heads at pos1, but by MASS pos1=0.100 vs pos0=0.083 — diffuse, baseline-class (0.10–0.12), no concentration. RF Gate-A "sink absent" stands; no off-pos0 sink hiding.

CAVEAT (honest): this re-run's probe batch = streaming first-32 (≠ cloud seed-0's non-stream tail-1024). Same batch across all arms/seeds here → internally apples-to-apples; location verdict unaffected by sample identity.

RETURN: none needed from EM — your call on the limitations wording. TASK 3 (per-position NORM profiles) still deferred to camera-ready (needs GPU + reprobe dump patch).
