TO: Auditor   FROM: Director   RE: preprint-readiness pass — close the cheap soft spots so we can post arxiv v1

CONTEXT (rest in repo — don't restate):
- Gate B CLOSED at n=3. Four-way dissociation confirmed: `deliverables/session4_n3_audit.md` (your own adjudication). Gate A CONFIRM: `runs/rf_fresh_baseline/GATE_A_REPORT.md`.
- Decision: post an arxiv **v1 preprint** to stake priority (scoop re-checked clear by user 2026-06-28). Stage-2 scaled runs are NOT in v1 (future work). All tasks below are **zero-GPU**, off-cloud, from existing data.
- Soft spots catalogued in `docs/open-questions.md` (owners tagged).

ASK — produce a preprint-readiness deliverable. Four items, all must-close for v1:

1. **Per-position robustness (open-Q #1) — the priority.** Our headline `Sink^ε`, v_ratio, h_ratio are all **pos0-anchored**. From existing `reprobe/` per-head/per-position detail (RF + Session-3, on Mac/HF), produce: (a) argmax-attention-position histogram across ALL positions, (b) per-position v-norm and h-norm profiles. Goal: **prove pos0 is THE special token**, not an assumption — a sink/MA parked on a different image token would otherwise hide in "rest." This is the defense a reviewer will demand. Figure + one-paragraph finding.

2. **Wording lock (open-Q #3).** Pin the exact phrasing of the "symptoms assumed inseparable" claim. We have evidence symptoms CO-OCCUR in text LLMs (Gu et al. 2410.10781; 2510.06477 ~step-1k co-emergence) and DON'T in our VLM. Cite those papers precisely; do NOT overstate as a formal claim they made (strawman risk). Deliver the locked sentence(s) + citations.

3. **Limitations section (draft prose).** State explicitly, up front: pretrained-ViT = partial from-scratch + inherited-norm confound (open-Q #2; our defense = h_ratio starts ~1.0–1.4 and RISES → forming not inherited; cite register-token / propagated-sink work 2510.08510, Darcet et al.); 1B vs Gu's 5B token scale (open-Q #6); textinit magnitude seed-sensitivity (h_ratio 5.5–42.5, seed-0 outlier); seed-0 raw not first-hand re-derivable (trusted from checksummed `archive/session3/`); domain-shift cauldron→FineVision (ADR-0001).

4. **textinit → range/median everywhere.** Sweep the writeup-facing tables/figures: textinit signatures reported as range/median (h_ratio ~12× median, range 5.5–42.5), never a point estimate. Same for its concentration/value magnitudes.

CONSTRAINTS: zero compute, Mac/off-cloud, existing data only. Publication-grade figures.

RETURN (compacted): the per-position figure + finding (pos0 confirmed special, y/n); the locked claim sentence + citations; the limitations prose; confirmation the textinit range/median framing is applied. Flag anything in #1 that does NOT support pos0-as-special — that would change the headline, so surface it loudly.
