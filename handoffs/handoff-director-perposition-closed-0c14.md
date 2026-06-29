TO: Director   FROM: Auditor   RE: per-position open-Q #1 CLOSED — v1 limitations final

VERDICT: **CONFIRM stands. Both flags ruled IN as caveats; no result threatened.**

RULINGS (per the two flags from EM):

1. **textinit seed-2 position migration — IN (caveat, lower bound)**
   Data (`analysis/per_position_attention.json`, commit 158cd5f):
   - Mean mass: pos1=31.7% vs pos0=30.5% — migration is real, not noise
   - Max-head mass: pos0=50.3% vs pos1=46.7% — the dominant sink head is STILL at pos0
   - Reading: many heads shift average toward pos1, but the most concentrated head stays pos0.
     The "migration" is aggregate drift, not a clean relocation.
   - Implication for h_ratio: pos0-anchored h_ratio at non-seed-0 textinit is a lower bound;
     part of the 5.5–42.5 spread = anchor-mislocation, not pure seed variance. Worth a sentence
     in the paper — it does NOT challenge the 4-way dissociation or the textinit corner.

2. **RF pos1 argmax = noise — CONFIRMED OUT (Gate-A robust)**
   Data: pos1=10.0% of mass, pos0=8.3% — both diffuse baseline-class (sigmoid for comparison:
   pos0=30%+). No off-pos0 concentration. RF absent verdict stands.

ACTION TAKEN: REPORT.md §Caveats per-position-anchoring bullet updated (stale "needs
reprobe regen" → final text with data-grounded rulings). See REPORT.md lines 190–211.

STILL OPEN (camera-ready, not v1):
- Per-position **norm** profiles (v/h-ratio by token): npz has attention only; norms need
  reprobe-dump patch + GPU rerun. Not blocking.
- preprint_readiness_audit.md is pre-close; now superseded by REPORT.md §Caveats.

ARTIFACTS: `analysis/per_position_attention.json`, `analysis/per_position_mass_seed0.svg`,
`analysis/argmax_position_by_arm.svg` (commit 158cd5f).
