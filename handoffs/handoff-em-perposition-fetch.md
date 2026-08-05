TO: Engineering Manager   FROM: Auditor   RE: per-position open-Q #1 — UPDATE

STATUS 2026-06-29: fetch + seed-0 location DONE (user ran on Mac). pos0 confirmed max-mass in
all 4 arms @seed-0 (textinit 0.63/0.99) → headline anchoring validated. TASKS 1-2 below = DONE.
REMAINING for EM (GPU box, not v1-blocking): regen reprobe npz for the runs HF lacks —
**seed-1, seed-2, RF** — then run `analysis/per_position_attention_from_npz.py` (point its RUNS
map at those dirs) to confirm whether textinit position migrates off pos0 at seed-2 (live-probe
argmax says it does). Optional TASK 3 (per-position NORM profiles) still needs the reprobe.py
patch. Original brief below for reference.
---

WHY: Director approved "pull npz from HF, close properly" for the preprint per-position defense.
The Auditor session has NO network/pip (denied even with sandbox off), so the fetch must run on
your networked box. Full context: `deliverables/preprint_readiness_audit.md` (item #1).

TASK 1 — fetch (zero-GPU, ~download only):
  huggingface-cli download nemesismaniac/vlm-sink-emergence-ckpts \
    --repo-type dataset --include "runs/*/reprobe/*.npz" --local-dir .
  Then commit the npz? NO — `*.npz` is gitignored. Instead either (a) run TASK 2 on the box and
  commit the resulting analysis json/svg, or (b) push the npz to a non-ignored path if Director
  wants them in git. Recommend (a).

TASK 2 — close the attention-location half (zero-GPU, on the box):
  python analysis/per_position_attention_from_npz.py
  This prints, per arm/seed, whether pos0 IS the max attention-mass position. Commit
  `analysis/per_position_attention.json` (+ figure) back. That is the headline defense:
  it proves/refutes "the concentration sink is at pos0" using mass, not just argmax.
  EXPECTED (from live-probe argmax, to be confirmed by mass): sigmoid pos0 ✓; baseline/g1gate
  diffuse; RF argmax pos1 (check if real mass-sink or noise); textinit position MIGRATES by
  seed (s1 pos0/pos5, s2 pos1/pos13) — this is the flag.

TASK 3 — OPTIONAL, only if Director wants the per-position NORM profiles too (item 1b full):
  The npz does NOT contain per-position v/h norms (reprobe stores pos0-vs-rest only). To get
  them, patch `reprobe.py` to also dump full per-position arrays:
    - add `vn_allpos = v_norm.mean(dim=0).cpu().numpy()`  # (H,T) per-head v-norm per position
    - add `hn_allpos = res_norm.mean(dim=0).cpu().numpy()` # (T,) residual norm per position
    - savez both; then re-run reprobe.py on the checkpoints (NEEDS GPU + ckpts from HF).
  Recommend deferring TASK 3 to camera-ready; TASK 2 is enough for v1.

RETURN: `analysis/per_position_attention.json` committed, + one line: "pos0 is max-mass for {arms}";
flag any arm where it is NOT (esp. RF, textinit). Auditor will fold into the v1 limitations.
