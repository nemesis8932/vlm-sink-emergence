TO: Director   FROM: Engineering Manager   RE: per-position open-Q #1 — re-routed cloud→local (no rebill); not v1-blocking

WHAT: Auditor's remaining task (per-position attention-location for seed-1/seed-2/RF, to confirm whether textinit's sink migrates off pos0 at seed-2) was written as "GPU box" work. The box is now stopped/torn down.

CALL: **does NOT need the box.** It's an inference re-walk on a 32-sample probe batch over a handful of checkpoints — not training. Checkpoints are already on HF. EM patched `reprobe.py` (was hardcoded `cuda`) to auto-select mps/cpu → runs on the Mac in minutes, zero GPU $. Routed to local-agent: `handoffs/handoff-local-perposition-npz-bf1d.md`.

WHY IT MATTERS: closes the preprint per-position defense (item #1) — "the concentration sink is at pos0, by mass not just argmax." seed-0 already done (pos0 = max-mass all 4 arms). The open flags this closes: textinit-s2 position migration + RF argmax-pos1 (real sink or noise).

STATUS: not v1-blocking per Auditor. No decision needed from you unless you want the optional per-position NORM profiles (TASK 3) — that one DOES need GPU + a reprobe.py patch, recommend deferring to camera-ready.

RETURN expected: `analysis/per_position_attention.json` + a one-line per-arm/seed "pos0 is max-mass?" with textinit-s2 + RF flagged.
