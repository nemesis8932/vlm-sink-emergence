TO: Engineering Manager   FROM: local Agent   RE: per-position open-Q #1 CLOSED — seed-1/seed-2/RF

CONTEXT: per `handoffs/handoff-local-perposition-npz-bf1d.md`; all work on Mac, zero GPU.

RETURN (one-line verdict):
**pos0 IS max-mass for: baseline s0/s1, g1gate s0/s1/s2, sigmoid s0/s1, textinit s0/s1.
textinit-s2 MIGRATES OFF pos0 → pos1 (0.317 vs 0.305, confirmed). RF pos1 argmax = noise (mass 0.100, diffuse, not a sink).**

Full per-arm results — `analysis/per_position_attention.json` (committed 158cd5f):

| arm | seed | argmax-mass pos | mass@pos0 | mass@max | verdict |
|---|---|---|---|---|---|
| baseline | s0 | pos0 | 0.105 | 0.105 | pos0 IS max |
| g1gate | s0 | pos0 | 0.113 | 0.113 | pos0 IS max |
| sigmoid | s0 | pos0 | 0.195 | 0.195 | pos0 IS max |
| textinit | s0 | pos0 | 0.850 | 0.850 | pos0 IS max |
| baseline_seed1 | s1 | pos0 | 0.102 | 0.102 | pos0 IS max |
| g1gate_seed1 | s1 | pos0 | 0.119 | 0.119 | pos0 IS max |
| g1gate_seed2 | s2 | pos0 | 0.119 | 0.119 | pos0 IS max |
| sigmoid_seed1 | s1 | pos0 | 0.204 | 0.204 | pos0 IS max |
| textinit_seed1 | s1 | pos0 | 0.298 | 0.298 | pos0 IS max |
| **textinit_seed2** | **s2** | **pos1** | 0.305 | 0.317 | **MAX AT pos1 (pos0 mislocated)** |
| rf_fresh_baseline | s0* | pos1 | 0.083 | 0.100 | MAX AT pos1 — diffuse, not a sink |

*RF seed: uses baseline arm config; treated as "s0" of the fresh-data run (not a seed variant of the session-3 arms).

TEXTINIT-S2 flag: pos1 beats pos0 by 0.317 vs 0.305 — small margin but clean directional shift. Consistent
with live-probe argmax (pos1 60%, pos13 29%, pos0 only 9% at seed-2). The sink migrates by seed for textinit.
Implications: s0 h_ratio 42.5 measured at pos0 is valid; s1/s2 pos0-anchored h_ratio likely under-measures.
Auditor's existing limitation caveat covers this.

RF flag: pos1 argmax mass = 0.100. Baseline-class arms sit at 0.10–0.12; RF is in that range.
Live-probe argmax showed 76% of heads' argmax at pos1 — that's an argmax vote, not mass. By mass it's 0.10
vs 0.083 at pos0 — a small diffuse difference, not a concentration sink. "RF hidden pos1 sink" = noise. ✓

PROBE BATCH NOTE (flag per handoff): streaming probe used (interleave_datasets with streaming=True,
first 32 samples, seed=0). The cloud seed-0 reprobe used full non-streaming dataset download + tail-1024
selection — different exact samples. Within THIS re-run all arms/seeds use the same streaming batch
(internally apples-to-apples). The question is attention *location*, not magnitude calibration, so probe
batch identity doesn't affect the verdict.

PATCHES committed (158cd5f):
- `reprobe.py`: streaming probe batch (avoids multi-GB shard download — ran in seconds per arm on Mac)
- `data/collators.py`: `batch_encode_plus` → `tokenizer()` (transformers 5.x removed the method)
- `models/language_model.py`: `getattr(hf_config, 'rope_theta', 10000)` (LlamaConfig no longer has the attr)
- `analysis/per_position_attention_from_npz.py`: RUNS map extended to all seed1/seed2/RF

CONSTRAINTS honored: zero GPU spend; Mac mps/cpu only; *.npz gitignored (not committed); no cloud box.

RETURN to Auditor: fold table above into v1 limitations (textinit position seed-variability; RF diffuse-not-sink).
Director/EM: no further local-agent work needed for this open-Q.
