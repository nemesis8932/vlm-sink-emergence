TO: Cloud-agent   FROM: Director   RE: figure dumps (entropy + sink-stripe) the last run MISSED — grab ALL, then exit (stop only)

WHY THIS EXISTS: prior box ran only the per-position NORM task; the amendment (per-key attention +
T×T matrices) arrived too late and was NOT executed. This is the complete, primary task — do not
trim. Everything here is inference re-walk of checkpoints ALREADY ON HF (no training, no litdata,
no leak risk). reprobe.py auto-selects device + already has the committed norm-profile patch.

PRE-FLIGHT GUARD (do FIRST, before the full sweep):
- Round-trip ONE checkpoint end-to-end: dump → push to HF → reload the npz and assert it's
  readable + shapes correct. Only after that round-trips do the full sweep. (Don't pay for a
  sweep that produces unreadable npz — the failure mode that costs a 3rd trip.)

DUMPS — all from the fixed probe batch (random.seed(0), 32 samples; same as the probe contract so
metrics are comparable):

1. PER-KEY ATTENTION MARGINAL — the (H, T) mean-attn-per-key vector (sink_probe.py:96, discarded
   after argmax today) for ALL (layer, head). Patch reprobe to save it.
   COVERAGE: all 4 arms (baseline, g1gate, sigmoid, textinit) seed-0 + RF, at EVERY available HF
   checkpoint (the dense {0,250,1k,2k,4k,8k,16k,32k,64k,...}). Dense coverage = a real entropy-
   over-training curve, not the sparse one we worried about (fig 5).
   → analysis/per_key_attention.npz

2. FULL T×T ATTENTION MATRIX — attn[:, h, :, :] batch-mean, for the top-2 sink heads per arm
   (by attn→pos0). COVERAGE: final + one early ckpt for each of the 4 arms; for textinit ALSO
   ~step-0 (the INHERITED sink at init). This is the iconic query×key sink-stripe (fig 6).
   → analysis/attn_matrix.npz

3. COVERAGE MANIFEST — small json on GIT (not HF) listing exactly which arms/seeds/ckpt-steps/heads
   were dumped, so the Auditor knows the figure coverage without opening the npz.
   → analysis/dump_manifest.json

OPTIONAL (only if 1–3 are clean and cheap — the box is up anyway): pos0-MASK ABLATION — at each
arm's final ckpt, val-loss WITH vs WITHOUT attention to pos0 → "is the sink load-bearing?" A strong
camera-ready experiment. Skip if it adds real time; it's recoverable later.

OUTPUT NAMING must match `handoffs/handoff-auditor-figure-suite.md` (per_key_attention.npz,
attn_matrix.npz). MVP: dump RAW arrays, no plotting on GPU — Auditor renders on Mac.

CONSTRAINTS: billed; this is a one-shot top-up — minimize, self-stop on waste. All ckpts are on HF
→ anything missed is recoverable, so do NOT overrun chasing completeness.

THEN EXIT (docs/exit-runbook.md): box-only sweep → confirm npz on HF + manifest/patch on git +
`git status` clean → vastai stop → confirm actual_status=stopped. STOP THERE. Agents NEVER destroy;
hand back to user — the user destroys the box.

RETURN (compacted): per_key + attn_matrix on HF (paths) + manifest committed; one-line confirm the
pre-flight round-trip passed; entropy/sink-stripe coverage (arms × steps); stop confirmed. Optional
ablation result if run.
