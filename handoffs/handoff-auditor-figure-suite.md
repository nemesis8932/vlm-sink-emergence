TO: Auditor   FROM: Director   RE: publication figure suite for arxiv v1 — phase-portrait is the hero

CONTEXT (rest in repo):
- Result locked (n=3 dissociation, `deliverables/session4_n3_audit.md`). This is the visual-evidence
  pass to make v1 compelling. All zero-GPU, off-cloud, Mac. Style: publication-grade, consistent
  palette across the suite (reuse the arm colors in `analyze_sinks.py`).
- Data: per-(layer,head,step) in `runs/*/probes.jsonl` (attn_to_pos0, argmax_key, per-head
  v-norms, h-norms). Both NEW dumps NOW LANDED (commit a4bf7b4/c2c807a; npz on HF; coverage in
  `analysis/dump_manifest.json`): (a) `per_key_attention.npz` — 38 arrays, dense 0→64k, all 5 runs
  → entropy (fig 5); (b) `attn_matrix.npz` — 9 T×T matrices, 4 arms early+final + textinit step-0
  → sink-stripe (fig 6). **All six figures are GO** — nothing deferred. NB textinit step-0
  raw_to_pos0=0.242 = inherited-sink-at-init data point, use it in fig 6.

ASK — build the suite. **HERO = the decoupling phase-portrait (main body, abstract-teaser
candidate); the rest are results/appendix support.** Return a proposed main-vs-appendix split +
one-line caption each for Director sign-off.

1. **Layer×head cross-section grid** (the original "sink formation" view) — 30 layers × 9 heads,
   color = attn→pos0, per arm, at a few key steps (init / early / final); plus an **emergence
   animation** (grid over training steps). Shows WHERE in the network the sink lives and how it
   forms. From probes.jsonl.
2. **Decoupling phase-portrait — HERO.** 2D trajectory, x=concentration (mean/max attn→pos0 or
   Sink^ε), y=value-norm ratio (and/or h_ratio); each arm a PATH that diverges into its own corner
   over training. The thesis as motion, not a table. From probes.jsonl summaries.
3. **Per-head decoupling scatter.** Every (layer,head) a dot: x=attn→pos0, y=v_ratio. Show the
   cloud is UNcorrelated → head-level proof concentration ≠ value-drain. Report a correlation coeff.
4. **Sink birth-map + lead-lag.** Heatmap of step-of-first-ε-crossing per (layer,head); plus
   lead-lag bars (norm signatures cross early, concentration late/never) per arm. From probes.jsonl.
5. **Entropy-collapse comparison** (waits on dump a). Attention entropy over checkpoints per arm;
   frame vs text-LM coupling (Gu 2410.10781; 2510.06477 entropy-collapse+sink co-emerge ~step 1k)
   — our VLM decouples them. Note checkpoint-sparse; state that.
6. **Attention-matrix sink-stripe** (waits on dump b). Render the iconic query×key heatmaps:
   textinit (strong) vs baseline (none) vs sigmoid (strong, value-amplified) vs g1gate, final +
   early (+ textinit step-0 inherited). The instantly-legible "pristine evidence" panel.

CONSTRAINTS: zero compute, existing data only. textinit magnitudes as range/median (seed-0 outlier).

ALSO — CORRECT the textinit caveat in `REPORT.md §Caveats` (the norm dump refuted the old hedge):
the h_ratio spread (44.8/8.2/14.7) is **genuine seed variance**, NOT a pos0-anchor lower bound —
re-anchoring at the true max-mass position gives LOWER values (s2 14.7→9.46), and the ‖h‖ peak
stays pinned at pos0 even when attention drifts to pos1/pos13, i.e. **massive-activation does NOT
migrate with attention**. Replace the "lower bound / anchor-mislocation" wording with this cleaner,
stronger finding.

RETURN: the figure files + proposed layout (main/appendix) + captions for sign-off; correlation
coeff for #3; one-line "does it support the thesis?" per figure.
