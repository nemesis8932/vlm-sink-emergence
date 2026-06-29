TO: Director   FROM: Auditor   RE: publication figure suite (6/6 built) + layout/captions for sign-off + 2 audit flags

All six figures built, zero-GPU, on Mac, committed `95f3db5` (`analysis/fig1–6.{svg,png}`,
`fig_common.py`; SVG = vector for the paper). Palette = `analyze_sinks.py` (RF added, purple).
Data: `probes.jsonl` (figs 1–4) + HF npz `per_key_attention`/`attn_matrix` (figs 5–6, fetched).

═══ PROPOSED MAIN vs APPENDIX SPLIT ═══
MAIN BODY (3):
  • Fig 2 — phase-portrait  ........ HERO / abstract teaser
  • Fig 6 — sink-stripe matrices ... the "pristine evidence" panel
  • Fig 4 — birth-map + lead-lag .... the ordering/causal-timing story
APPENDIX / SUPPORT (3):
  • Fig 3 — per-head scatter ........ head-level proof + correlation table
  • Fig 5 — entropy collapse ........ ties to text-LM literature
  • Fig 1 — layer×head grid ......... where-in-network + animation source

═══ CAPTIONS (one line each) + DOES-IT-SUPPORT-THE-THESIS ═══
Fig 2 (HERO): "Decoupling phase-portrait — each lever drives the three sink signatures into a
   distinct corner of (concentration × value-norm × massive-activation) space over training."
   → YES, strongest single visual: 4 levers, 4 corners, motion = independence.
Fig 6: "Query×key attention of the top sink head: the pos0 stripe is absent in baseline, total
   in textinit, and already present at init (inherited from the text LM)."
   → YES for baseline-absent / textinit-total / inherited@init. ⚠ see Flag 2 (sigmoid panel).
Fig 4: "Lead–lag — in softmax-scratch arms the norm signatures cross threshold early and
   concentration never arrives; sigmoid is the mirror (concentration only); textinit inherits
   all three at init." → YES, this is the ordering/dissociation evidence.
Fig 3: "Per-(layer,head) concentration vs value-norm: the correlation SIGN flips by arm
   (+0.67 baseline … −0.76 textinit; pooled −0.20) — no universal head-level coupling."
   → YES, and stronger than expected (see Flag 1).
Fig 5: "Attention-entropy collapse separates by arm — only sigmoid & textinit collapse;
   baseline/g1gate/RF stay flat — collapse tracks concentration, not the norm signatures."
   → YES, cleanly ties our decoupling to the text-LM coupling literature.
Fig 1: "Per-(layer,head) attn→pos0 over training (row-normalized): baseline/RF stay cold,
   textinit hot at init, sigmoid lights a band." → YES, supporting/orienting.

═══ CORRELATION COEFFICIENT (Fig 3 ask) ═══
Pearson r(attn→pos0 , value-norm ratio), per-head, final ckpt, n=270/arm:
  baseline +0.67***  g1gate +0.53***  sigmoid −0.03 ns  textinit −0.76***  RF +0.43***
  POOLED (n=1350) −0.20***
Reading: NOT a null cloud — the sign is arm-dependent. If concentration and value-drain were
one mechanism the sign would be consistent; it isn't → independent control. I reframed the
figure around this (honest + stronger than the "uncorrelated" hypothesis in your brief).

═══ AUDIT FLAGS (2) ═══
FLAG 1 — Fig 3 reframed, your call. Brief said "show the cloud is UNcorrelated." Data says
  per-arm correlations are strong but sign-FLIPPING (+0.67 → −0.76); pooled is weak only
  because arms cancel. I titled it "no universal coupling: sign is arm-dependent." This is
  more defensible than "uncorrelated" (a reviewer would compute per-arm r and catch us). If
  you prefer the original framing tell me; I'd advise against it.

FLAG 2 — Fig 6 sigmoid panel is under-powered, NOT fixable without GPU. The cloud dumped the
  T×T matrix for sigmoid's top-2 heads selected by *raw* (unnormalized) gate score → both are
  ~0.12 normalized pos0, NOT sigmoid's true sink head (L7H3 = 0.87 normalized, in the reprobe
  npz but no full matrix dumped). So the sigmoid stripe looks weak. I (a) display the stronger
  of the two dumped heads, (b) annotated the panel + softened the suptitle to claim only
  baseline-absent / textinit-total / inherited. Sigmoid's true concentration is fully carried
  by Figs 1/3/5. Cheapest fix if you want the sigmoid stripe in Fig 6: a ~1-ckpt reprobe
  re-dump of L7H3's matrix (single inference walk) on the next box — recoverable, low priority.

═══ ALSO DONE — textinit caveat corrected in REPORT.md §Caveats (your instruction #38–43) ═══
I VERIFIED the norm dump before rewriting and your proposed wording was partly inaccurate —
corrected to the cleaner true finding:
  • CONFIRMED: re-anchoring s2 under the attention sink (pos1) gives 9.46 < pos0's 14.7, so the
    old "lower-bound/anchor-mislocation" hedge is WITHDRAWN; spread 44.8/8.2/14.7 = genuine
    seed variance. ✓ (matches your numbers)
  • CORRECTED: your "‖h‖ peak stays pinned at pos0" is not what the data shows — the ‖h‖ peak
    is at pos1 (s1) and pos13 (s2), i.e. the massive-activation does not sit on pos0 OR on the
    attention sink. The accurate, stronger statement: the 3 signatures decouple in POSITION as
    well as magnitude (attn pos1, ‖h‖ pos13, drain pos13 — all different tokens). I wrote that
    version. Flagging so the abstract/body don't repeat "pinned at pos0."

RETURN TO ME: sign-off on the main/appendix split + Flag-1 framing; go/no-go on the optional
sigmoid L7H3 re-dump for Fig 6.
