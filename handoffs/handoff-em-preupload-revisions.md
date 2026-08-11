TO: Engineering Manager   FROM: Director   RE: final pre-upload revision package — arXiv v1 (STE edition)

CONTEXT:
- Paper = `deliverables/vlm-sink-preprint-vlm4rwd-ste/`. Dual build already works
  (`python3 build.py --arxiv` → named `paper-v2-ste-arxiv.*`; default → anonymous workshop).
- An external agent review produced a blocker list. Director adjudicated it; rulings below are
  binding — where I say "reject" or "modify", do NOT apply the reviewer's version.
- **All prose edits go into the shared `sections/`** so the workshop build inherits them. Never
  fork the prose.
- Split the work: **Drafter** owns prose/claims; **local-agent** owns build, figures, format.
- This is the last package before upload. After it lands: rebuild BOTH PDFs, then one clean-room
  read of the arXiv build.

═══════════════ DRAFTER — prose & claims ═══════════════

A. ACCURACY FIXES (all confirmed against our audited record — apply as written)

1. **"From scratch" is inaccurate.** SigLIP encoder is pretrained+trainable; textinit uses a
   pretrained decoder. Replace throughout with "vision–language pretraining with randomly
   initialized decoders". Keep "from-scratch *decoder*" only where explicitly qualified.
   Title must drop "From-Scratch" (see F).

2. **RF is not repetition-free.** 2.39 effective visual epochs = examples DO repeat. Replace
   "repetition-confound-free" / "fresh, non-repeated" / "removes the repetition confound" with
   **"low-repetition (2.39 effective visual epochs), with no observed overfit (held-out val
   tracks train)"**. Verify these five disclosures are all present in the STE limitations
   (several already are — confirm, don't duplicate): weights-only optimizer restart at ~57M
   tokens; shuffle buffer 1500→500; RF's probe batch is still the fixed repeated-Cauldron tail;
   "~74 epochs" describes the 1B repeated pool, not the 100M comparison arms; `<3% overlap`
   must be qualified as an estimate unless we have measured dedup evidence (we do not).

3. **Metric units — the abstract overclaims.** Per `docs/conventions.md`, h-ratio is measured
   **per layer**, not per head. Correct the units everywhere (abstract, method, results):
   concentration → per query head; value-norm ratio → per KV head (90 independent values under
   GQA); residual-norm ratio → per layer (30 values). Also: h-ratio alone does NOT establish
   "massive activations" (normally defined by channel-level outliers, which we never measured).
   Call it a **massive-activation proxy** / "position-specific residual-norm asymmetry" on first
   use, then use the short form.

4. **Drop the significance stars.** Table 4 prints `***` while the text admits pseudoreplication
   (nested/duplicated observations under GQA, single seed). Remove p-value stars and all
   inferential language for v1. Report r descriptively; the sign flip stays as **descriptive**
   evidence. Do not claim "no fixed coupling law" — say no consistent-sign relationship across arms.

5. **Stale per-position text — real self-contradiction.** Limitations/conclusion say full
   per-position scans are future work; they exist (see `REPORT.md` + the norm dump). Integrate
   the finding and DELETE the future-work line: in textinit the attention max sits at pos1 while
   the residual peak / value trough sit at pos13 — the signatures dissociate in **position** as
   well as magnitude, and pos0 is not the single common anchor in that arm. Keep this as a
   supporting observation, not a second headline (pos0 anchoring holds for the from-scratch arms).

6. **G1 attribution is wrong — and our own plan pre-warned this.** Qiu et al. use ordinary
   initialization; our exact-zero init makes the gate σ(0)=0.5 at step 0, i.e. an initial
   half-scale attention intervention. Rewrite as **"Qiu-style G1 with our zero-initialized
   variant"** and state the scale confound explicitly in method + limitations.

7. **Wording precision (all correct, apply):**
   - g1gate concentration: "suppressed" → **"near-absent"** (it is *above* baseline's 0.000).
   - g1gate value-norm: "neutral" → **"milder drain"** (0.81–0.85 is still 15–19% drain).
   - textinit concentration: "total" → **"strong"**.
   - RF h-ratio: not monotonic at probe resolution → **"positive long-horizon trend"**.
     Use exact 1.43→3.22 (≈2.3×) everywhere; **retire "+130%"**. Abstract's "more than doubles"
     is already safe — keep.
   - textinit at init: do NOT say it inherits "everything at step 0" — norm signatures are
     present at init, but Sink^0.3 is 0 at step 0 and crosses later.
   - sigmoid: "no massive activation" → **"no strong pos0-specific residual-norm asymmetry"**
     (absolute residual norms are globally large; only the ratio is ≈1).
   - Replace **"independently controllable"** / "depending only on the lever" with dissociation
     language: four non-factorial arms show **distinct intervention-associated profiles**; "no
     two arms share a signature triple" is our audited phrasing — use it.

8. **Novelty vs Luo et al. [7] — narrow it.** Luo's App. A.4 does track sink-dimension magnitudes
   across alignment checkpoints (frozen ViT, pretrained LLM). Add one clause acknowledging this
   and scope our claim to **dense, joint tracking of all three signatures under randomly
   initialized decoders**. Do not claim prior multimodal work "cannot study emergence."

9. **Reporting completeness** — add to method/appendix: token accounting definition (image tokens
   + non-padding text tokens); probe size n=32 and probe version; validation-set size; seeds per
   arm; exact checkpoint token counts; aggregation formulas. Report per-arm validation losses and
   **state explicitly that MMStar was not run**. Separate equal-token comparisons from
   unequal-competence ones (textinit vs random-decoder arms).

B. DIRECTOR OVERRIDES — do NOT follow the reviewer here

- **Grounding hook: KEEP** the single hedged sentence + the closing "we do not test this" line.
  It is cited to the one verified direct source and the venue organizer confirmed our setup in
  writing. Grounding stays OUT of the title. Reject the reviewer's full removal.
- **Repetition story: KEEP in the main text.** It is not a fault to bury — it is the motivation
  for RF, which the reviewer themself called the strongest evidence in the paper. Reframe in one
  sentence: "the four-arm comparison reuses a small image pool; we therefore re-test the central
  negative result under low repetition." Do not exile it to the figures section.
- **Fesser et al. (2606.08105): DO NOT CITE YET.** Not in any Researcher-verified list. Route to
  Researcher/browser to confirm it exists and actually uses value norms to distinguish sink
  algorithms. Cite only after verification. (Agent-suggested references have been wrong before.)

C. ADDITIONS THE USER ASKED FOR

10. **Notation block in §2** (approved; must not grow total length — compress elsewhere if
    needed). Max four displayed equations: Sink^ε_1, v-ratio, h-ratio — each WITH its index set,
    which simultaneously fixes item A3 — plus one line on the softmax sum-to-one constraint as
    the standing mechanism for sink formation (cite Gu [1]).

11. **"Interpretation (hypotheses)" paragraph** in results or discussion, explicitly flagged as
    speculative, offering *why* the signatures dissociate: the output gate can afford
    concentration because it suppresses what the sink injects into the residual stream; sigmoid
    removes the softmax sum-to-one coupling that otherwise forces value compensation; textinit
    imports structure rather than forming it. Cite Gu's key-bias account. Never phrase as claims.

12. **Abstract citation fix — NOT YET APPLIED, verify and apply.** Abstract still reads
    "deployed VLMs [21, 22]". Browser verification: [22] (SAGE) directly supports sink→
    hallucination in VLMs; [21] (Kang) is attention-misallocation, not hallucination; [24] is
    text-only. Change to **[22] only**. Leave [21]/[24] wherever they appear in related work.

13. **Reference [23]** — add verified publication data: X. Zhang, Y. Zhu, C. Gu, J. Cao, H. Cheng,
    K. Wu. *Information Processing & Management*, 63(2A), art. 104431, 2026.
    DOI 10.1016/j.ipm.2025.104431. (Author order already VERIFIED — add vol/issue/art/DOI only.)

═══════════════ LOCAL-AGENT — build, figures, format ═══════════════

14. **arXiv build hygiene.** Ship from the `--arxiv` lineage. Strip "Preprint draft v2 ·
    Simplified Technical English edition" and all workshop/version language from the named build.
    Confirm the named PDF carries author metadata + working repro links, and the anonymous build
    leaks neither.

15. **Regenerate Figure 3 with the correct sigmoid head (L7H3).** The excuse is gone —
    `reprobe.py` has the mps/cpu auto-device patch, so a one-checkpoint, one-head T×T dump runs
    on the Mac in minutes with **no GPU box**. Omitting the arm's true sink head from the
    principal qualitative figure is not acceptable. Also add **raw, unnormalized** sigmoid
    diagnostics to the main paper (row-normalization changes the measured object; Gu's result
    concerns unnormalized sigmoid attention).

16. **Figure corrections:** make explicit where Fig 1 uses max-head attention while tables report
    fraction-of-heads-above-threshold; disclose the 5/9-point smoothing and retain faint raw
    points; label every figure with seed + exact checkpoint/token count; mark birth-map values as
    interval-censored by checkpoint resolution; increase small labels.

17. **PDF/format:** add page numbers, clickable links, author PDF metadata, bookmarks if cheap.
    Fix the page-17 three-line orphan / appendix break.

18. **Format mimic (lean).** Read `/Users/samvattiwari/Documents/ResPprs/2511.17036v1.pdf` (vision
    or code) and port only the lean layout skeleton — margins, font stack, heading and caption
    style — into `build.py` CSS. Do not restructure content or chase a pixel match.

═══════════════ CLOSE-OUT ═══════════════

19. Rebuild BOTH PDFs. Then a **clean-room read of the arXiv build** by someone who has not been
    editing it — this review caught issues four of us missed; the final read must be of the final
    artifact, not the sections.
20. Confirm the repo and HF checkpoints dataset are **public** (dead links on day one are worse
    than no links). Report, don't guess.

TITLE (Director recommendation, user to confirm): **"Four Levers, Four Corners: Attention-Sink
Signatures Dissociate in Vision–Language Pretraining"** — keeps the hook, drops the inaccurate
"From-Scratch", grounding stays out.

GUARDRAILS UNCHANGED: novelty scoped to the conjunction, never "first to decouple"; textinit as
range/median; no downstream-capability claims; numbers must match `deliverables/session4_n3_audit.md`
and `runs/rf_fresh_baseline/GATE_A_REPORT.md`, not raw `REPORT.md`.

RETURN (compact): what landed vs deferred; any place a fix conflicts with an audited number
(surface, do not silently resolve); Fesser verification result; repo/dataset visibility; both PDF
paths. Nobody submits anything — upload is the user's action.
