# Preprint spine — vlm-sink-preprint (arXiv v1)

Target: ~6–8 pg empirical preprint, workshop/TMLR-grade. User writes all prose + visuals;
Drafter maintains this spine, supplies sourced bullet-scaffolds on request, reviews against it.

**WRITE ORDER (not document order):** 04-Results → 03-Method → 05-Related-Work → 02-Intro →
06-Limitations → 07-Conclusion → 01-Abstract (last). Appendix anytime.

---

## Central claim (one sentence)

> In from-scratch vision-language pretraining, the three signatures commonly treated as facets
> of a single "attention-sink" phenomenon — attention concentration, value-norm drain, and
> massive activations — dissociate: four training levers drive them into four distinct corners
> of signature space, and massive activations grow over 1B fresh tokens while concentration
> stays exactly zero.

## Contribution bullets (3)

1. **Four-way dissociation (n=3 seeds).** Four training arms (baseline, gated-attention,
   sigmoid-attention, text-LM-init) land in four distinct corners of (concentration ×
   value-norm × massive-activation) space; no two arms share a signature triple, and
   value-norm alone takes three directions (drain / neutral / amplify).
   [SOURCE: `deliverables/session4_n3_audit.md` — CONFIRMED verdict]
2. **Confound-free decoupling at 1B tokens.** On a fresh-data (non-repeated) stream, massive
   activation rises +130% (h_ratio 1.43→3.22) while the concentration sink stays 0.000 for
   the entire 0→1B run. [SOURCE: `runs/rf_fresh_baseline/GATE_A_REPORT.md` — Gate A CONFIRMED]
3. **No universal head-level coupling.** Per-head correlation between concentration and
   value-norm flips sign by arm (+0.67 baseline → −0.76 textinit; pooled −0.20) — inconsistent
   with a single shared mechanism. [SOURCE: figure-suite handoff `0e9a`, Fig 3 + r table]

**NOVELTY WORDING (guardrail — verbatim frame):** first to show all three sink signatures
*independently controllable* during *from-scratch multimodal* pretraining, adding value-norm
drain as a third axis. NEVER "first to decouple" (2603.05498 + 2603.17771 decouple two axes in
text — must cite-and-distinguish). "Causally unified" reserved strictly for 2510.06477. NEVER
"one inseparable phenomenon" (strawman); use Auditor's locked sentence
(`deliverables/preprint_readiness_audit.md` §Item-2) verbatim.

---

## Sections (document order) — `sections/NN-name.md`, one file each

### 01-abstract.md — ~150 wds — WRITE LAST
Purpose: central claim + 3 contributions + Fig 2 teaser numbers. No "pinned at pos0"
(Auditor: ‖h‖ peak sits pos1/pos13 by seed, NOT pos0 — flag 0e9a). No benchmark claims.

### 02-intro.md — ~600 wds
Purpose: why sinks matter → text-LM co-emergence background (use locked Item-2 sentence) →
gap: never dissected in from-scratch multimodal → contributions verbatim from above.
Figure: Fig 2 (phase-portrait, HERO) referenced early.

### 03-method.md — ~800 wds
Purpose: model (222M nanoVLM = SigLIP-B/16 + SmolLM2-135M-arch decoder, from-scratch decoder),
data (the_cauldron; FineVision fresh stream for RF), 4 arms as *levers*, 3 metric definitions
(Sink^ε_1, v_ratio, h_ratio — all pos0-anchored, state this openly), probe cadence.
[SOURCES: `docs/conventions.md`, `docs/experiments.md`, `docs/adr/`]
Must state: pos0-anchoring is a measurement choice, defended in Limitations.

### 04-results.md — ~1400 wds — WRITE FIRST
Purpose: the evidence, data-first. Sub-blocks:
- **4.1 Four corners (Fig 2 hero + n=3 table).** Corrected table from
  `session4_n3_audit.md` ONLY (not REPORT.md). textinit magnitudes as RANGE/MEDIAN:
  h_ratio median ~12×, range 5.5–42.5, seed-0 outlier on every signature. baseline/sigmoid n=2.
- **4.2 Fresh-data 1B decoupling (RF).** Sink 0.000 entire run; h_ratio 1.43→3.22 continues
  post-warmup; v_ratio = supporting context only, NOT framed as emergence (Auditor ruling).
  [SOURCE: GATE_A_REPORT.md]
- **4.3 Head-level: no universal coupling (Fig 3).** Sign flips by arm; NOT "uncorrelated."
- **4.4 Timing/ordering (Fig 4 lead-lag) + stripes (Fig 6).** Fig 6 claims ONLY
  baseline-absent / textinit-total / inherited-at-init — sigmoid panel under-powered (wrong
  head dumped), do not lean on it.
- Supporting observation (not headline): textinit signatures decouple in POSITION too
  (attn→pos1, ‖h‖→pos13, drain→pos13; seed-dependent).
Figures main body: Fig 2, Fig 6, Fig 4 (Auditor split).

### 05-related-work.md — ~500 wds
Purpose: near drop-in from Researcher report. Must cite-and-distinguish 2603.05498 +
2603.17771 (two-axis text decoupling); "causally unified" → 2510.06477 only; Gu et al.
2410.10781 (co-occurrence); 2510.08510 (ViT-propagated vs LLM-emerged); Darcet registers.
[SOURCE: Researcher report — ⚠ NOT in repo yet, see BLOCKERS]

### 06-limitations.md — ~400 wds
Purpose: honest, up-front. Base = Auditor draft prose (`preprint_readiness_audit.md` §Item-3),
6 items: pos0-anchoring (seed-0 per-position mass CLOSED — pos0 is max-mass all arms;
seed-1/2 position migration = caveat), pretrained-ViT confound (+ h_ratio-rises-from-~1 defense),
≤1B vs 5B scale, textinit seed variance, seed-0 provenance, domain shift (ADR-0001).
Plus: single-seed RF. MMStar/accuracy NEVER measured — no benchmark implication anywhere.

### 07-conclusion.md — ~200 wds
Purpose: restate conjunction-novelty claim, one forward look (random-ViT, per-position norm
scan, scale). No new claims.

### 08-appendix.md — no budget cap
Figs 1/3/5 + full per-seed tables (from session4_n3_audit.md) + per-position mass table
(seed-0) + refs. Fig captions verbatim from figure-suite handoff 0e9a unless user rewrites.

---

## Figure map (Auditor split, signed layout)

| slot | fig | caption source | claim boundary |
|---|---|---|---|
| main | Fig 2 phase-portrait (HERO) | 0e9a | 4 levers → 4 corners |
| main | Fig 6 sink stripes | 0e9a (softened) | baseline-absent / textinit-total / inherited@init ONLY |
| main | Fig 4 birth-map/lead-lag | 0e9a | ordering evidence |
| appx | Fig 3 per-head scatter | 0e9a | sign flips, NOT "uncorrelated" |
| appx | Fig 5 entropy collapse | 0e9a | collapse tracks concentration only |
| appx | Fig 1 layer×head grid | 0e9a | orienting |

## Standing guardrails (checked on every section review)

1. Quantitative claims cite `session4_n3_audit.md` / `GATE_A_REPORT.md` /
   `preprint_readiness_audit.md` / 0e9a handoff only — never raw REPORT.md.
2. No MMStar / accuracy / benchmark claims (not measured).
3. Novelty = conjunction wording above; Related Work distinguishes the two text-decoupling papers.
4. textinit = range/median, never point; never "pinned at pos0."
5. Fig 6 sigmoid panel not load-bearing.
6. Drafter never submits. Before "ready": final arXiv cs.CV/cs.LG Jun 10–29 scoop browse +
   first-time-submitter endorsement sorted — user does both.

## STATUS (2026-07-02)

- v2 complete, user-directed full draft: all sections written (Drafter, at user request —
  division-of-labor override), figures embedded, PDF rendered.
- Artifacts: `sections/01–09` (source of truth) → `python3 build.py` → `paper-v2.html`,
  `paper-v2.pdf`, `draft-v2.md` (stitched). Figures self-contained in `figures/`.
- Title: "Four Levers, Four Corners: …". Author name spelling UNCONFIRMED — verify before
  submission.
- References: ALL 20 bibliographically verified by Researcher (handoff-...-d927, report
  2026-07-07). 9 corrected (IDs/authors/titles), 0 fabricated. Flags cleared.
- Researcher report in `sources/researcher-related-work.md`.
- Review fixes applied (2026-07-10): "repetition-confound-free" wording + (n=1) inline at
  RF headline; "no fixed-sign coupling" replaces "single shared mechanism"; Fig 3 sigmoid
  panel DROPPED (dump missed true top head L7H3 — caveat now prose-only); Table 1 header
  reworded (textinit read at 60M); GQA verified from `models/config.py` (9Q/3KV per layer
  → 270 attn obs but 90 independent value obs) and stated in §2 + §3.3 + Fig A2 caption;
  Pearson-star independence caveat added to §3.3.

## BLOCKERS / user actions

- [ ] Read v2 in own voice; edit sections; rerun `python3 build.py`.
- [x] Bibliographic verification — DONE (Researcher, 2026-07-07; corrections applied).
- [ ] Confirm author-name spelling in `build.py` before submission.
- [ ] Final arXiv cs.CV/cs.LG Jun 10–29 scoop browse + first-time-submitter endorsement.
