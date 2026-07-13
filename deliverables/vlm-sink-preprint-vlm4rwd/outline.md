# Fork delta — vlm-sink-preprint-vlm4rwd

Forked from `deliverables/vlm-sink-preprint/` (arXiv v1, unchanged, stays the priority copy).
Target: **VLM4RWD 2nd ed., NeurIPS 2026 workshop, Sydney, Dec 2026** — "Grounded and Faithful
Vision-Language Models for Real-World Deployment." Non-archival; NeurIPS 2026 template, ≤8pp
excl. refs/appendix, double-blind/OpenReview. **Deadline Aug 31, 2026, notify Sep 30.**
(Source: https://vlm4rwd.github.io/, fetched by Director 2026-07-14 — not elsewhere in-repo.)

This is a **reframing fork, not new research.** Science/figures/tables are locked and audited
in v1 — do not reopen Method/Results for new claims here.

---

## File status (updated 2026-07-14, post-verdict execution)

| file | status |
|---|---|
| `sections/03-method.md` | copied verbatim |
| `sections/04-results.md` | **lightly compressed** (user-authorized trim scope; every number, claim boundary, and locked wording byte-identical — cuts are connective tissue only) |
| `sections/05-related-work.md` | **compressed + NEW ¶ "Sinks and hallucination in deployed VLMs"** citing [21–24] with cite-and-distinguish axes |
| `sections/06-limitations.md` | copied verbatim — **the "no claim about downstream capability" line is load-bearing for this fork; do not soften it** |
| `sections/08-references.md` | +4 new refs [21] VAR / [22] SAGE / [23] IP&M / [24] SinkProbe (verification status below) |
| `sections/09-appendix.md` | copied verbatim |
| `figures/fig{1..6}_*.svg` | copied verbatim, zero re-renders |
| `sections/01-abstract.md` | **bridge sentences WRITTEN** (Drafter, user-authorized override of scaffold-only division of labor, 2026-07-14) — hypothesis-level, no abstract citations |
| `sections/02-intro.md` | **bridge paragraph WRITTEN** (after gap ¶): cites [21–24] as the established sink→hallucination line, positions our gap, ends on the not-measured hedge; light compression elsewhere |
| `sections/07-conclusion.md` | **4th next-step WRITTEN** (grounding benchmarks, POPE/CHAIR/AMBER, explicitly untested here) + repro line anonymized for double-blind |
| `build.py` | TITLE → option B; AUTHOR/EMAIL anonymized for double-blind |
| `bridge-scaffold.md` | superseded — verdict landed, prose written; kept as provenance record |

## Researcher verdict (2026-07-14) — gate CLEARED, ruling logged

Generic sink→VLM-hallucination bridge **already exists** (SAGE, VAR, IP&M, SinkProbe,
OPERA/PAI/GIFT cluster) → bridge CITES it, per Director handoff's "cite it, don't reinvent
it" branch. Residual gap confirmed unclaimed: 3-way dissociation × from-scratch pretraining
dynamics × grounding lens. 3 scoop-adjacent papers flagged; escalation resolved by user
acting as Director 2026-07-14: **cite-and-distinguish, fork proceeds.** Distinguishing
axes (now in related-work ¶): mitigation/detection vs. diagnostic; inference-time on
trained models vs. pretraining dynamics; 2-signature coupling vs. 3-way dissociation;
LLM-only ([24]) vs. VLM. Re-run scoop search before camera-ready (field moving fast).

New refs verification: [21] [22] [24] verified against arXiv abs pages by Drafter
2026-07-14 (exact titles/authors). **[23] IP&M: title verified via ScienceDirect; author
list still from Researcher's snippet reconstruction — user verifying publisher page; keep
the VERIFY-BEFORE-SUBMIT flag until confirmed.**

---

## Title / subtitle — DECIDED 2026-07-14: option B, applied to build.py

> "Four Levers, Four Corners: Attention-Sink Signatures Dissociate in From-Scratch
> Vision–Language Pretraining — A Precondition for Grounding Fidelity"

Original options kept below for the record.

## Title / subtitle options — YOUR CALL, not mine

Current v1 title: *"Four Levers, Four Corners: Attention-Sink Signatures Dissociate in
From-Scratch Vision–Language Pretraining."* Options, safest → most reframed:

**A — unchanged.** Keep the title exactly as-is; let the new intro paragraph alone carry the
relevance signal. Zero retitling risk, but weakest for a reviewer skimming a title list for
grounding/faithfulness relevance.

**B — added subtitle clause.** *"...Vision–Language Pretraining — A Precondition for
Grounding Fidelity."* Signals relevance in the title itself; "precondition" (not "cause" or
"driver") keeps the hedge intact even at the title level.

**C — added parenthetical.** *"...Vision–Language Pretraining (Toward Grounded VLMs)."*
Lighter touch than B, softer hedge language, less likely to read as a claim.

**D — question-led subtitle.** *"...Vision–Language Pretraining — Does Where Attention Goes
Matter for Grounding?"* Most workshop-native phrasing (framing as an open question in the
title itself), furthest from the arXiv v1 title, weakest "same paper" recognizability.

Recommend **B** if you want a title-level hook, **A** if you'd rather the bridge paragraph do
all the work and keep both titles identical for cross-referencing. Not picking for you — flag
your choice back and I'll apply it to `build.py`'s `TITLE` constant.

---

## Page budget

Word counts, current v1 body (sections 01–07, i.e. everything except refs/appendix):

| section | words |
|---|---|
| 01-abstract | 226 |
| 02-intro | 733 |
| 03-method | 747 |
| 04-results | 1893 |
| 05-related-work | 659 |
| 06-limitations | 621 |
| 07-conclusion | 309 |
| **total** | **5188** |

**3 main-body figures** (Fig 1 phase-portrait dual-panel, Fig 2 birth-map/lead-lag two-row
multi-panel, Fig 3 sink-stripe 2×4 grid) + **4 tables** (Table 1: 10-row signature table,
Table 2: 4-row corner summary, Table 3: 4-row RF trajectory, Table 4: 6-column correlation
row). 3 more figures + full per-seed tables already live in the appendix (not counted against
the 8pp body limit).

**Estimate, NOT a real render:** NeurIPS 2-column template at ~950 words/page of pure text
puts the body text alone at ~5.5pp. The 3 main figures at reasonable 2-column sizing are
roughly 0.4–0.6pp each (~1.5pp total); the 4 tables add another ~0.3–0.5pp of column-flow
interruption. **Rough total ≈ 7.3–7.5pp before the bridge additions.** The bridge itself
(abstract sentence + intro paragraph + conclusion paragraph, ~300–400 words per
`bridge-scaffold.md`) adds another ~0.35–0.4pp → **≈ 7.7–7.9pp, inside budget but with
near-zero margin.** This is a word-count proxy, not a real NeurIPS-template render — confirm
by pasting final text into the actual template before trusting it.

**Trim decision (2026-07-14):** user chose **keep all 3 main figures; compress body text
instead** (authorized scope: 04-results, 02-intro, 05-related-work; numbers/claims/locked
wordings byte-identical). Applied: ~150 words cut from results, ~30 from intro, ~40 from
related-work, offsetting part of the ~460 words the bridge + new related-work ¶ added.
Net ≈ +240 words vs. pre-bridge → proxy estimate ≈ **7.6–7.8pp, still inside budget,
still near-zero margin.** Fig 3 → appendix demotion remains the fallback ONLY if the real
NeurIPS-template render exceeds 8pp (user explicitly did not pre-authorize it).

---

## Fact-check gate — CLEARED 2026-07-14 (see "Researcher verdict" section above)

Verdict: literature exists → bridge cites it; residual 3-way-dissociation × pretraining gap
confirmed unclaimed; scoop-adjacent flag resolved by Director-role ruling (cite-and-
distinguish, proceed). Handoff `handoff-researcher-vlm-grounding-link-f487.md` closed.

## Standing guardrails (unchanged from v1, doubly enforced here)

1. Quantitative claims cite audited v1 sources only (`session4_n3_audit.md`,
   `GATE_A_REPORT.md`, `preprint_readiness_audit.md`) — same as v1, no new claims introduced.
2. No MMStar / accuracy / benchmark claims. This fork adds a THIRD prohibition specific to
   itself: no grounding / hallucination / faithfulness measurement claims. We did not measure
   any of these. The bridge motivates future work; it does not report a result.
3. Novelty wording unchanged from v1 (conjunction-scoped, cite-and-distinguish the two
   text-only decoupling papers).
4. Drafter never submits. Before "ready": **"Confirm on OpenReview yourself before
   submitting"** + Aug 31, 2026 deadline + double-blind requirement (strip author-identifying
   info per NeurIPS 2026 template rules — this includes the acknowledgments/checkpoint-hosting
   line currently in `07-conclusion.md`, which names your HF username; that line must be
   removed or anonymized for the workshop submission, restored only in a camera-ready).

## BLOCKERS / user actions

- [x] Researcher fact-check verdict — landed 2026-07-14, gate cleared, ruling logged above.
- [x] Bridge insertions into 01/02/07 — written by Drafter (user-authorized override).
- [x] Title option — B, applied to build.py.
- [x] Double-blind stripping — HF/GitHub links replaced by release-on-acceptance line;
      AUTHOR/EMAIL anonymized in build.py.
- [ ] **VERIFY [23] IP&M author list against the ScienceDirect page** (title already
      confirmed; user checking now). Fix `08-references.md` if the list differs.
- [ ] Read the bridge prose in your own voice; edit anything that doesn't sound like you.
- [ ] Confirm page budget against the real NeurIPS 2026 template (proxy says 7.6–7.8pp);
      Fig 3 → appendix is the fallback if over, not pre-authorized.
- [ ] Re-run the scoop search shortly before camera-ready (Researcher rec — field is fast).
- [ ] Later, for arXiv v1 (decided 2026-07-14, "fork now, v1 later"): add the [21–24]
      cite-and-distinguish passage to v1's related work before arXiv submission.
