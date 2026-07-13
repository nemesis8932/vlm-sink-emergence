# Fork delta — vlm-sink-preprint-vlm4rwd

Forked from `deliverables/vlm-sink-preprint/` (arXiv v1, unchanged, stays the priority copy).
Target: **VLM4RWD 2nd ed., NeurIPS 2026 workshop, Sydney, Dec 2026** — "Grounded and Faithful
Vision-Language Models for Real-World Deployment." Non-archival; NeurIPS 2026 template, ≤8pp
excl. refs/appendix, double-blind/OpenReview. **Deadline Aug 31, 2026, notify Sep 30.**
(Source: https://vlm4rwd.github.io/, fetched by Director 2026-07-14 — not elsewhere in-repo.)

This is a **reframing fork, not new research.** Science/figures/tables are locked and audited
in v1 — do not reopen Method/Results for new claims here.

---

## What's copied verbatim (no changes)

| file | status |
|---|---|
| `sections/03-method.md` | copied verbatim |
| `sections/04-results.md` | copied verbatim |
| `sections/05-related-work.md` | copied verbatim |
| `sections/06-limitations.md` | copied verbatim — **the "no claim about downstream capability" line is load-bearing for this fork; see bridge-scaffold.md, do not soften it** |
| `sections/08-references.md` | copied verbatim (+ 1 new entry pending Researcher gate, see below) |
| `sections/09-appendix.md` | copied verbatim |
| `figures/fig{1..6}_*.svg` | copied verbatim, zero re-renders |
| `sources/researcher-related-work.md` | copied verbatim (existing prior-art pass) |
| `build.py` | copied verbatim |

## What's new (the actual fork work)

| file | status |
|---|---|
| `sections/01-abstract.md` | **copied verbatim as starting point** — bridge sentence(s) NOT yet written into it. See `bridge-scaffold.md` §1 for sourced bullets; user writes final prose. |
| `sections/02-intro.md` | **copied verbatim as starting point** — bridge paragraph NOT yet inserted. See `bridge-scaffold.md` §2. |
| `sections/07-conclusion.md` | **copied verbatim as starting point** — bridge next-step NOT yet inserted. See `bridge-scaffold.md` §3. |
| `bridge-scaffold.md` | **NEW file** — sourced bullet-scaffolds for the 3 touch points above, per standing division of labor (Drafter scaffolds, user writes final prose in their own voice). |

Per the handoff and standing guardrails, I have not written full replacement prose into
01/02/07 — the copies are unmodified base-paper text pending your bridge sentences going in
by hand from the scaffold. This keeps every word in the fork traceable to either "audited v1
text, byte-identical" or "new, sourced, user-authored."

---

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

**Flagged trim (not applied):** if the real render goes over 8pp, the cleanest cut is
**demoting Fig 3 (sink-stripe) from main body to appendix.** Rationale: Fig 1 (hero,
4-corners) and Fig 2 (ordering/lead-lag) carry the core claim; Fig 3 is corroborating
single-head evidence, already partially caveated (sigmoid panel dropped per the 2026-07-10
review pass), and its argument survives in prose (§3.4 already states the stripe-absent /
stripe-total / inherited-at-init claims in text, citing the figure but not solely dependent on
it). This is a **figure-placement change, not a content cut** — nothing gets deleted, per the
handoff's explicit instruction. Flagging per the ask; not applying without your go-ahead.

---

## Fact-check gate — BLOCKING, routed to Researcher

Handoff requires Researcher confirmation before this fork ships: does existing literature
already link attention sinks / massive activations / value-norm drain to VLM hallucination or
grounding failure? See `handoffs/handoff-researcher-vlm-grounding-link-f487.md`. **Do not
finalize the bridge paragraphs until that comes back** — the citation-vs-hypothesis framing in
`bridge-scaffold.md` §2 depends on the answer.

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

- [ ] Wait on Researcher's fact-check verdict (`handoff-researcher-vlm-grounding-link-f487.md`).
- [ ] Write the 3 bridge insertions from `bridge-scaffold.md` into 01/02/07 in your own voice.
- [ ] Pick a title/subtitle option (A/B/C/D above) or propose your own.
- [ ] Confirm page budget against the real NeurIPS 2026 template once bridge text is final.
- [ ] Strip author-identifying info (HF username in conclusion; any other self-identification)
      for double-blind submission.
- [ ] Decide on the Fig 3 → appendix demotion if the real render exceeds 8pp.
