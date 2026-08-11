# Reviewer-revision tracker — arXiv v1 (temporary; delete before/after upload)

Source: researcher review, 2026-08-10. Process: **user edits by hand; Director reviews for
validity before push.** Nothing is edited without proposing first.

**ONE PAPER, ONE TRIM (decided 2026-08-10).** Separate arXiv and workshop versions are not worth
maintaining. There is one paper, and it must land **under 8 pages of body** (excl. refs/appendix)
— which satisfies VLM4RWD, so compaction is never revisited before Aug 30. **≤8pp is a hard
constraint, not a target.** The binding test is the rendered page count in the NeurIPS 2026
template; word budgets below are only a proxy — measure real pages.

Target: body (§1–7) **9,129 → ~5,500 words**. Now **5,473** — target met. The binding number is pages, and the body renders at **10pp**.

⚠ **Words are no longer the binding constraint; figures are.** With every section at or near
its word target the body renders at **10pp**. Rendered with the three figures stripped out and
nothing else changed, the same text is **8pp**. The two remaining body figures cost 2pp.

---

## ✅ RESOLVED — motivation reframed (user + reviewer, 2026-08-10)

**Decision: drop the hallucination bridging experiment as "future work"; cut hallucination to
1–2 mentions total, as downstream stakes only.** §1/§2/§5 are now UNBLOCKED.

New framing: the attention sink has **three signatures**; prior work focuses on one or another,
and conclusions vary depending on which you measure. That is the motivation — not hallucination.

From **12 mentions** (intro 5, related work 4, abstract 2, conclusion 1) → **1–2 total**.

Concrete consequences to handle:
- **Abstract**: replace the hallucination opening hook with the three-signature framing. The
  closing "we do not test grounding/hallucination" line can go too — with the motivation removed
  there is nothing left to disclaim.
- **Intro**: 5 → at most 1, as stakes, not as the frame.
- **Related work**: 4 → 0–1. Keep the [3]/[4] cite-and-distinguish ¶ and the Luo [7] narrowing
  clause; those are unrelated to hallucination and stay.
- **Conclusion**: 1 → 0.
- **⚠ Citation integrity**: refs **[21], [22], [24]** were added solely for the grounding bridge.
  After the cut, check each is still cited somewhere; **delete any that are now orphaned.** An
  uncited reference is a review flag. (This also supersedes the pending "[21,22] → [22] only"
  abstract fix — moot if that sentence is gone.)
- **Venue-fit note**: this weakens fit for VLM4RWD ("Grounded and Faithful VLMs"). Accepted —
  under the one-paper decision there is no workshop-only variant, so the grounding framing stays
  cut. Do not fork the prose.

---

## Word budget

| section | now | target | main action |
|---|---:|---:|---|
| 01-abstract | 406 → 288 → **226** | ~200 | ✅ reframed on three signatures; caveats and closing disclaimer removed |
| 02-intro | 1031 → 838 → **765** | ~700 | ✅ hallucination 5 → 1; new which-signature-you-measure frame |
| 03-method | 1630 → 1314 → **1144** | ~1100 | ✅ notation block kept; reporting details → App. F |
| 04-results | 3341 → 1924 → **1466** | ~1900 | ✅ target met. Per-seed table → App. B; sigmoid raw/normalized → App. D; hypotheses → App. E |
| 05-related-work | 1211 → 835 → **695** | ~650 | ✅ hallucination ¶ → 1 sentence; [9]/[10] ¶ and Luo [11] clause kept |
| 06-limitations | 1075 → 941 → **908** | ~900 | ✅ deduped against §2, absorbed 4 migrated caveats; every item kept |
| 07-conclusion | 435 → 312 → **269** | ~250 | ✅ shortened |
| 09-appendix | 1051 → 2179 → **2612** | ~1500 | (grew again: §3.4 prose is now App. H, plus Fig. A4) ✅ grew: B per-seed, D measurement, E hypotheses, F reporting |

---

## Items

- [x] **Consolidate repeated caveats → Limitations** (unblocked sections). Results/method now
      cross-reference §5 instead of restating. Abstract still blocked.
- [x] **Abstract**: shorter, no caveats, novelty sentence kept. Opens on the three signatures
      and on why which-one-you-measure changes the conclusion. Closing grounding disclaimer
      removed with the motivation it disclaimed.
- [x] **Per-seed results → appendix**; main text keeps the collapsed four-corner view.
      Tables renumbered 1–3; old Table 1 is now Appendix B.
- [x] **Related work**: shortened. The [9]/[10] cite-and-distinguish passage and the Luo [11]
      Appendix-A.4 narrowing clause are intact (renumbered from [3]/[4] and [7]).
- [x] **Figure captions**: shortened (Figs 1–3). Seed, checkpoint and token labels kept;
      smoothing and sigmoid-provenance detail moved to Appendix D.
- [x] **Sentence-level**: done in the unblocked sections.
- [x] **Conclusion**: shorter.
- [x] **Hallucination framing**: 12 → **2** body mentions (intro 1, related work 1), both as
      downstream stakes only. Conclusion's hallucination next-step removed. Motivation reframed around the three
      signatures (see RESOLVED section above).
- [x] **Citation integrity sweep** (pass 5): 25 references declared, **0 orphaned, 0 dangling,
      0 duplicate numbers**. Nothing had to be deleted or renumbered. Original note follows.
- [ ] ~~**Citation integrity sweep**: after the hallucination cut, remove any now-orphaned~~
      references ([21]/[22]/[24] are the candidates).

---

## Must survive the trim (validation checklist)

Aggressive cutting risks silently deleting a load-bearing qualifier and re-creating an overclaim
we already fixed. These are **inline qualifiers, not caveats** — they stay where they are even
though they look like hedges:

- `(n=1)` on the RF run; `n = 2–3 seeds` on the four-arm comparison
- textinit magnitudes as **range/median**, never a point estimate
- h-ratio described as a **massive-activation proxy** (no channel-level evidence measured)
- **"low-repetition (2.39 effective visual epochs)"** — never "repetition-confound-free"
- **"randomly initialized decoders"** — never "from scratch" (ViT is pretrained)
- **"Qiu-style G1 with our zero-initialized variant"** + the σ(0)=0.5 scale confound
- sigmoid: "no strong pos0-specific residual-norm asymmetry", not "no massive activation"
- g1gate: "near-absent" concentration, "milder drain" value-norm
- Novelty scoped to the **conjunction**; never "first to decouple"
- MMStar **explicitly not run**; no downstream-capability claims
- Correlation reported **descriptively** — no p-value stars, no "no fixed coupling law"

## Numbers policy

All figures/tables must match `deliverables/session4_n3_audit.md` and
`runs/rf_fresh_baseline/GATE_A_REPORT.md` — not raw `REPORT.md`. RF h-ratio stated as
1.43 → 3.22 (≈2.3×); "+130%" is retired.

---

## Close-out

- [x] Rebuild **both** PDFs (`build.py` and `build.py --arxiv`)
- [x] Confirm named build has author + working repro links; anonymous build leaks neither
- [ ] Clean-room read of the arXiv build before upload


---

## Caveat vs inline-qualifier audit (pass 3)

Hedge-phrase density did not fall in the earlier pass because the hits are overwhelmingly
**inline qualifiers**, which must stay. Classification of every flagged phrase:

| section | hits | caveats → migrated to §5 | inline qualifiers (stay) | not hedges |
|---|---:|---:|---:|---:|
| 03-method | 13 | 4 | 7 | 2 |
| 04-results | 15 | 1 | 13 | 1 |

**Migrated to §5** (main text keeps a cross-reference only): the domain-matched control we
did not run; the repetition-vs-domain-shift trade; the justification for calling h-ratio a
proxy; and the RF `val_seen`/`val_unseen` statement, which had been said three times (§2,
§3.2, §5) and is now said once, in §5's new "What cannot be checked on RF".

**Stayed in place** because each scopes a specific number where it appears: `n = 1`,
`n = 2–3 seeds`, "massive-activation proxy", "low-repetition (2.39 effective visual
epochs)", "randomly initialized decoders", "not a factorial design", "descriptive
statistics, no p-values", "understate", "supporting context", "positive long-horizon
trend", the interval-censoring note, and the MMStar line.

## Structural cut — DIRECTOR'S LADDER, EXECUTED (pass 5, 2026-08-12)

Ruling: §5 Limitations stays in the body. §3.4 prose → appendix, approved. Figures stay in
the body if at all possible. Fig 3 (sink stripe) may not move without the user's approval.

| rung | action | body pages | body words |
|---|---|---:|---:|
| — | start of pass 5 | 11 | 6,174 |
| 1 | §3.4 prose → Appendix H (figures kept in body) | 11 | 5,781 |
| 2 | compact every section to its word target | 11 | 5,580 |
| 3 | Fig 2 bottom panel (birth-maps) → Appendix H, Fig. A4 | **10** | 5,555 |
| 4 | Fig 2 entirely → appendix | 10 | 5,471 |
| 4b | **Fig 2 restored to the body** (rung 4 bought 0 pages, and figures-in-body outranks it) | **10** | 5,473 |

**Where it plateaus: 10pp, over the hard limit by 2.** The next rung is Fig 3, which is
blocked pending the user's decision.

**The diagnosis is that this is now a figure problem, not a word problem.** Rendered with all
figures stripped and nothing else changed, the same text is **8pp**. Body words fell 6,174 →
5,473 across this pass and bought exactly one page, all of it at rung 3. Cutting to the word
targets was worth ~0 pages on its own.

Two ways to close the remaining 2pp, both needing a decision:

1. **Move Fig 3 (sink stripe) to the appendix**, which the ruling forbids without approval.
   Measured cost of the two body figures together is 2pp, so this plus Fig 2 reaches 8pp.
2. **Cut ~1,100 more words** with both figures kept. That is a fifth of the remaining body,
   and every section is already at its target, so it would mean removing content, not
   tightening it — most plausibly §3.3 or the §5 items.


---

## External review, 2026-08-12 — four correctness defects, all verified and fixed

Every claim was re-derived from the run data before acting. All four held.

1. **RF held-out loss is not monotone.** `val_loss` has **62 increases across 131
   evaluations** (`runs/rf_fresh_baseline/train_log.jsonl`). "Falls throughout and never
   turns upward" was false in §1, §2, §3.2 and §5. Replaced everywhere with the claim the
   data supports, and that Appendix F already used: negative fitted slope over the second
   half, ends at 0.638, individual evaluations fluctuate.
2. **The ordering paragraph contradicted itself.** It said *sigmoid* crosses concentration
   with neither norm signature crossing, then concluded "in every arm the norm signatures
   precede concentration". Introduced in pass 5 while compacting §3.4 — the original scoped
   that sentence to *textinit*. Rewritten per-arm and scoped to seed 0.
3. **Per-position masses were normalized over a 20-position display slice**, not the full
   128 (`analysis/per_position_attention_from_npz.py`). RF's reported 0.100/0.083 are
   really **0.053 at pos1 and 0.044 at pos0**. Script fixed so the argmax and the fractions
   always use the full sequence. The diffuse-profile reading survives. **Only the RF numbers
   were affected** — the softmax-arm rows of Appendix C already summed to one over the full
   sequence, and sigmoid's 0.30 is raw unnormalized mass, its profile summing to 2.25.
   Logged in Appendix G.
4. **Table 3 printed sigmoid r = −0.04; the exact value is −0.034548**, which rounds to
   −0.03, as Fig. A2 already printed. Table corrected.

Claim tightenings applied: "inherited sink already in place" to a subthreshold
first-position bias with Sink^0.3 = 0.000 at step 0; "90 independent value projections" to
"distinct"; seed-0 scoping added to the correlations in the abstract, §3.3 and the
conclusion; Figure 2's "per-head threshold" to "threshold", since h and v are
layer-aggregates; "repetition is the dominant confound" to "we chose to prioritize the
repetition confound"; "came apart everywhere we looked" and "each axis moved on its own"
softened.

**Not accepted:** dropping "the coupling is optional". That is an existence claim — there
exist levers under which the signatures do not co-move — not an independence claim, and it
is the paper's thesis. Kept.

**Submission-format defect found while checking:** the double-blind build stamped
"July 2026 · Preprint draft v2 · Simplified Technical English edition" on the title block.
Draft and edition language does not belong on a submission copy, and it was stale.
`build.py` now emits an empty date line for the anonymous build, as it already did for the
arXiv one.

### Open, needs the Auditor

`runs/rf_fresh_baseline/train_log.jsonl` **does** carry a `val_seen` series, and it tracks
`val_unseen` closely (mean gap 0.007, last-20 gap 0.021) against the repeated arms' 0.44
versus 1.18. §5 currently says RF has no seen split and that the memorization comparison
cannot be run for it. If that series is genuinely a seen split, this **strengthens** the RF
result rather than breaking it. The provenance was not confirmable from `run_config.json`,
so nothing was claimed in the paper. Do not use it until the Auditor confirms what it is.

### Page route, re-measured AFTER the corrections

The corrections added about 67 words and crossed a page boundary, so the route measured
before them no longer lands where it did.

| variant | before corrections | after corrections |
|---|---:|---:|
| current | 10 | 10 |
| drop Fig 1 only | 10 | 10 |
| drop Fig 2 only | 10 | 10 |
| drop Fig 3 only | **9** | 10 |
| drop Figs 2 + 3 | 9 | 9 |
| drop all three | **8** | 9 |

Text-only was 8pp at 5,492 words and is 9pp at 5,559. **The 8pp text-only boundary sits at
roughly 5,500 body words.** Reaching 8pp with any figure left in the body therefore needs a
figure move *and* a further cut, not one or the other.

**Unresolved, and larger than the trim:** the build is a custom HTML-to-PDF pipeline, not
the official NeurIPS 2026 LaTeX template the CFP requires. `measure_pages.py` is a geometric
proxy for that template, not the template. Confirm the real count in `neurips_2026.sty`
before submitting.
