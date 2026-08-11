# Reviewer-revision tracker — arXiv v1 (temporary; delete before/after upload)

Source: researcher review, 2026-08-10. Process: **user edits by hand; Director reviews for
validity before push.** Nothing is edited without proposing first.

**ONE PAPER, ONE TRIM (decided 2026-08-10).** Separate arXiv and workshop versions are not worth
maintaining. There is one paper, and it must land **under 8 pages of body** (excl. refs/appendix)
— which satisfies VLM4RWD, so compaction is never revisited before Aug 30. **≤8pp is a hard
constraint, not a target.** The binding test is the rendered page count in the NeurIPS 2026
template; word budgets below are only a proxy — measure real pages.

Target: body (§1–7) **9,129 → ~5,500 words**. Now **7,329**.

⚠ **Pass two will not close the gap on its own.** The three remaining sections total **2,538**
(abstract 406, intro 921, related work 1,211); trimming them to ~1,550 removes ~988 → **~6,341**,
still ~840 over. The already-trimmed sections sit ~640 above their own targets (method 1,309 vs
~1,100; results 2,215 vs ~1,900; conclusion 341 vs ~250). Expect a **second-round cut across
already-trimmed sections** — to be proposed and reviewed, not applied unilaterally.

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
| 01-abstract | 406 → **296** | ~200 | ✅ reframed on three signatures; caveats and closing disclaimer removed |
| 02-intro | 1031 → **870** | ~700 | ✅ hallucination 5 → 1; new which-signature-you-measure frame |
| 03-method | 1630 → **1314** | ~1100 | ✅ notation block kept; reporting details → App. F |
| 04-results | 3341 → **2193** | ~1900 | ✅ per-seed table → App. B; sigmoid raw/normalized → App. D; hypotheses → App. E |
| 05-related-work | 1211 → **916** | ~650 | ✅ hallucination ¶ → 1 sentence; [9]/[10] ¶ and Luo [11] clause kept |
| 06-limitations | 1075 → **926** | ~900 | ✅ deduped against §2; every item kept |
| 07-conclusion | 435 → **341** | ~250 | ✅ shortened |
| 09-appendix | 1051 → **2179** | ~1500 | ✅ grew: B per-seed, D measurement, E hypotheses, F reporting |

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
- [ ] **Citation integrity sweep**: after the hallucination cut, remove any now-orphaned
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
