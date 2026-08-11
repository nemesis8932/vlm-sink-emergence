# Reviewer-revision tracker — arXiv v1 (temporary; delete before/after upload)

Source: researcher review, 2026-08-10. Process: **user edits by hand; Director reviews for
validity before push.** Nothing is edited without proposing first.

Target: body (§1–7) **9,129 → ~5,500 words** (now **7,422**; the remaining gap sits almost
entirely in the three blocked sections, 01/02/05 = 2,648 words against ~1,550 target) (≈8pp NeurIPS format; +refs +appendix ≈ 12pp total).
One pass serves both arXiv v1 and the eventual VLM4RWD ≤8pp cut.

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
- **Note for August**: this weakens venue fit for VLM4RWD ("Grounded and Faithful VLMs"). If a
  grounding paragraph is wanted for the workshop build only, treat it as a separate
  build-conditional decision then — do not fork the prose now.

---

## Word budget

| section | now | target | main action |
|---|---:|---:|---|
| 01-abstract | 406 | ~200 | ⏸ blocked — shorten; **remove caveats entirely**; reframe opening to three-signature motivation |
| 02-intro | 1031 | ~700 | ⏸ blocked — tighten; hallucination 5 → ≤1 |
| 03-method | 1630 → **1314** | ~1100 | ✅ notation block kept; reporting details → App. F |
| 04-results | 3341 → **2193** | ~1900 | ✅ per-seed table → App. B; sigmoid raw/normalized → App. D; hypotheses → App. E |
| 05-related-work | 1211 | ~650 | ⏸ blocked — halve; keep cite-and-distinguish ¶ intact |
| 06-limitations | 1075 → **926** | ~900 | ✅ deduped against §2; every item kept |
| 07-conclusion | 435 → **341** | ~250 | ✅ shortened |
| 09-appendix | 1051 → **2179** | ~1500 | ✅ grew: B per-seed, D measurement, E hypotheses, F reporting |

---

## Items

- [x] **Consolidate repeated caveats → Limitations** (unblocked sections). Results/method now
      cross-reference §5 instead of restating. Abstract still blocked.
- [ ] ⏸ **Abstract**: shorter, no caveats, keep the audited novelty sentence. BLOCKED.
- [x] **Per-seed results → appendix**; main text keeps the collapsed four-corner view.
      Tables renumbered 1–3; old Table 1 is now Appendix B.
- [ ] ⏸ **Related work**: shorten. Keep the [3]/[4] cite-and-distinguish passage and the Luo [7]
      narrowing clause — those are review-critical, not padding. BLOCKED.
- [x] **Figure captions**: shortened (Figs 1–3). Seed, checkpoint and token labels kept;
      smoothing and sigmoid-provenance detail moved to Appendix D.
- [x] **Sentence-level**: done in the unblocked sections.
- [x] **Conclusion**: shorter.
- [ ] ⏸ **Hallucination framing** (BLOCKED): cut 12 → 1–2 mentions; reframe motivation around the three
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
