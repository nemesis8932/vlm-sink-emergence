# Reviewer-revision tracker — arXiv v1 (temporary; delete before/after upload)

Source: researcher review, 2026-08-10. Process: **user edits by hand; Director reviews for
validity before push.** Nothing is edited without proposing first.

Target: body (§1–7) **9,129 → ~5,500 words** (≈8pp NeurIPS format; +refs +appendix ≈ 12pp total).
One pass serves both arXiv v1 and the eventual VLM4RWD ≤8pp cut.

---

## ⏸ ON STANDBY — blocks §1/§2/§5 rewrites

**Motivation framing.** Awaiting reviewer's answer to:
1. For a preprint v1, can the hallucination bridging experiment stay as *future work*, or does
   that read as overclaiming?
2. If not → trim to 1–2 mentions as downstream stakes, focus on sinks + three signatures.
3. Target ~12 pages for the preprint (eventually <8 for VLM4RWD)?

Current state: **12 hallucination mentions** — intro 5, related work 4, abstract 2, conclusion 1.
Do not start abstract/intro/related-work rewrites until this resolves; everything else can proceed.

---

## Word budget

| section | now | target | main action |
|---|---:|---:|---|
| 01-abstract | 406 | ~200 | shorten; **remove caveats entirely** |
| 02-intro | 1031 | ~700 | tighten; hallucination pending standby |
| 03-method | 1630 | ~1100 | keep notation block; move detail → appendix |
| 04-results | 3341 | ~1900 | **per-seed results → appendix**; biggest single cut |
| 05-related-work | 1211 | ~650 | halve; keep cite-and-distinguish ¶ intact |
| 06-limitations | 1075 | ~900 | **absorbs** migrated caveats, deduped |
| 07-conclusion | 435 | ~250 | shorten |
| 09-appendix | 1051 | ~1500 | grows: per-seed tables land here |

---

## Items

- [ ] **Consolidate repeated caveats → Limitations.** Currently scattered: results 11, method 6,
      abstract 4. See "must survive" below — migration ≠ deletion.
- [ ] **Abstract**: shorter, no caveats, keep the audited novelty sentence.
- [ ] **Per-seed results → appendix**; main text keeps the collapsed four-corner view.
- [ ] **Related work**: shorten. Keep the [3]/[4] cite-and-distinguish passage and the Luo [7]
      narrowing clause — those are review-critical, not padding.
- [ ] **Figure captions**: shorten. Keep seed + checkpoint/token labels (added last round).
- [ ] **Sentence-level**: shorter, concise; drop repetitive and unsubstantiated lines.
- [ ] **Conclusion**: shorter.
- [ ] **Hallucination framing**: pending standby above.

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

- [ ] Rebuild **both** PDFs (`build.py` and `build.py --arxiv`)
- [ ] Confirm named build has author + working repro links; anonymous build leaks neither
- [ ] Clean-room read of the arXiv build before upload
