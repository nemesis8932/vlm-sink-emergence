TO: Drafter   FROM: Director   RE: onboard — build/maintain the arxiv v1 preprint spine; user writes the human prose

DOC: `deliverables/vlm-sink-preprint/` (doc-slug = vlm-sink-preprint). Target = ~6–8pg empirical
preprint, arxiv v1, workshop/TMLR-grade. Deadline: 2–3 days (user executes; this is a portfolio
piece for a JHU application, then a workshop→TMLR upgrade path).

DIVISION OF LABOR (important — respect it):
- The USER writes the final prose in their own voice and creates all visuals by hand. They are NOT
  outsourcing the writing.
- YOU own the OUTLINE SPINE + guardrails + fact-checking. Draft a section ONLY when the user asks;
  otherwise supply sourced bullet-scaffolds they write into, and review what they write against the
  spine. Keep your prose minimal — the human voice is theirs.

FIRST TASK: create `deliverables/vlm-sink-preprint/outline.md` — the one fully-loaded artifact:
the one-sentence central claim, the 3 contribution bullets, section headings with 1-line purpose +
word budget. Then stop and let the user write. Sections live in `sections/NN-name.md`, one per file.

WHERE TO FIND WHAT (all in-repo unless noted):
- Findings/narrative: `REPORT.md` (context only — per guardrail 1, quantitative claims must cite the
  AUDITOR-APPROVED sources below, not raw REPORT text).
- Auditor-approved results: `deliverables/session4_n3_audit.md` (the n=3 four-way dissociation table
  + 4-corner analysis) and `runs/rf_fresh_baseline/GATE_A_REPORT.md` (Gate-A RF).
- Figures + layout + captions + audit flags: `analysis/fig1–6.{svg,png}` and
  `handoffs/handoff-director-figure-suite-0e9a.md` (main = Fig2 hero / Fig6 / Fig4; appendix = 1/3/5).
- Related work + safe wording + verified citations: the Researcher report at
  `~/Downloads/compass_artifact_wf-dda9e750-a812-4c43-9a7b-a6934a953609_text_markdown.md`
  (NOT in repo — ask the user to copy it into `deliverables/vlm-sink-preprint/sources/` for durability).
- Method/setup detail: `docs/conventions.md`, `docs/experiments.md`. Terminology: `CONTEXT.md`.
  Design rationale: `docs/adr/`. Limitations: `docs/open-questions.md`.

SECTION ORDER (write data-first, frame last — this is what makes 2 days feasible):
Method/Setup → Results → Related Work (near drop-in from Researcher) → Intro → Limitations →
Conclusion → Abstract (last). Appendix = figs 1/3/5 + full per-seed tables + refs.

PAPER-SPECIFIC GUARDRAILS (beyond your standard ones):
1. NOVELTY = the CONJUNCTION only: "first to show all three sink signatures independently
   controllable during FROM-SCRATCH MULTIMODAL pretraining, adding value-norm drain as a third
   axis." NEVER "first to decouple" (Researcher: 2603.05498 + 2603.17771 already decouple two axes
   in text). Related Work MUST cite-and-distinguish those two; reserve "causally unified" strictly
   for 2510.06477; NEVER write "one inseparable phenomenon" (strawman).
2. textinit magnitudes as RANGE/MEDIAN (h_ratio 44.8/8.2/14.7, seed-0 outlier). Do NOT write
   "pinned at pos0" — the Auditor found positional decoupling (attn→pos1, ‖h‖→pos13, drain→pos13 =
   three different tokens); frame that as a supporting observation, not a second headline.
3. Fig 3 = "no UNIVERSAL coupling, correlation sign is arm-dependent" (+0.67→−0.76), NOT
   "uncorrelated." Fig 6 sigmoid panel is under-powered (wrong head dumped) → claim only
   baseline-absent / textinit-total / inherited-at-init from it.
4. MMStar accuracy NOT measured — never imply a benchmark/accuracy result.
5. State limitations honestly + up front: 222M/1B scale (vs Gu 5B), single-seed RF, pretrained ViT
   confound, domain-shift (ADR-0001), textinit seed-variance.

CONSTRAINTS: outline-first, one section file at a time, never load the whole doc. Fact-check
requests go peer-to-peer to Auditor/Researcher (docs/handoff-contract.md), no Director routing.
You NEVER publish/submit — when a draft is "ready," print: "Confirm on arXiv yourself before
submitting" + remind the user to (a) do the final arxiv cs.CV/cs.LG Jun 10–29 browse and (b) sort
first-time-submitter endorsement.

RETURN: outline.md spine for user sign-off; thereafter, guardrail flags + sourced scaffolds as the
user writes. Report up to Director only on blockers or claim-risks.
