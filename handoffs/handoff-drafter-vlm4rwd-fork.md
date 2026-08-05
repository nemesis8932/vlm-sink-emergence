TO: Drafter   FROM: Director   RE: fork a VLM4RWD-workshop-targeted copy of the preprint — reframe, don't re-research

CONTEXT — what already exists (rest in repo, don't restate):
- The base paper is DONE and content-complete: `deliverables/vlm-sink-preprint/paper-v2.pdf` (+ its
  `outline.md`/`sections/` source). Science, figures, tables, limitations are locked and audited —
  DO NOT reopen Method/Results/figures for new claims. This fork is a REFRAMING exercise, not new
  research.
- The base paper's claim (unchanged in the fork): four training levers land in four distinct
  (concentration × value-norm × massive-activation) corners in from-scratch multimodal pretraining;
  confound-free at 1B fresh tokens; no universal head-level coupling; novelty scoped to the
  conjunction (NOT "first to decouple" — two text-only papers already do two-way decoupling, see
  Related Work in v2).
- This fork targets ONE specific venue: **VLM4RWD (2nd ed.), NeurIPS 2026, Sydney, Dec 2026.**
  Non-archival, so a close variant of the same paper is legitimate and expected — workshops want
  reframed intros, not new papers.

WORKSHOP SCOPE (source: https://vlm4rwd.github.io/, fetched 2026-07-14 — persist this, it's not
elsewhere in-repo):
- Full title: "Grounded and Faithful Vision-Language Models for Real-World Deployment."
- Topics: grounded perception, faithful reasoning, hallucination mitigation, spatial/temporal
  reasoning, causal reasoning, embodied agents, robustness evaluation, benchmarks,
  **interpretability**, world models.
- Format: NeurIPS 2026 template, ≤8 pages excluding refs/appendix, double-blind, OpenReview.
- Deadline Aug 31, 2026 notify Sep 30.

THE GAP (why this is a fork, not a resubmit-as-is): our paper's core identity is attention-sink
SIGNATURE DISSOCIATION during pretraining — it does NOT measure grounding, faithfulness, or
hallucination directly (no benchmark, stated explicitly in §5 Limitations). The workshop's center of
gravity is grounding/faithfulness/deployment; "interpretability" is one topic among several, not the
theme. A reviewer skimming for grounding/hallucination relevance may see this as off-topic unless we
build an EXPLICIT, HONEST framing bridge. That bridge is the entire value-add of this fork.

ASK — produce `deliverables/vlm-sink-preprint-vlm4rwd/` as a forked copy:

1. **Copy, don't rewrite, Method/Results/figures/tables verbatim.** These are audited; touching
   numbers or claims here is out of scope for this task.

2. **NEW framing bridge (the actual work).** Add/rewrite ONLY: Abstract framing sentence(s), an
   Introduction paragraph, and a Conclusion/Next-steps paragraph that connect signature dissociation
   to grounding/faithfulness — HONESTLY HEDGED, not asserted as measured:
   - Legitimate angle: attention allocation is the mechanism by which a VLM grounds language in image
     content; if a large share of attention mass is absorbed by a positional artifact (the sink)
     rather than image content, that is a PRECONDITION question for grounding fidelity — worth
     establishing when/how it forms before asking whether it degrades grounding.
   - Frame explicitly as: "this paper establishes the training-dynamics precondition; whether
     sink/value-drain/massive-activation dissociation predicts grounding or hallucination behavior
     is the natural next question" — future work, not a result of this paper.
   - DO NOT write or imply that we measured hallucination, grounding accuracy, or faithfulness. The
     existing Limitations line "we make no claim about how signature dissociation relates to
     downstream capability" must be preserved and, if anything, sharpened for this audience, not
     softened.

3. **FACT-CHECK GATE before this ships — route to Researcher, peer-to-peer, before finalizing:**
   ask Researcher to check whether any existing literature already links attention sinks / massive
   activations / value-norm drain to VLM hallucination or grounding failure. If such work exists,
   it's the legitimate citation for the bridge paragraph (cite it, don't reinvent it) — and it may
   also be a scoop-adjacent risk worth flagging to Director. If nothing exists, keep the bridge
   explicitly hypothesis-level ("we hypothesize," "an open question") and say so — do not cite
   absence of evidence as evidence.

4. **Retitle/subtitle, minimally.** Keep the core title recognizable (same paper, same arxiv
   priority claim) but consider an added subtitle or framing clause signaling relevance to the
   venue — propose options, don't just pick one; flag to the user for final call, this is their
   authored voice.

5. **Page budget check.** Confirm the reframed version still fits ≤8pp excluding refs/appendix under
   the NeurIPS 2026 template once the new paragraphs are added — the base draft is already dense
   with 6 figures; trimming (likely appendix-demotion of a figure, NOT a content cut) may be needed
   and should be flagged, not silently done.

GUARDRAILS (in addition to your standing ones):
- This is a FORK, not a replacement — the arxiv v1 preprint stays as-is, unchanged, for priority.
- Every new sentence in the bridge is subject to guardrail #1 (Auditor-approved findings only) and
  #3 (Researcher confirmation for any novelty/prior-art claim) from your onboarding handoff
  (`handoffs/handoff-drafter-preprint.md`) — this applies doubly here since the bridge is exactly
  where overclaiming risk concentrates.
- Division of labor unchanged: the user writes/approves the final bridge prose in their own voice;
  you scaffold + guard + flag drift. Draft a full paragraph only if asked; otherwise sourced bullets.
- You never submit/publish. When ready, print: "Confirm on OpenReview yourself before submitting"
  + remind the user of the Aug 31 deadline and double-blind requirements (strip author-identifying
  info per NeurIPS 2026 template rules).

RETURN: the forked outline delta (what's copied vs new) for sign-off; the bridge paragraph(s) as
sourced scaffolds; Researcher's fact-check verdict on the hallucination/grounding literature link;
page-budget confirmation or the trim proposal.
