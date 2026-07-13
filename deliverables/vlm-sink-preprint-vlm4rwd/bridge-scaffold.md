# Bridge scaffold — VLM4RWD reframing

Sourced bullets for the 3 touch points, NOT full prose — per standing division of labor, you
write the final sentences in your own voice; I flag drift when you do. Every bullet below
cites either (a) a number already audited in v1, or (b) is explicitly marked as needing
Researcher's fact-check verdict before it can be written as anything stronger than a
hypothesis. **Do not promote any (b) item to a citable claim until that verdict lands.**

---

## §1 — Abstract framing sentence(s)

Goal: 1–2 sentences, appended or lightly woven in, connecting the dissociation finding to why
a grounding/faithfulness audience should care. Insert near the end of `01-abstract.md`
(current final sentence ends "...are separately controllable by ordinary training-time
choices.").

Sourced bullets to draw from:
- (a) Motivation framing, no new claim: attention allocation is the mechanism by which a VLM
  connects language to image content; when attention mass concentrates on a positional
  artifact instead, that mass is — by construction — not being spent on image tokens. This is
  a definitional point about what `Sink^ε` measures (`03-method.md` §2, "Concentration"
  bullet), not a new empirical result.
- (a) Quantified example, already audited: in the *textinit* arm, Sink^0.2 reaches 0.56–0.85
  (Table 1) — the majority of heads pass a substantial concentration threshold on position 0
  alone. Useable as a concrete "how much attention" anchor without overclaiming what it means
  for grounding.
- (b) DO NOT WRITE without Researcher's verdict: any sentence asserting or implying that sink
  concentration *causes*, *predicts*, or *correlates with* grounding failure or hallucination.
  Hold this hedge-level ("we hypothesize", "an open question") regardless of the verdict,
  since the abstract is not the place to introduce a new citation-dependent claim even if one
  exists — the abstract should motivate, not assert.

## §2 — Introduction paragraph

Goal: one new paragraph connecting the paper's identity (signature dissociation during
pretraining) to the workshop's center of gravity (grounding/faithfulness/deployment). Insert
after the existing gap paragraph in `02-intro.md` (the one ending "...a setting no prior
decoupling or co-emergence study has examined") and before "We do it at deliberately small
scale..." — or wherever reads best once you draft it.

Sourced bullets:
- (a) The mechanism argument, restated for an intro (not abstract) register: grounding
  requires attention to land on the right image tokens; a sink is a specific, measurable
  failure mode of *where* attention lands, decoupled (per this paper's own results) from
  whether norm-based signatures also form. Framing the paper's contribution as "establishing
  when/how this precondition forms" is honest — it's what §3.1–3.4 actually show.
- (b) NEEDS RESEARCHER VERDICT before drafting: if prior work has already shown attention
  sinks / massive activations / value-drain relate to VLM grounding or hallucination, THAT is
  the citation this paragraph should lean on (cite it, don't reinvent the connection — per the
  Director's explicit instruction). If Researcher finds nothing, the paragraph should say so
  plainly ("to our knowledge, no prior work has directly tested this link") rather than
  implying a gap that might not exist — do not cite absence of evidence as evidence of
  novelty.
- (a) Explicit scope sentence (recommend including verbatim in spirit, not necessarily
  wording): this paper establishes the training-dynamics precondition; whether the
  dissociation predicts grounding or hallucination behavior is the natural next question, and
  is NOT tested here. This directly mirrors the Director's required framing (handoff §2) and
  should probably be the paragraph's last sentence so the hedge is the thing a skimming
  reviewer reads last, not first.

## §3 — Conclusion / next-steps paragraph

Goal: extend or add to the existing "Next steps" paragraph in `07-conclusion.md` (currently:
random-init ViT control, per-position norm scan, extending RF past 1B tokens) with one more
forward-looking item connecting to grounding — explicitly future work, not a result.

Sourced bullets:
- (a) A 4th next-step item, parallel in structure to the existing 3: testing whether
  signature dissociation (or its absence) predicts grounding/hallucination behavior on a
  benchmark — naming a candidate benchmark is optional and should wait on Researcher's answer
  (a benchmark already used in a hallucination/grounding+sink paper, if one exists, would be
  the natural one to cite/reuse rather than picking cold).
- (a) The existing Limitations line — "we make no claim about how signature dissociation
  relates to downstream capability" (`06-limitations.md`, final paragraph) — should be echoed
  or cross-referenced here, not contradicted. Director's instruction: sharpen for this
  audience, don't soften. Suggest the next-steps item explicitly say the current paper does
  NOT test this, immediately before or after naming it as future work.

---

## What Researcher's verdict changes

- **If literature exists** linking sink/massive-activation/value-drain to VLM grounding or
  hallucination: cite it in §2 (intro) as the motivating precedent, note in §1/§3 if relevant.
  Director also flagged this as a possible scoop-adjacent risk — if the existing work is close
  enough to threaten the novelty framing (not just motivate it), that goes back to Director,
  not silently absorbed into the bridge.
- **If nothing exists:** keep every bridge sentence at hypothesis-level explicitly ("we
  hypothesize," "an open question," "to our knowledge, untested") and say so plainly rather
  than implying a literature gap you haven't actually confirmed.
