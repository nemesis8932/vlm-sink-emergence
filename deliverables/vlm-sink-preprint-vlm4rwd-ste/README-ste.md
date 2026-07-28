# STE edition — vlm-sink-preprint-vlm4rwd-ste

A prose rewrite of `deliverables/vlm-sink-preprint-vlm4rwd/` (the VLM4RWD workshop fork)
into ASD-STE100 Simplified Technical English, **STE-flavored mode**. Same paper, same
science, same figures. Only the sentences change.

Build: `python3 build.py` → `paper-v2-ste.html`, `draft-v2-ste.md`, `paper-v2-ste.pdf`.

## What did NOT change

- Every number, table cell, and figure. Tables and `figures/*.svg` are byte-identical to
  the fork. A numeric-token diff of the two stitched drafts shows no lost or altered value.
- Every claim boundary and hedge: the conjunction-scoped novelty sentence, the
  range/median treatment of *textinit*, the single-seed labels, the "we make no claim about
  how signature dissociation relates to downstream capability" line, the not-measured
  hedge in the grounding bridge.
- References [1]–[24], unchanged.
- Double-blind state: author and email anonymized, repro links withheld.

## How the protocol was read for a research paper

STE is written for maintenance manuals. Three rules need interpretation before they fit an
empirical paper.

1. **One name for one thing.** Applied strictly, and it is the largest single change. The
   fork used "arm", "run", "lever", and "configuration" for overlapping ideas. This edition
   fixes the vocabulary: a **lever** is the training change under test, an **arm** is a
   training configuration that a lever defines (baseline, g1gate, sigmoid, textinit, RF),
   and a **signature** is one of the three measured quantities (concentration, value-norm
   drain, massive activation).
2. **Short common words.** Applied, except where a technical term is the correct name.
   "Dissociate", "concentration", "grouped-query attention", and "monotonically" stay.
   Replacing them would cost precision, which STE never asks for.
3. **Sentence length.** The descriptive cap of 25 words drives most of the rewrite. Long
   compound sentences became sequences of short declaratives, and every semicolon became a
   period.

Also applied: active voice with a named actor, plain verbs instead of nominalizations, no
contractions, and no marketing adjectives.

## Deliberate exceptions (2)

Both are longer than 25 words on purpose.

1. **The Gu et al. [1] direct quotation** in §4 (50 words). Never edit a quotation.
2. **The novelty sentence** in §4 Positioning (36 words). It is the Auditor-locked
   verbatim frame from `deliverables/preprint_readiness_audit.md`. Splitting it would risk
   changing the scope of the claim, so it stays as written.

## Lint state

Across sections 01–07 and 09: 0 semicolons, 0 contractions, 0 banned marketing words, and
0 over-length sentences other than the two exceptions above. A few apparent hits from the
mechanical checker are artifacts of bold run-in headings, not real long sentences.

## Trade-off to judge before submitting

STE trades cadence for clarity. Short declaratives read as plain and unambiguous, and they
also read as flatter than a normal ML-venue voice. The fork under
`../vlm-sink-preprint-vlm4rwd/` keeps the earlier voice, and both editions carry the same
science. Pick one edition to submit. Do not merge them sentence by sentence, because the
vocabulary rule in §"How the protocol was read" only holds if it holds throughout.

Word count: 6033 body words (sections 01–07), against 5570 for the fork. Sentence
splitting adds about 8%. The letter-format proxy PDF grows from 16 pages to 17. Treat the
page budget as tighter than the fork's, not equal to it: the fork's proxy estimate was
about 7.6–7.8 pages under the real NeurIPS template, so this edition likely lands at or
slightly over 8 pages. **Re-check it in the real template before submitting.** The
pre-agreed trim, if it is needed, is the Fig 3 demotion to the appendix that
`../vlm-sink-preprint-vlm4rwd/outline.md` describes.

The fork also uses `..` paths in this file only as pointers, not as build inputs. This
edition builds standalone from its own `sections/` and `figures/`.
