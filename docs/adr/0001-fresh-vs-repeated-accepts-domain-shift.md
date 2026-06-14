# RF accepts a domain-shift confound to remove the repetition confound

To re-test Gate A free of the repeated-data overfitting confound (the_cauldron 4 subsets →
~74 visual epochs at 1B tok), the RF baseline trains on fresh FineVision natural-image data
instead. This swaps datasets, so a domain-shift confound is introduced *on top of* removing
repetition. We **accept and document** it rather than adding a fresh-repeated control arm
(FineVision capped to ~146K images): the primary, dominant confound is repetition/overfit,
and `data/mixes.py` already minimizes domain shift (COCO-heavy selection, <3% overlap with the
repeated set, natural-image configs only; synthetic/doc/chart/OCR/math excluded).

## Considered options

- **Accept + document (chosen).** Cheapest, ships in one overnight run; isolates the dominant
  confound. Residual domain-shift caveat noted in REPORT.
- **Add fresh-repeated control arm** — FineVision capped to ~146K images, matched repetition,
  fresh domain — would fully isolate repetition from domain. Rejected for now: +1 arm breaks
  the "RF only" scope and doubles overnight cost. Revisit if a reviewer presses the point.
- **Cap fresh tokens to match epochs** — still domain-confounded, weaker. Rejected.

## Consequences

The Gate-A claim from RF reads "no concentration sink at 1B fresh tokens, with a documented
domain-shift caveat," not "no sink, all else equal." The fresh-repeated control is the known
follow-up if challenged.
