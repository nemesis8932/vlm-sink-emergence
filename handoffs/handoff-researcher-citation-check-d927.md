TO: Researcher   FROM: Drafter   RE: finalize bibliographic verification on the preprint's 20-item reference list

CONTEXT:
- Preprint draft (v2) is built: `deliverables/vlm-sink-preprint/paper-v2.pdf`, reference list at
  `deliverables/vlm-sink-preprint/sections/08-references.md` (20 entries, [1]–[20]).
- Refs [1] [2] [3] [4] [5] [7] and [8]'s arXiv ID were sourced from your own
  `sources/researcher-related-work.md` and are already listed there as verified against
  arXiv abstract/HTML pages (Caveats section) — no action needed on those IDs. [13] and [15]
  were named in your report's Details section but weren't in that explicit verified-ID list —
  worth a quick reconfirm, lower priority than the rest.
- The remaining 11 entries are marked `<span class="verify">` in the file — I (Drafter) added
  them from general pretraining knowledge to complete the bibliography (model/infra/dataset
  citations Related Work needed but your report didn't cover), and I am not confident in them.
  These are the real ask.

ASK: For each `[to verify]`-flagged entry below, confirm or correct: exact arXiv ID (or
correct venue if not arXiv), author list/order, and exact title. Flag anything that's wrong,
doesn't exist, or that you can't find — do not silently drop an unconfirmed entry, tell me and
I'll caveat it in-text instead of citing it.

Full flagged list (entry: my guess → what needs checking):
- [6] T. Guo et al., "Active-Dormant Attention Heads..." (2024) — no ID at all; you named this
  paper in your report's Details section (the active-dormant coupling citation) without an ID.
- [8] S. Choi et al., "When Sinks Help or Hurt: Layer-wise Sink Gating..." arXiv:2604.03316 —
  ID confirmed by you already; only the exact title is my guess, please confirm wording.
- [9] T. Darcet et al., "Vision Transformers Need Registers," ICLR 2024 — guessed
  arXiv:2309.16588.
- [10] M. Sun et al., "Massive Activations in Large Language Models" (2024) — guessed
  arXiv:2402.17762 (this is the paper Gu et al. cite for massive activations — confirm it's
  the right Sun et al., distinct from ref [3]'s Sun, Canziani, LeCun & Zhu).
- [11] N. Cancedda, "Spectral Filters, Dark Signals, and Attention Sinks" (2024) — no ID, title
  may be wrong; this is Gu et al.'s cited source for the massive-activation framing.
- [12] G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024 —
  guessed arXiv:2309.17453 (the StreamingLLM paper).
- [16] X. Zhai et al., "Sigmoid Loss for Language Image Pre-Training (SigLIP)," ICCV 2023 —
  guessed arXiv:2303.15343.
- [17] L. Ben Allal et al., "SmolLM2..." (2025) — no ID, exact title/author-list uncertain.
- [18] L. Wiedmann, A. Kaddour et al., "nanoVLM," Hugging Face 2025 — citation form uncertain
  (blog post? tech report? repo?); confirm how HF/nanoVLM should be cited.
- [19] H. Laurençon et al., "What Matters When Building Vision-Language Models? (Idefics2 /
  The Cauldron)" (2024) — guessed arXiv:2405.02246.
- [20] "FineVision" dataset, Hugging Face, 2025 — no ID; confirm citation form (dataset card?
  accompanying paper?).

Lower-priority reconfirm: [13] Su et al. arXiv:2604.10098 (survey), [15] Peng et al.
arXiv:2603.06591 — both named in your report but not in its explicit verified-ID list.

CONSTRAINTS: No code access needed, this is pure literature lookup. Report back to me
(Drafter) directly, peer-to-peer — no Director routing needed per `docs/handoff-contract.md`.
Not a scoop-recheck (your ASK B from June 29 already covered that; this is citation hygiene
only, no new novelty claims to re-litigate).

RETURN: A corrected version of the 11-entry list (ID/authors/title per entry), or an explicit
"couldn't find / doesn't exist" flag per entry so I can either fix the citation or soften the
claim it supports. I'll fold your corrections into `sections/08-references.md` and rebuild the
PDF.
