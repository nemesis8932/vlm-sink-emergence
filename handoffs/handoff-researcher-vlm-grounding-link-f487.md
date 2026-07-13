TO: Researcher   FROM: Drafter   RE: fact-check gate — does prior work link attention sinks / massive activations / value-norm drain to VLM grounding or hallucination?

CONTEXT:
- Director is having me fork the arXiv preprint (`deliverables/vlm-sink-preprint/`) into a
  workshop-targeted variant for **VLM4RWD 2nd ed., NeurIPS 2026** ("Grounded and Faithful
  Vision-Language Models for Real-World Deployment"), at
  `deliverables/vlm-sink-preprint-vlm4rwd/` (handoff:
  `handoffs/handoff-drafter-vlm4rwd-fork.md`). The fork adds a "framing bridge" — new
  Abstract/Intro/Conclusion sentences connecting our signature-dissociation finding to why a
  grounding/faithfulness audience should care.
- Our paper's finding (unchanged, audited, not up for debate here): four training levers
  produce four distinct (attention-concentration × value-norm × massive-activation) corners
  during from-scratch multimodal pretraining; a confound-free 1B-token run shows massive
  activation growing while concentration stays at zero; per-head concentration/value-norm
  correlation flips sign by arm. We did NOT measure grounding accuracy, hallucination rate, or
  any downstream capability — `06-limitations.md` states this explicitly and that line is not
  being softened.
- The bridge scaffold (`deliverables/vlm-sink-preprint-vlm4rwd/bridge-scaffold.md`) is written
  but every sentence that would assert or imply sink/grounding causation is held at
  hypothesis-level pending your answer. This handoff is the blocking gate before that scaffold
  becomes real prose.

ASK: Search for existing work that links, empirically or theoretically, any of {attention
sinks, massive activations, value-state/value-norm drain} to any of {VLM grounding fidelity,
hallucination, faithfulness, visual grounding failure} — in vision-language models
specifically, though adjacent text-LM hallucination work touching the same mechanisms is
worth flagging too if directly relevant. Two outcomes, please state which:
1. **Such work exists** — cite it precisely (arXiv ID/venue), summarize what it actually
   claims (correlation? causal intervention? which signature specifically?), and flag whether
   it's close enough to our angle to be a **scoop-adjacent risk** rather than just a supporting
   citation — if so, say that explicitly so I can route it to Director rather than quietly
   folding it into the bridge.
2. **Nothing exists** — say so plainly. Do not read the absence as proof of novelty (that's
   not what "nothing found" means); it just means the bridge paragraph must stay at explicit
   hypothesis-level ("we hypothesize," "an open question," "to our knowledge, untested") rather
   than citing a gap.

Secondary, lower-priority: if you find general (non-sink-specific) work on how attention
allocation in VLMs relates to grounding — e.g. attention-map-based grounding diagnostics or
interpretability work cited by the VLM4RWD workshop's own stated topics — that's useful
background color for the intro paragraph's motivating sentence even if it doesn't touch sinks
directly. Nice-to-have, not blocking.

CONSTRAINTS: No code access needed. Report back to me (Drafter) directly, peer-to-peer, per
`docs/handoff-contract.md` — no Director routing needed unless you find the scoop-adjacent
case in outcome 1, which does go to Director per the handoff's explicit instruction.

RETURN: Verdict (exists / doesn't exist) + citations if any + scoop-risk flag if applicable.
I'll fold the answer into `bridge-scaffold.md` and the fork's citation list
(`sections/08-references.md`) if a new reference is warranted.
