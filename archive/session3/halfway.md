# Session-2 half-way report (Run 1 in progress)

**As of:** Run 1 (baseline extend, seed-0) at **~470M / 1000M tokens (~47%)**, generated
live from `runs/baseline_ext/probes.jsonl` while training continues. Run 2 (seed-1, 4 arms)
and Run 3 (reprobe) have not started yet.

## Run 1 — does the baseline concentration sink appear with more training?

Resumed cleanly from the Session-1 final checkpoint (step 18287, 174.4M): the re-probe at
the resume point reproduced S1's numbers exactly (sink 0/0/0, v_ratio 0.711, h_ratio 2.201),
and there was no cold-optimizer loss spike.

Trajectory (mean over 30 layers × 9 heads on the fixed 32-sample probe):

| Mtok | step | Sink^0.2 | Sink^0.3 | mean a→0 | max a→0 | v_ratio | h_ratio | argmax@pos0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 174 (S1 end) | 18287 | 0.000 | 0.000 | 0.0595 | 0.165 | 0.711 | 2.201 | 0.78 |
| 250 | 26200 | 0.000 | 0.000 | 0.0563 | 0.173 | 0.687 | 2.271 | 0.73 |
| 350 | 36700 | 0.000 | 0.000 | 0.0536 | 0.169 | 0.689 | 2.166 | 0.66 |
| 433 | 45400 | 0.000 | 0.000 | 0.0538 | 0.161 | 0.656 | 2.298 | 0.75 |

**Headline (interim):** the attention-**concentration** sink is still entirely absent at 470M
tokens — **2.7× the Session-1 budget** — with `Sink^0.2 = Sink^0.3 = 0.000` throughout and the
strongest single head's mean attention→pos0 stuck at ~0.16–0.18, never approaching the 0.2
threshold. First-crossing of 0.2: **NONE so far.**

Meanwhile the *other two* sink signatures persist unchanged, so the decoupling from Session 1
holds and if anything strengthens:
- **massive activations** (h_ratio) flat at ~2.2–2.3;
- **value-norm drain** (v_ratio) steady ~0.66–0.72 (drifting slightly *more* drained);
- **argmax-key@pos0** stays high (~0.66–0.78): most heads point at pos0 by argmax but none
  concentrate >0.2 of their mass — the "soft sink" of Session 1, now confirmed stable over a
  much longer horizon.

**Preliminary Gate A:** strong interim evidence that the Session-1 absence of a concentration
sink was **not** an undertraining artifact — at 2.7× the tokens the metric is still identically
0.000. Pending the full 1.0B confirmation.

## Caveat that grew with training: overfitting from data repetition

Train loss has fallen 0.30 → ~0.09 while held-out val loss is flat-to-rising (~1.20 vs 1.16 at
S1 end). With only 146,731 unique images, 470M tokens ≈ ~50 visual epochs (≈75 at 1.0B). So the
"no sink at 1.0B" result will be on **heavily repeated** data, not 1B *fresh* tokens (Gu et al.
used 5B fresh). This is the main confound to state alongside Gate A: it remains possible that
fresh-data scale, not token count per se, is what drives concentration-sink formation. The
norm/value signatures, by contrast, are clearly present and robust regardless.

## Still to come this session
- Run 1 to 1.0B (final Gate-A number + first-crossing, if any).
- Run 2: seed-1 × {baseline, textinit, g1gate, sigmoid} to 100M → Gate B (seed agreement;
  the fragile g1gate is the one to watch).
- Run 3: per-head v_ratio (un-pooled), raw per-position sigmoid mass, image-swap invariance.
