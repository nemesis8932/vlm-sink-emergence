# 2. Setup

**Model and token layout.** All runs use a 222M-parameter nanoVLM [18]: a SigLIP-B/16
vision encoder [16] (pretrained, trainable) feeding a decoder with the SmolLM2-135M
architecture [17] through a learned modality projector. Except in the *textinit* arm, the
decoder trains **from random initialization** — the point is to watch sink signatures form,
not to probe a model that already has them. Sequences are 49 image tokens (a causal prefix)
followed by 79 left-padded text tokens, 128 total. **Position 0 is the first image token;
there is no BOS.** Any sink behavior at position 0 is therefore a property of the visual
prefix, not inherited BOS machinery.

**Arms.** Four training levers, each targeting one sink-relevant mechanism, with everything
else held byte-identical:

| arm | attention | LM init | ViT init | lever precedent |
|---|---|---|---|---|
| *baseline* | softmax | random | pretrained | — |
| *g1gate* | softmax + elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | G1 gating [5] |
| *sigmoid* | unnormalized sigmoid, no softmax | random | pretrained | Gu et al. [1] |
| *textinit* | softmax | pretrained SmolLM2-135M | pretrained | — (novel control) |

The *g1gate* and *sigmoid* levers have established text-only sink effects [5, 1]; *textinit*
has no sink-literature precedent and acts as an inheritance control — it imports whatever
sink structure text pretraining already built into SmolLM2.

**Data, and the two training regimes.** The four-arm comparison trains on four curated
subsets of `the_cauldron` [19] (~146K images), matched at ~100M tokens/arm (60M for
*textinit*, which reaches its three-signature floor earlier; §4.1). Reusing a 146K-image
pool means high visual-epoch counts, so a "no sink emerges" reading could in principle be an
overfitting artifact. The **RF** run (random-fresh) therefore re-trains the *baseline* arm
on a fresh FineVision stream [20] (~4.6M natural images; 2.39 effective visual epochs at 1B
tokens vs. ~74 for the repeated pool) to 1B tokens. Swapping datasets trades the repetition
confound for a domain-shift confound, which we accept and document (the fresh pool is
natural-image, COCO-heavy, <3% overlap with the repeated subsets) rather than run a third,
domain-matched control (§5).

**Three signatures, tracked separately.** The three sink symptoms reported together in the
text-LM literature are logged as independent per-(layer, head) quantities, following the
metric conventions of Gu et al. [1] at fixed sequence length:

- **Concentration** — Sink<sup>ε</sup><sub>1</sub>: the fraction of (layer, head) pairs
  whose *mean* attention to position 0 exceeds ε, plus mean/max attention→pos0. We use the
  ε = 0.3 default of [1] with robustness checks at ε ∈ {0.2, 0.4}; cross-arm tables report
  the stricter ε = 0.2, which makes absence claims *harder*, not easier, to pass.
- **Value-norm ratio** (v-ratio) — ‖v‖ at position 0 over the mean ‖v‖ of the remaining
  positions. Below 1 is value-drain [6]; above 1 is amplification.
- **Massive activation** (h-ratio) — residual-stream norm at position 0 over the rest.

All three are **pos0-anchored by construction.** This is a stated measurement choice, not a
silent assumption: at seed 0 we verified per-position attention *mass* directly, and
position 0 is the maximum-mass token in every arm (Appendix D); the residual seed-level
caveat is handled in §5. The *sigmoid* arm reports the row-normalized attention view so
concentration is comparable across arms (raw sigmoid mass is also logged).

**Probe.** `probe_sinks()` re-walks the decoder in eager/fp32/no-grad mode from the live
module weights, independent of the training path (fused SDPA kernels, autocast,
`torch.compile`). Every call validates its hidden states against the model's real forward
pass (relative error < 10⁻²), so the probe cannot silently drift from what the trained model
computes. The probe batch is fixed across all runs and seeds, making every number in this
paper comparable run-to-run. Probes fire every 100 optimizer steps — dense enough to
timestamp each signature's first threshold crossing (§4.4).

**Recipe.** AdamW (weight decay 0.1, following [1]), gradient clip 1.0, cosine schedule with
3% warmup (LM 4e-4, projector 2e-3, ViT 1e-4), bf16 autocast, `torch.compile`, batch size
128. Arms in a comparison differ only in the lever under test. Training is healthy in all
reported runs; on the 1B fresh run, held-out fresh validation loss falls 1.46 → 0.638,
tracking train loss with no overfit gap.
