# 2. Setup

**Model and token layout.** All runs use a 222M-parameter nanoVLM [18]. A pretrained,
trainable SigLIP-B/16 vision encoder [16] feeds a decoder with the SmolLM2-135M
architecture [17] through a learned modality projector. The decoder has 30 layers of
grouped-query attention. Each layer has 9 query heads that share 3 KV heads. The decoder
therefore has 270 (layer, query-head) pairs but only 90 (layer, KV-group) value
projections. This distinction matters when we count per-head observations (§3.3). The
decoder trains **from random initialization** in every arm except *textinit*, because the
point is to watch the signatures form. Each sequence holds 49 image tokens as a causal
prefix, then 79 left-padded text tokens, for 128 tokens in all. **Position 0 is the first
image token, and there is no BOS token.** What happens at position 0 is therefore a property
of the visual prefix. It is not inherited BOS machinery.

**Arms.** We use four training levers. Each lever targets one sink-relevant mechanism.
Everything else stays byte-identical across arms.

| arm | attention | LM init | ViT init | lever precedent |
|---|---|---|---|---|
| *baseline* | softmax | random | pretrained | — |
| *g1gate* | softmax + elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | G1 gating [5] |
| *sigmoid* | unnormalized sigmoid, no softmax | random | pretrained | Gu et al. [1] |
| *textinit* | softmax | pretrained SmolLM2-135M | pretrained | — (novel control) |

The *g1gate* lever and the *sigmoid* lever have established sink effects in text-only
models [5, 1]. The *textinit* lever has no precedent in the sink literature. It works as an
inheritance control. It imports whatever sink structure text pretraining already built into
SmolLM2.

**Data and the two training regimes.** The four-arm comparison trains on four curated
subsets of `the_cauldron` [19], which hold about 146K images. The arms are matched at about
100M tokens each. *textinit* stops at 60M tokens, because it reaches its three-signature
floor earlier (§3.1). Reuse of a 146K-image pool gives high visual-epoch counts. A reader
could therefore treat a "no sink emerges" result as an overfitting artifact. The **RF** arm
(random-fresh) answers that objection. It re-trains the *baseline* recipe on a fresh
FineVision stream [20] to 1B tokens. That stream holds about 4.6M natural images and gives
2.39 effective visual epochs, against about 74 epochs for the repeated pool. A change of
dataset trades the repetition confound for a domain-shift confound. We accept that trade
and document it. The fresh pool holds natural images, leans heavily on COCO, and overlaps
the repeated subsets by less than 3%. We did not run a third, domain-matched control (§5).

**Three signatures, tracked separately.** We log the three sink symptoms that the text-LM
literature reports together as independent per-(layer, head) quantities. The metric
conventions follow Gu et al. [1] at a fixed sequence length.

- **Concentration** — Sink<sup>ε</sup><sub>1</sub>: the fraction of (layer, head) pairs
  whose *mean* attention to position 0 is above ε. We also log the mean and the maximum
  attention to position 0. We use the ε = 0.3 default of [1] and check robustness at
  ε ∈ {0.2, 0.4}. Cross-arm tables report the stricter ε = 0.2, which makes an absence
  claim harder to pass.
- **Value-norm ratio** (v-ratio) — ‖v‖ at position 0 divided by the mean ‖v‖ of the other
  positions. A value below 1 is value-drain [6]. A value above 1 is amplification. Under
  grouped-query attention the value vector belongs to a KV group. Each per-(layer,
  query-head) v-ratio therefore repeats its group value across 3 query heads, which gives
  90 independent value observations, not 270.
- **Massive activation** (h-ratio) — the residual-stream norm at position 0 divided by the
  mean norm of the other positions.

All three metrics **anchor on position 0 by construction**. We state this as a measurement
choice, and we check it. At seed 0, per-position attention mass makes position 0 the
maximum-mass token in every arm (Appendix C). Section 5 handles the remaining seed-level
caveat. The *sigmoid* arm reports the row-normalized attention view, which keeps
concentration comparable across arms. We also log the raw sigmoid mass.

**Probe.** The function `probe_sinks()` re-walks the decoder from the live module weights
in eager mode, in fp32, with no gradients. It is independent of the training path, which
uses fused SDPA kernels, autocast, and `torch.compile`. Every call validates its hidden
states against the real forward pass of the model, to a relative error below 10⁻². The
probe therefore cannot drift from what the trained model computes. The probe batch stays
fixed across all runs and all seeds, so every number in this paper is comparable from run
to run. Probes fire every 100 optimizer steps. That cadence is dense enough to timestamp
the first threshold crossing of each signature (§3.4).

**Recipe.** We use AdamW with weight decay 0.1, following [1], and a gradient clip of 1.0.
The schedule is cosine with 3% warmup. The learning rates are 4e-4 for the language model,
2e-3 for the projector, and 1e-4 for the vision encoder. We use bf16 autocast,
`torch.compile`, and a batch size of 128. Arms in a comparison differ only in the lever
under test. Training stays healthy in all reported runs. On the 1B fresh run, held-out
fresh validation loss falls from 1.46 to 0.638 and tracks train loss with no overfit gap.
