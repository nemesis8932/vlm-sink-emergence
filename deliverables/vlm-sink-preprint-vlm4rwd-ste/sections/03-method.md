# 2. Setup

**Model and token layout.** All runs use a 222M-parameter nanoVLM [17]. A pretrained,
trainable SigLIP-B/16 vision encoder [18] feeds a decoder with the SmolLM2-135M architecture
[19] through a learned modality projector. The decoder has 30 layers of grouped-query
attention, 9 query heads per layer sharing 3 KV heads, so it has 270 (layer, query-head)
pairs but only 90 (layer, KV-group) value projections, a distinction that matters in §3.3.
It trains **from random initialization** in every arm except *textinit*, because the point
is to watch the signatures form. Each sequence holds 49 image tokens as a causal prefix,
then 79 left-padded text tokens, 128 in all. **Position 0 is the first image token, and
there is no BOS token**, so what happens there is a property of the visual prefix rather
than inherited BOS machinery. *textinit* is the designed exception: it imports
first-position structure from text pretraining and shows a sink before any multimodal step
(§3.1).

**Arms.** We use four training levers. Each lever targets one sink-relevant mechanism.
Everything else stays byte-identical across arms.

| arm | attention | LM init | ViT init | lever precedent |
|---|---|---|---|---|
| *baseline* | softmax | random | pretrained | — |
| *g1gate* | softmax + elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | G1 gating [20] |
| *sigmoid* | unnormalized sigmoid, no softmax | random | pretrained | Gu et al. [6] |
| *textinit* | softmax | pretrained SmolLM2-135M | pretrained | — (novel control) |

The *g1gate* and *sigmoid* levers have established sink effects in text-only models [20, 6].
*textinit* has no precedent in that literature and works as an inheritance control, carrying
in whatever sink structure text pretraining already built into SmolLM2.

**A scale confound in our gate variant.** Qiu et al. [20] use ordinary initialization for
the G1 gate. We initialize at exactly zero, so the sigmoid opens at σ(0) = 0.5 and the arm
begins as a half-scale attention-output intervention as well as a gating one. We call it
**Qiu-style G1 in our zero-initialized variant** throughout (§5).

**Data and the two training regimes.** The four-arm comparison trains on four curated
subsets of `the_cauldron` [21], about 146K images, matched at about 100M tokens per arm.
*textinit* stops at 60M tokens, where its signatures have plateaued (§3.1). Reuse of that
pool gives high visual-epoch counts, so a reader could treat a "no sink emerges" result as
an overfitting artifact. The **RF** arm (random-fresh) answers that objection: the
*baseline* recipe re-trained on a fresh FineVision stream [22] to 1B tokens, over about 4.6M
natural images, at **2.39 effective visual epochs**. RF is therefore a **low-repetition**
run, not a repetition-free one, since examples do repeat about 2.4 times on average. We
estimate the overlap between the fresh pool and the repeated subsets at under 3%, from the
config-level composition of the two pools rather than image-level deduplication. Swapping
datasets trades the repetition confound for a domain-shift confound, which §5 takes up.

**Three signatures, tracked separately.** We log the three sink symptoms that the text-LM
literature reports together, each at its own granularity, following Gu et al. [6] at fixed
sequence length. The decoder has *L* = 30 layers, *H* = 9 query heads per layer, *G* = 3 KV
heads per layer. Every quantity below is averaged over valid query positions and over the
fixed probe batch.

*Concentration* (Sink^ε_1) is the fraction of the *L·H* = 270 (layer, query-head) pairs
whose mean attention to position 0 exceeds ε. We use the ε = 0.3 default of [6] and check
ε ∈ {0.2, 0.4}. Cross-arm tables report the stricter ε = 0.2, which makes an absence claim
harder to pass.

*Value-norm ratio* (v-ratio) is the value norm at position 0 divided by the mean value norm
over the other valid positions, computed per layer and then averaged over the 30 layers.
Below 1 is value-drain [4]. Above 1 is amplification. Under grouped-query attention a value
vector belongs to a KV group and repeats across 3 query heads, so the ratio rests on
*L·G* = 90 independent value projections, not 270. This matters for §3.3.

*Residual-norm ratio* (h-ratio) is the residual-stream norm at position 0 divided by the
mean norm over the other positions, again per layer and then averaged over the 30 layers. We
call it a **massive-activation proxy**, because massive activations are normally defined by
channel-level outliers [2, 5], which we never measured (§5).

All three metrics **anchor on position 0 by construction**. We state this as a measurement
choice and check it: at seed 0, per-position attention mass makes position 0 the
maximum-mass token in every arm (Appendix C). §5 handles the remaining seed-level caveat.
The *sigmoid* arm reports the row-normalized attention view, which keeps concentration
comparable across arms, and we also log the raw sigmoid mass (Appendix D).

**Why a sink is expected at all.** Softmax forces every attention row to sum to one, so a
head with nothing worth retrieving must still put its mass somewhere. Gu et al. [6] argue
the sink token absorbs that surplus and acts "more like key biases." Our *sigmoid* arm,
which removes normalization, tests that constraint directly.

**Probe.** The function `probe_sinks()` re-walks the decoder from the live module weights in
eager mode, in fp32, with no gradients, independent of the training path, which uses fused
SDPA kernels, autocast and `torch.compile`. Every call validates its hidden states against
the real forward pass to a relative error below 10<sup>−2</sup>, so the probe cannot drift
from what the trained model computes. The probe batch is fixed across all runs and seeds, so
every number here is comparable run to run, and probes fire every 100 optimizer steps, dense
enough to timestamp each signature's first threshold crossing (§3.4). Appendix F gives the
probe-batch composition, token accounting and validation protocol.

**Recipe.** We use AdamW with weight decay 0.1, following [6], and a gradient clip of 1.0.
The schedule is cosine with 3% warmup. The learning rates are 4 × 10<sup>−4</sup> for the
language model, 2 × 10<sup>−3</sup> for the projector, and 10<sup>−4</sup> for the vision
encoder. We use bf16 autocast, `torch.compile`, and a batch size of 128. Arms in a
comparison differ only in the lever under test. We use two seeds for *baseline* and
*sigmoid*, three for *g1gate* and *textinit*, and one for *RF*.

**Validation losses, and what they do and do not license.** Training stays healthy in all
reported runs. At the matched 100M-token checkpoint the held-out losses are 1.182 for
*baseline*, 1.133 for *g1gate*, 1.206 for *sigmoid* and 0.877 for *textinit*, and RF reaches
0.638 at 1B tokens (Appendix F). Two cautions. *textinit* starts from a pretrained text
decoder, so its lower loss reflects unequal competence rather than a lever effect, and only
*baseline*, *g1gate* and *sigmoid* are equal-token, equal-initialization comparisons. The
repeated-data arms also show a large train–validation asymmetry (`val_seen` near 0.44
against `val_unseen` near 1.18), the overfitting signal that motivates RF. RF has no distinct
seen split, so the weaker statement its data supports is that its held-out loss falls
throughout and never turns upward (§5). **We did not run MMStar or any other downstream
benchmark on any arm**, so this paper makes no capability claim (§5).
