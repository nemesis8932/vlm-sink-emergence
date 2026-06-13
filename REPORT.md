# When Do Attention Sinks Emerge in VLM Pretraining? — Session 1 (Stage-1 validation)

**Date:** 2026-06-12/13 · **Hardware:** 1× RTX 4090 48GB (vast.ai instance 40436103) ·
**Repo:** `nemesis8932/vlm-sink-emergence`, branch `sink-emergence` (fork of nanoVLM v0.1)

Executes Stage 1 of `plan_when-sinks-emerge-in-vlms.md`: build and validate the
sink-metric pipeline on from-scratch nanoVLM-222M training, with dense probing,
and collect first emergence curves for 4 arms.

## Setup

- **Model:** nanoVLM 222M = SigLIP-B/16-224 (85M, *pretrained init, trainable*) +
  SmolLM2-135M-architecture decoder (30 layers × 9 heads, GQA 3 kv-heads, d=576) +
  pixel-shuffle MP. Sequence = **49 image tokens (causal prefix) + 79 text tokens
  (left-padded) = 128**. Position 0 is the first *image* token — there is no BOS.
- **Data:** the_cauldron {vqav2, cocoqa, aokvqa, vsr} = 146,731 images; one QA pair
  sampled per image visit (≈740k unique QA texts). bs 128 (16,384 tok/step),
  bf16 autocast + torch.compile, ~54k tok/s.
- **Optimization:** AdamW wd=0.1 (Gu et al. setting), grad-clip 1.0, cosine LR with
  3% warmup: LM 4e-4 (from-scratch arms) / MP 2e-3 / ViT 1e-4.
- **Probe (every 100 steps):** fixed 32-sample batch; eager fp32 re-walk of the
  decoder, **validated against the model's own forward each call** (rel. err < 1e-2
  on valid positions). Metrics per (layer, head): mean attention→pos0 (row-normalized
  for the sigmoid arm), Sink^ε_1 for ε∈{0.2,0.3,0.4} (Gu et al.), attention mass by
  segment (pos0 / image / text), argmax key position, value-vector ℓ2 norm at pos0
  vs rest, residual-stream norm at pos0 vs rest. bf16 checkpoints at
  {0, 250, 1k, 2k, 4k, 8k, 12k, 16k, 24k}.

## Arms

| arm | attention | LM init | ViT init | wall budget |
|---|---|---|---|---|
| baseline | softmax | random | pretrained | 1.6 h |
| textinit | softmax | SmolLM2-135M pretrained | pretrained | 0.55 h |
| g1gate | softmax + G1 elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | 1.0 h |
| sigmoid | unnormalized sigmoid (no softmax) | random | pretrained | 1.0 h |

## Results

*(figures in `analysis/`; full per-(layer,head) data in `runs/*/probes.jsonl`)*

### Headline: the VLM position-0 sink is inherited and amplified, not formed de novo

**From-scratch baseline (174.4M tokens, 18,287 steps):** the attention-concentration
sink **never fires** — Sink^0.2_1 = 0 throughout; the strongest single head reaches
only ~0.18 mean attention→pos0 (L1H1, plateauing from ~step 5k). Yet the *other* sink
signatures emerge early and persist: residual-norm ratio at pos0 crosses 2× by step
400 (~4M tokens) and stabilizes ≈2.2; value-norm ratio falls below 0.8 by step 1200
and settles ≈0.71–0.75. By step 10k, **80% of (layer,head) pairs have their argmax
attention key at position 0** (none in the text segment) — a broad, *flat* positional
preference without per-head mass concentration. Plan gate A (Sink^0.3 < 5% at this
budget) technically fires for concentration — but the norm/value machinery is
demonstrably in place, so "no sink" would be the wrong reading; the VLM-regime sink
at this scale is *soft*.

**Text-init (pretrained SmolLM2 decoder, 59.6M tokens):** the text LM's sink
relocates to the first **image** token *instantly* and alignment training amplifies
it explosively:

| step | Sink^0.2_1 | mean attn→pos0 | v_ratio | h_ratio |
|---|---|---|---|---|
| 0 | 0.193 | 0.13 | 0.73 | 3.5 |
| 100 | 0.552 | 0.25 | 0.48 | 10.5 |
| 500 | 0.830 | 0.39 | 0.34 | 20.7 |
| 1000 | 0.837 | 0.53 | 0.38 | 30.6 |
| 6244 (end) | 0.852 | **0.63** | 0.38 | 42.5 |

Mean attention to pos0 saturates at ~0.63 — *higher* than the ~0.47 Qiu et al. report
for text LLMs — on a token that is a SigLIP patch embedding, not BOS. The same
training data, architecture, and optimizer that produce **zero** concentration sink
from random init produce a near-total one from text-pretrained init.

### Decoupling (the plan's distinctive hypothesis) — supported in both directions

- From-scratch: massive-activation (h_ratio>2 @ step 400) and value-drain
  (v_ratio<0.8 @ step 1200) emerge **without** attention concentration ever crossing
  ε=0.2 (174M tokens).
- Notably v_ratio and max_a0 move in *opposite* directions in mid-training
  (steps 2k→4k: v_ratio 0.71→0.79 while max_a0 0.12→0.16).
- Text-init step 0: norm signatures (h_ratio 3.5, v_ratio 0.73) transfer to pos0
  ahead of concentration (Sink^0.2 = 0.19, Sink^0.3 = 0 at step 0; 0.83/0.75 by step 500).

### Intervention arms

*(g1gate / sigmoid filled below when runs complete)*

### Step-0 sanity observations (already informative)

- **baseline/g1gate (random init):** mean attn→pos0 ≈ 0.053 ≈ uniform-causal
  expectation; Sink^0.2 = 0; v_ratio ≈ 1.0. No sink at init, as expected.
- **textinit (pretrained SmolLM2 decoder, image prefix):** at step 0 the *text* LM's
  sink machinery partially transfers to the multimodal sequence: residual-stream norm
  at pos0 is **3.2×** the rest (massive-activation transfer to an *image* token), and
  value-norm ratio at pos0 is **0.64** (the value-drain signature), yet max attn→pos0
  is only 0.17 (Sink^0.2_1 = 0) — i.e. **the norm/value signatures transfer to
  position 0 before the attention-concentration signature does**. Within a handful of
  optimization steps attn→pos0 begins rising. This is a step-0 decoupling data point.

## Caveats

- Single seed per arm; 222M scale; ≤0.5B tokens per arm (Gu-scale is 5B) — this
  session is the *pipeline-validation + early-window* stage, not the full map.
- Data repeats images across epochs (~13 visual epochs at baseline length); text
  targets resampled per visit.
- ViT is pretrained (not random) in all arms this session; "full from-scratch"
  (random ViT) is left for the next stage.
- MMStar accuracy not measured (GPU budget went to emergence curves).
