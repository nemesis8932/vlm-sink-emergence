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

*(filled at end of session — see analysis/ for figures)*

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
