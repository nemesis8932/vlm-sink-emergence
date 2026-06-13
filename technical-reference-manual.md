# Project: VLM Sink Emergence ("the emergence study")

> **Purpose**: Single source of truth for ANY agent (coding or research) entering this work cold.
> Read top to bottom and you have all required context: what the thesis is, what the data shows
> so far, what is NOT yet trusted, and the operational facts you must not violate.
>
> **Version**: 0.1 — 2026-06-13. First TRM for the emergence study. Does NOT supersede the GVLM
> TRM v2.1/2.2 (that documents the two closed theses — characterization + retrofit gate — and
> remains the historical record). This study continues the GVLM project after both closed.
> Companion docs: `HANDOFF-emergence-study.md` (origin brief), `REPORT.md` (Session-1 full record),
> GVLM `technical-reference-manual.md` v2.x (graveyard + reusable code), repo `gating-paligemma2`.

---

## 0. TL;DR — Where the Project Stands Today

1. **Thesis (live)**: run Gu et al.'s "watch the sink emerge during from-scratch pretraining"
   playbook (arXiv:2410.10781) on **multimodal** models — nobody has. Train small VLM pairs from
   scratch, checkpoint densely, map WHEN / WHERE / WHAT-drains, and test interventions as arms.
   **Distinctive hypothesis**: the "attention sink" is not one phenomenon but ≥3 separable
   signatures — attention concentration, value-norm drain, massive activation — that **decouple**.
2. **Session 1 (DONE, 2026-06-13, cloud)**: pipeline built on a from-scratch nanoVLM-222M; 4 arms
   probed in the early window (≤174M tokens). **Result: a clean 4-way dissociation** — each lever
   (softmax / text-init / G1 gate / sigmoid) produces a *different combination* of the 3
   signatures. This is the paper's spine and it is genuinely novel vs all prior art (which is
   inference-time / frozen-model only). See §2.
3. **NOT yet trusted (blocks scale-up)**: every headline rests on **n=1 seed**, **unequal token
   budgets**, and a live **undertraining confound** (baseline saw ~15× too few tokens, so "softmax
   forms no concentration sink" may just be "not yet"). Two seductive inversions — "sigmoid CREATES
   a sink in VLMs" and "the G1 gate CREATES rather than prevents one" — are leads, NOT results.
   See §7 anti-zombie.
4. **Next (Session 2, cloud, $25 cap)**: extend baseline to 1.0–1.5B tokens (kill/confirm the
   undertraining confound) + 2nd seed of all arms at matched 100M. Analysis + figures move to the
   **Mac** ($0). 1B-class A100 confirmation is GATED behind Gate B passing.
5. **Budget**: **$200 ceiling, with early-exit** (user-set, 2026-06-13; supersedes the handoff's
   $100). Spent so far ≈ **$4.5**. Strategy: small-scale validate → scale only if promising.

Primary model: **nanoVLM-222M** (from-scratch). Target: NeurIPS-2026 workshop (Aug/Sep) → ICLR
2027 full paper (~Sep 24, UNVERIFIED) / TMLR (rolling) fallback.

---

## 1. Project History (the graveyard — why emergence survives two deaths)

| Era | Thesis | Outcome |
|---|---|---|
| Jan–Jun 2026 | **Thesis 1** — retrofit G1 gates onto a *pretrained* VLM via PEFT to kill sinks/hallucination | DEAD. Gates mathematically locked at σ≈0.5 (weight std ~10× too small; o_proj LoRA absorbs any rescale; sinks are pretraining-scale circuits). Autopsy: GVLM TRM §4.4 |
| Jun 12–13 2026 | **Thesis 2** — "VLM sinks carry full-magnitude values, unlike LLMs" | DEAD. FAIL-B: a measurement artifact of *pooling* positions 0–4. Per-position decomposition showed Qwen3-VL is a textbook consensus model (sink at pos 1, values drained 0.04–0.28, input-independent). Cross-validated on two machines |
| Jun 13 2026 → | **This study** — emergence during multimodal *from-scratch* pretraining | LIVE. Survives both deaths: Thesis 1 failed in the *retrofit* regime (from-scratch was never tested — and is exactly Gu/Qiu's regime); Thesis 2's collapse proved sink properties are training-/architecture-dependent, which is itself the motivation to study *emergence* not snapshots |

**Design principle (do not violate)**: *heads-I-win-tails-I-win*. The paper exists in every outcome
branch. Gates prevent VLM sinks → original idea validated in the only regime where it could work.
Gates don't → still the first emergence map of multimodal sinks. The two dead theses each existed
in only one branch. Never design that way again.

---

## 2. Canonical Numbers — Session 1 (cite these; mind the caveats)

**Status: single seed per arm, early-window only (≤174M tokens; Gu-scale is 5B). Treat as
pipeline-validation + first emergence curves, NOT the final map.** All on nanoVLM-222M.
Source: `REPORT.md` 2026-06-13, vast.ai instance 40436103 (RTX 4090).

### Setup (identical across arms unless noted)
- **Model**: SigLIP-B/16-224 (85M, *pretrained init, trainable*) + SmolLM2-135M-architecture
  decoder (**30 layers × 9 q-heads, GQA 3 KV-heads, d=576**) + pixel-shuffle modality projector.
- **Sequence = 49 image tokens (causal prefix) + 79 text tokens (left-padded) = 128.**
  **pos0 = the first *image* token. There is NO BOS** (architectural difference from the Qwen
  setup, where the sink sat at pos 1). A sink at pos0 here lands on a content-bearing patch.
- **Data**: the_cauldron {vqav2, cocoqa, aokvqa, vsr} = 146,731 images, ~740k unique QA texts,
  1 QA/visit. bs 128 (16,384 seq-tok/step), bf16 autocast + torch.compile, ~54k seq-tok/s.
- **Opt**: AdamW wd=0.1 (Gu setting), grad-clip 1.0, cosine LR, 3% warmup. LR: LM 4e-4
  (from-scratch arms) / MP 2e-3 / ViT 1e-4.
- **Probe (every 100 steps)**: fixed 32-sample batch; eager fp32 re-walk of the decoder validated
  against the model's own forward (rel err <1e-2). bf16 ckpts at {0,250,1k,2k,4k,8k,12k,16k,24k}.
- **Token accounting**: "tokens (M)" in all tables = *text/loss tokens* (~79 of 128 positions),
  not total sequence tokens. Keep this unit consistent.

### The 4 arms

| arm | attention | LM init | ViT init | tokens (M) | wall |
|---|---|---|---|---|---|
| baseline | softmax | random | pretrained | 174.4 (18,287 steps) | 1.6 h |
| textinit | softmax | SmolLM2-135M pretrained | pretrained | 59.6 | 0.55 h |
| g1gate | softmax + G1 elementwise σ-gate (zero-init, post-SDPA) | random | pretrained | 102.9 (10,786) | 1.0 h |
| sigmoid | unnormalized sigmoid (no softmax) | random | pretrained | 101.7 (10,664) | 1.0 h |

### THE headline table — 4-way dissociation (at end-of-run; budgets differ — see caveat)

| arm | concentration (Sink^0.2₁) | value-norm @ pos0 (v_ratio) | massive activation (h_ratio) | val loss |
|---|---|---|---|---|
| baseline (softmax, scratch) | **none** (0.00; max head ~0.18) | drained (~0.71) | moderate (~2.2×) | 1.161 |
| g1gate (softmax+gate, scratch) | weak/volatile (~0.004; first head >0.2 @ step 3.4k; ≤7 heads) | mild drain (~0.82) | **suppressed** (~1.6×) | 1.138 |
| sigmoid (no softmax, scratch) | **strong/fastest** (0.83; >0 @ step 500; max head 0.87) | **amplified/anti-drain** (~1.48×) | **none** (~1.0–1.15×) | 1.108 |
| textinit (softmax, pretrained text LM) | **total** (0.85; present @ step 0) | **severe drain** (~0.38) | **extreme** (~42×) | 0.832 |

### Per-arm detail worth keeping
- **baseline**: Sink^0.2₁ = 0 throughout 174M tokens; norm signatures emerge early regardless —
  h_ratio crosses 2× by **step 400 (~4M tok)**, settles ≈2.2; v_ratio <0.8 by **step 1200**,
  settles ≈0.71–0.75. By step 10k, **80% of (layer,head) pairs have argmax key at pos0** (flat
  positional preference, no per-head mass concentration). → *norm signatures WITHOUT concentration.*
- **textinit** (the inheritance result): the text LM's sink **relocates onto the first image
  token instantly** and alignment amplifies it. step0 → end: Sink^0.2₁ 0.193→0.852; mean attn→pos0
  0.13→**0.63** (higher than Qiu's ~0.47 for text LLMs); v_ratio 0.73→0.38; h_ratio 3.5→42.5.
  At **step 0** the norm/value signatures transfer to pos0 *before* concentration does
  (Sink^0.2=0.19, Sink^0.3=0). → *inherited, relocated, amplified — never formed de novo.*
- **g1gate** (inverts the LM literature): the ONLY from-scratch arm to develop ε-threshold sink
  heads, yet its norm signatures are *milder* than baseline. Gate σ trains away from init (unlike
  the retrofit regime). → *gate decouples concentration from the norm pathology.* Qiu's "4.8%
  first-token attention" was text-pretrained at far larger scale; here the gate's effect on *where
  attention goes* is opposite within 100M tokens.
- **sigmoid** (the boldest, riskiest result): concentration emerges fastest of any arm; verified
  *not* a normalization artifact (raw unnormalized mean sigmoid score to pos0 ≈0.25 vs ≈0.05
  baseline). But norms invert: value at pos0 *amplified*, no massive activations. → confirms HALF
  of Gu (kills the residual-norm blowup), CONTRADICTS the other half (does not kill concentration).

### Compute (Session 1)
RTX 4090 @ **$0.7037/hr**. Productive 249 min ≈ 4.15 GPU-h ≈ **$2.92**. Setup/download ≈ 37 min
(~$0.44). Waste: ~3 OOM probes (bs256/512) + 1 baseline relaunch ≈ 12 min (~$0.14); **idle tail
≈ 2.2 h (~$1.5)** because `vastai stop` did not take after the backstop fired (see §7.1).

---

## 3. Current Research Plan (the emergence map)

**Thesis**: "When attention sinks emerge in vision-language models." Map, over from-scratch
multimodal pretraining: **WHEN** the sink forms (inherited from text init / re-formed in alignment
/ fresh from scratch), **WHERE** it sits and whether it MOVES, **WHAT** drains (does attention
concentration co-occur with value drain or decouple). Interventions are *arms, not theses*:
G1 gate, sigmoid attention, injection style (prefix vs DeepStack-additive), text-init vs scratch.

### Experiment spine (status as of v0.1)
- **Stage 0 — pipeline + arms, early window (DONE, Session 1)**: §2. Gate G1 result is a *fork,
  not a kill* — baseline shows no *concentration* sink at this budget, but the norm machinery is
  present, and per heads-I-win the dissociation is itself the finding.
- **Stage 1 — consolidation (Session 2, cloud, $25 cap)** — resolve what blocks any scale-up:
  - **R1 baseline EXTEND** (resume seed-0) → **1.0B** tokens (stretch 1.5B). Decides the
    undertraining confound. Record first token-count any head crosses 0.2, and Sink^0.3₁ @ 1.0B.
  - **R2 seed-2 of all 4 arms** @ matched **100M** tokens. Input to Gate B.
  - **R3 re-probe saved ckpts**: per-(layer,head) v/h distributions (not pooled); image-swap
    invariance; raw per-head sigmoid mass as primary sigmoid metric.
  - Full instructions: `CLOUD-instructions-session2.md`.
- **Stage 1b — analysis + writing (Mac, $0)**: per-head audit, image-swap verdict, raw-sigmoid
  metric, local figure regen, matched-budget 2-seed dissociation table, **pre-registration doc**.
  Full instructions: `MAC-deferred-tasks.md`.
- **Stage 2 — scaled confirmation (cloud, A100-80GB, ~$45–70)**: ONE ~1B-class run, ~5B tokens,
  Gu-style. **Only if Gate B passes** (arms separate across seeds at matched budget) AND the
  baseline confound is resolved.
- **Deferred arms** (not yet run): injection style A3 prefix vs **A4 DeepStack-additive**
  (arXiv:2406.04334); full from-scratch with **random ViT** (Session 1 used pretrained ViT in all
  arms).

### Pre-registered gates / decision thresholds
- **Gate A (baseline forms a sink)**: Sink^0.3₁ ≥ ~5%. Session 1 → concentration fired NO, norm
  signatures fired YES → **FORK**: resolve via R1 before claiming absence; report the soft-sink
  either way.
- **Gate B (arms separate)**: pre-registered as **>2σ across seeds**. **Not passable on n=1.**
  Honest limit: n=2 cannot give a true 2σ — practical workshop bar = tight seed agreement + large
  effect. sigmoid (0.83 vs 0.00) expected to survive; **g1gate (0.004, volatile) is the fragile
  arm** — needs seed-3 if seeds disagree.
- **Branch outcomes** (all reportable): baseline sink stays absent at 1B + dissociation holds →
  green-light Stage 2. Baseline sink arrives late → pivot thesis to *emergence timing*
  (softmax-late / sigmoid-early / inherited-instant) — arguably stronger. Budget overrun → ship
  the 222M sweep as a focused workshop empirical paper, drop Stage 2.

### Scoop / novelty (as of 2026-06-13)
LOW–MEDIUM, *strengthened* by Session 1. All prior VLM sink work is inference-time / frozen
(arXiv:2510.08510 ViT-side; 2604.03316 V/L-sink taxonomy + LSG; 2503.03321 training-free VAR).
Gu's from-scratch-emergence method has been applied to text + diffusion LMs, **never multimodal**.
The dissociation, the two inversions, and the step-0 inherited-relocation observation are all
orthogonal to that prior art. Residual MEDIUM risk = fast-moving 2026 literature → **pre-register
to date-stamp priority** (Mac task M6).

---

## 4. Methodology Anchors (read before coding)

- **Gu et al. 2410.10781** (ICLR-25 Spotlight) — THE template. Sink metric Sink^ε₁ (ε-threshold,
  fix T e.g. 64 for fair comparison; ε=0.3 default, robustness {0.2,0.4}); training-dynamics
  methodology; **sigmoid-attention-without-normalization prevents sinks + massive activations ≤1B
  in text LMs**; "sink position follows the data distribution"; sinks act as **key biases** (store
  attention, contribute little value); instruction tuning barely moves sinks → it's a *pretraining*
  phenomenon. Mirror their figures for reviewer recognition.
- **Qiu et al. 2505.06708** (`gated_attention` GitHub; NeurIPS-25 Oral) — G1 gate spec:
  `Y' = Y_sdpa ⊙ σ(XW_θ)`, head-specific multiplicative sigmoid gate on the SDPA output. **Correct
  numbers**: first-token attention avg **46.7% → 4.8%** across layers (NOT "→~0"); layer-21 sink
  83% → 4%. **No special init for the main G1 gate** (only an input-independent *control* variant
  is zero-init — do not attribute a σ≈0.5 bias-init to Qiu; that is the later 2601.15305). Gate
  trains under the global LR; a benefit is tolerance of *larger* LRs.
- **Sun et al. 2402.17762** — massive activations; zero-vs-mean intervention design.
- **Active-Dormant 2410.13835** — value-norm-vs-position plot style (our Figure-1 template);
  mutual-reinforcement of extreme tokens.
- **DeepStack 2406.04334** — arm A4 injection design (additive, visual positions, early decoder
  layers). Mechanics verified in GVLM TRM §3.
- **Yoo et al. 2603.14337 (VERIFY)** — sink value DIRECTION matters even at small norm → measure
  directional alignment over training, not just norms.
- **VLM sink taxonomy / positioning**: 2604.03316 (V/L-sinks), 2510.08510 (ViT-side), 2503.03321
  (visual sink, VAR).

### Code that transfers (repo `gating-paligemma2`)
- `models/modeling_gated_qwen3_vl.py` — working G1 widened-q_proj + the 0.5×-at-init equivalence
  test (`scripts/verify_qwen3_port.py`). Port the *pattern* to nanoVLM; ALWAYS keep the test.
- `scripts/t0_pos_resolved.py` (Windows) — validated per-position/per-layer/per-KV-head harness;
  becomes the checkpoint-measurement module. (Its pooled-window ρ_sink is DEPRECATED — per-position
  only; pooling caused FAIL-B.)
- Eval: `evaluation/pope_eval.py`, CHAIR harness, AMBER (local clone). `training/` SFT trainer
  with grad-checkpointing + label-mask fixes.

---

## 5. Hardware, Budget, Logistics

| Resource | Use | Notes |
|---|---|---|
| **Cloud (vast.ai)** — billed **$0.7037/hr** for the Session-1 4090 | Training workhorse for THIS study (user's choice; nothing local yet) | Instance **40436103** (stop, never destroy; rsync before reuse). A100-80GB for Stage 2 confirmation |
| **MacBook M4 Max 24GB** | Analysis, figures, paper, pre-reg — all $0 | Pure-tensor work over saved probes; MLX optional. Do NOT pay $0.70/hr to plot |
| Desktop RTX 4070 Super 12GB | NOT yet used this study | Available as expendable from-scratch workhorse; reserved for later |
| **Budget** | **$200 ceiling, early-exit** | Spent ≈$4.5. Pre-register before any cloud Stage-2 spend |

**Timeline**: Jun 15–Jul 15 build + main sweep. Jul 15–Aug 15 analysis + one scaled confirmation.
Aug: workshop submission. Sep: ICLR paper. (TMLR negative-result paper is a separate ~90%-written
track — do NOT let this study cannibalize it.) Deadlines UNVERIFIED — confirm ICLR 2027 (~Sep 24)
and the NeurIPS-2026 workshop list when posted.

---

## 6. Repository Map (`nemesis8932/vlm-sink-emergence`, branch `sink-emergence`)

> Fork of nanoVLM v0.1. **Internals below are inferred from REPORT.md + the nanoVLM layout — an
> agent with repo access should correct this section.** Username note: HF artifacts live under
> `nemesismaniac`; this GitHub handle is `nemesis8932` — confirm both are intended.

```
models/                  # nanoVLM: vision_transformer.py, language_model.py, modality_projection.py
  (+ gated/sigmoid variants ported from gating-paligemma2 — keep the 0.5×-at-init test)
training/                # from-scratch train loop, per-arm config, dense ckpt {0,250,1k,2k,4k,8k,12k,16k,24k}
analysis/                # figures (move plotting HERE on the Mac, not cloud)
runs/                    # per-arm outputs
  <arm>/probes.jsonl     # per-(layer,head) probe records every 100 steps
  <arm>/reprobe/         # Session-2 per-head v/h, image-swap, raw-sigmoid dumps
plan_when-sinks-emerge-in-vlms.md   # the staged plan REPORT.md executes
REPORT.md                # Session-1 full record
```

### Artifacts (do not lose)
- **HF Hub**: `nemesismaniac/gated-qwen3vl-artifacts` (or a new emergence-specific repo) — push
  ckpts + probes.jsonl after EVERY run.
- **vast.ai instance 40436103** (stopped, disk intact): Session-1 `runs/`, probes, ckpts.
  `vastai start` to access; **rsync after every stage; stop, never destroy.**

---

## 7. Operational Facts / Hard-Won Rules (violating these wastes money or kills a finding)

**Measurement discipline (carried from GVLM; FAIL-B was born from breaking #1):**
1. **LOCATE the sink before measuring it** — per-position, per-layer, per-KV-head. **No pooled
   first-k window metrics, ever.** Here pos0 = first image token; verify the sink actually sits
   there and isn't smeared.
2. **Direct ‖v‖ only.** The α·‖v‖ proxy gave irreconcilable cross-machine numbers.
3. **Image-swap invariance check** on every new harness/model. Here pos0 is content-bearing, so it
   SHOULD vary across images — if it doesn't, the harness or the "sink" is an artifact.
4. **Per-KV-head always.** Head-averaging can fake or hide anything (one anti-drained head among
   nine fakes the sigmoid "amplification"). Confirm v_ratio/h_ratio are per-head, not pooled.
5. **Identical templates/prompts/seeds across compared arms.** The inheritance/dissociation
   findings are only valid because data/opt/arch were matched. Session 1 violated *budget* matching
   (174/60/103/102M) — fix in Session 2 (matched 100M).
6. **0.5×-at-init equivalence test** before any gated training run.

**Anti-zombie guard (this study's candidate false headlines — reject on sight until earned):**
7. NOT YET TRUE, do not headline:
   - "from-scratch softmax VLM forms no concentration sink" — likely **undertraining** (baseline
     saw ~15× too few tokens). Resolve via R1 (extend to 1B) first.
   - "sigmoid CREATES a sink in VLMs" — n=1, metric measured via re-normalized scores; needs raw
     per-head mass + 2 seeds.
   - "G1 gate CREATES rather than prevents sinks" — n=1, weak/volatile signal; needs ≥2 seeds.
   All three are *leads*. Earn them with matched budget + ≥2 seeds + the raw-mass metric.

**Cloud ops (Session-1 cost lessons):**
8. **Auto-stop failed once (2.2h idle, ~$1.5).** Use redundant stop: in-script `vastai stop` on
   completion AND an in-instance `shutdown -h +<minutes>` hard backstop. Verify it actually stopped.
9. **Pricing is ~$0.70/hr, NOT the $0.31 in the original plan.** Budget against $0.70 (~280 GPU-hr
   in $200).
10. **tmux for long runs; `WANDB_MODE=offline`; push ckpts to HF after every run; rsync after every
    stage; stop instances, never destroy.**
11. **Move analysis/plots to the Mac.** Re-probing saved ckpts is seconds of GPU; matplotlib is $0
    on the M4. Don't burn cloud hours on figures.

**Process:**
12. **Evaluate AGAINST the hypothesis. Pre-register gates before runs.** No salvage proposals
    inside a results report — decisions happen at the director level, not in the data summary.

---

## 8. Metric Definitions (as used HERE)

- **pos0** — the FIRST IMAGE token (no BOS in this architecture). All sink metrics target pos0
  unless a per-position scan says otherwise. Always report the argmax-key-location split
  (pos0 / image-1..48 / text) alongside, to confirm pos0 is really where mass lands.
- **Sink^ε₁** — fraction of (layer, head) pairs whose *mean* attention to pos0 exceeds ε, at fixed
  sequence length. ε ∈ {0.2, 0.3, 0.4}; 0.3 is the Gu default. **For the sigmoid arm**, scores are
  row-normalized to compute this — but the PRIMARY sigmoid evidence is **raw per-head sigmoid mass
  to pos0** (un-normalized), since row-normalization can inflate apparent concentration when total
  mass shrinks.
- **mean attn→pos0** — average attention weight on pos0 across heads/queries; the un-thresholded
  companion to Sink^ε₁.
- **v_ratio** — ‖v(pos0)‖₂ / ‖v(rest)‖₂, per (layer, KV-head). <1 = value drain (classic sink);
  >1 = anti-drain (sigmoid arm). **Direct norms only**, never the α·‖v‖ proxy. Report the per-head
  distribution — never average heads before inspecting it.
- **h_ratio** — residual-stream norm at pos0 / at the rest. ≫1 = massive activation. Distinct from
  value norm (they co-occur with OPPOSITE signs — keep them separate).
- **val loss** — next-text-token CE on held-out cauldron QA. Cross-arm comparisons are confounded
  by competence (textinit starts pretrained → far lower loss); use only for arm-internal trends.
- **Three signatures** (the dissociation axes): (1) attention concentration [Sink^ε₁ / mean attn],
  (2) value-norm drain [v_ratio], (3) massive activation [h_ratio]. The thesis = these are
  independently controllable, not facets of one phenomenon.

---

## 9. Adjacent Docs (read on demand; not duplicated here)

- `HANDOFF-emergence-study.md` — origin brief (thesis, graveyard, assets F1–F4, design skeleton).
- `REPORT.md` — Session-1 full record (setup, per-arm curves, compute ledger, caveats).
- `CLOUD-instructions-session2.md` — next cloud session (R1 extend, R2 seeds, R3 re-probe, ops).
- `MAC-deferred-tasks.md` — M1–M6 analysis + pre-registration.
- GVLM `technical-reference-manual.md` v2.x — the two closed theses, the gate autopsy (§4), and
  the reusable harness/eval/gate code described in §4 here.
- `RESCUE_SCENARIO.md` — parked retrofit-gate hypotheses (Thesis 1).

### Open items before Stage 2
- [ ] R1: baseline first-crossing token, or "none at 1.0B".
- [ ] R2/M5: matched-budget dissociation table with per-seed ranges.
- [ ] M2: image-swap verdict (pos0 varies with image? expected yes).
- [ ] M3: raw sigmoid-mass verdict (concentration real, not normalization).
- [ ] M6: pre-registration live + timestamped.
- [ ] Verify ICLR 2027 / NeurIPS-2026 workshop deadlines.
