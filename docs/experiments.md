# Experiments

## Arms (4 arms × init)

One process = one arm (`train_sinks.py --arm`). Levers isolate which sink signature each
produces — the four-way dissociation is the headline result.

| arm | attention | LM init | ViT init |
|---|---|---|---|
| baseline | softmax | random | pretrained |
| g1gate | softmax + elementwise σ-gate (zero-init, post-SDPA; Qiu G1) | random | pretrained |
| sigmoid | unnormalized sigmoid, no softmax (Gu et al.) | random | pretrained |
| textinit | softmax | pretrained SmolLM2-135M | pretrained |

`--vit_init random` → fully-from-scratch variant of any arm.

Knobs: `models/config.py` (`lm_attn_gate`, `lm_attn_impl`). Attention impl:
`models/language_model.py` → `LanguageModelGroupedQueryAttention`.

Sequence = 49 image tokens (causal prefix) + 79 text (left-padded) = 128.
**Position 0 is the first image token — there is no BOS.**

## Training recipe (`train_sinks.py`)

AdamW wd=0.1 (Gu et al.), grad-clip 1.0, cosine LR w/ 3% warmup: LM 4e-4 / MP 2e-3 /
ViT 1e-4. bf16 autocast + `--compile`. Probe every 100 steps; bf16 ckpts at `CKPT_STEPS`.
Resume loads weights only (Adam restarts cold → brief loss transient).

## Run

```bash
# local Mac: source venv first. cloud: source /workspace/venv/bin/activate
python train_sinks.py --arm baseline --batch_size 128 --compile --max_tokens_M 100 \
  --out_dir runs/baseline --ckpt_steps 0,250,1000,2000,4000,6000
python reprobe.py --arm baseline --run_dir runs/baseline --ckpts 0,250,1000
python analyze_sinks.py        # figures → analysis/
```

## Hardware / VRAM

bs≤16 fits ~8GB at seq 128; bs128 needs a 24–48GB cloud GPU. Run `measure_vram.py` before
committing cloud spend. `--compile`/bf16 are CUDA-only; on Mac use MPS/CPU + tiny
`--max_steps` for smoke tests — real runs go to vast.ai (see `docs/cloud-agent.md`).
