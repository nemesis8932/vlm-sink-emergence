# Conventions & key files

## Metric conventions (Gu et al. 2410.10781, fixed T)

`Sink^ε_1` = fraction of (layer, head) pairs whose mean attention→pos0 > ε. Default ε=0.3,
fixed T; robustness at {0.2, 0.4}. The sigmoid arm reports the **row-normalized** attention
view so it's comparable across arms (raw sigmoid mass is also logged).

Track three signatures **separately** — their decoupling is the contribution:
- concentration: `Sink^ε_1`, mean/max attn→pos0
- value-norm ratio: ‖v‖ at pos0 vs rest
- massive activation: residual-stream norm at pos0 vs rest

## Probe contract (`sink_probe.py`)

`probe_sinks()` re-walks the decoder eager/fp32/no_grad from module weights — never touches
the training path (SDPA, autocast, compile). Every call **validates its hidden states
against the model's real forward** (asserts rel-err < 1e-2); the mirror cannot silently
drift. The probe batch is fixed (`random.seed(0)`) → byte-identical across seeds/runs, so
metrics are comparable seed-to-seed.

Per (layer, head): mean attn→pos0, img/text mass, argmax key position, ‖v‖ pos0 vs rest,
residual norm pos0 vs rest. Summary: `Sink^ε_1` for ε∈{0.2,0.3,0.4}, v-ratio, h-ratio.

## Key files

- `train_sinks.py` — from-scratch training + dense probing (recipe in `docs/experiments.md`).
- `sink_probe.py` — `probe_sinks()`, the validated probe above.
- `reprobe.py` — re-walk saved checkpoints for full per-head/per-position detail → `reprobe/`.
- `analyze_sinks.py` — emergence/loss/heatmap/location figures → `analysis/`.
- `gate_summary.py` — machine-readable acceptance-gate summary (prose REPORT is human-written).
- `measure_vram.py` — VRAM probe; run before cloud spend.
- `run_session2.sh`, `watchdog.sh`, `stop_verify.sh` — cloud orchestration (`docs/cloud-agent.md`).
- `runs/<arm>/` — `probes.jsonl`, `train_log.jsonl`, `run_config.json`, `reprobe/`.
- `models/config.py` — `VLMConfig` incl. sink knobs; `models/language_model.py` — attention.
