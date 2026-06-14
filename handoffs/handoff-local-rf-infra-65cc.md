TO: local Agent   FROM: Engineering Manager   RE: build RF infra OFF the billed clock (loader + monitor + failsafe), pre-launch

CONTEXT (rest in repo — Director brief `handoffs/handoff-em-rf-baseline.md`, `docs/adr/0002-*`, `docs/cloud-agent.md`):
- All four deliverables are zero-cost local. They MUST exist + be unit-tested before the cloud box launches RF — the monitor/failsafe protect that run; the loader gates it.
- Recipe is LOCKED (ADR-0002): do NOT touch `--batch_size`/`grad_accum`/LR. Compute-bound at 100%; VRAM slack is unrecoverable, not a lever.

ASK (priority order):
1. **Streaming loader** (critical path, the only new model-irrelevant code). `load_mix_streaming` via HF `interleave_datasets(streaming=True)` over `mixes.FRESH`, `.shuffle(buffer_size~10k, seed)`, reuse existing VQACollator/template (data-only — do NOT touch model-facing template). Add fresh-mode branch to `get_data` in `train_sinks.py`: `IterableDataset` has NO random access → fresh val = deterministic first-N drained off the stream BEFORE training consumes it (`--val_size 1024`). Unit-test parse/schema offline (mock or tiny real stream); you CANNOT faithfully test disk-bounded eviction locally → that's the cloud's dry-run, leave a clean seam.
2. **Compute monitor** → append `compute_log.jsonl` every ~30–60s: nvidia-smi util%/VRAM/power/temp; tok_s+loss+step parsed from train log; `df /workspace` + shard-cache footprint; cost via `vastai show instance <id>` `dph_total`. Consumers = failsafe (reads it) + EM (tails for report) + human (a terse rolling `compute_status.txt` / stdout line — latest util/VRAM/tok_s/disk/$ at a glance, no jsonl-parsing needed). No off-box streaming.
3. **Failsafe profile** = a few env vars `run_session` exports; watchdog reads them. RF profile: `NAN_KILL=1 VAL_KILL=0 CAP_HOURS=8`. Conditions that STOP: crash/orchestrator-gone (existing), NaN/Inf loss, 8h hard cap. Do NOT auto-kill on hang(util~0)/disk/divergence — monitor LOGS these, Director judges. Keep `stop_verify.sh` verified-stop.
4. **Bloat cleanup** on branch `mac-reorg` ONLY (merge AFTER RF done+box stopped): gitignore+untrack `hf_upload.log` (466KB), archive stock `nanoVLM.ipynb`/upstream `README.md`, drop dead `*.log`/`*.out`. NO structural moves / import rewrites this round (campaign arms still pending). Use `improve-codebase-architecture` for the future structural plan, not now.

CONSTRAINTS: zero GPU spend; everything local. Reorg strictly on `mac-reorg`; loader/monitor/failsafe land on `sink-emergence` (the branch cloud pulls). If cloud hits a mid-run bug, that hotfix is P0 and preempts reorg.

RETURN: confirmation loader passes offline parse-test + fresh-val drain works; monitor emits valid jsonl on a local stub; failsafe env-var contract documented for `run_session`. Flag any seam the cloud dry-run must close.
