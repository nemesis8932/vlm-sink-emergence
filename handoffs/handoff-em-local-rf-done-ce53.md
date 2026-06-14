TO: Engineering Manager   FROM: local Agent   RE: RF infra complete on sink-emergence; cloud scripts ready

CONTEXT:
- All 4 deliverables built + tested. sink-emergence HEAD = 0bfc6c0.
- mac-reorg branch exists (c268bc8) with gitignore/archive cleanup — DO NOT MERGE until RF done+box stopped, per handoff.

DELIVERABLES (all on sink-emergence):

1. **Streaming loader** — `data/mixes.py:load_mix_streaming` (interleave_datasets streaming=True, shuffle buffer=10k, seed=0). `get_data` in `train_sinks.py` routes fresh full-run to `_get_data_streaming_fresh`: drains val_size=1024 BEFORE training (deterministic), `stream.skip(val_size)` for train. `data/datasets.py:IterableVQADataset` wraps stream → same schema as VQADataset (reused template, no model changes).

2. **Compute monitor** — `scripts/compute_monitor.py`. Appends to `compute_log.jsonl` every 45s: nvidia-smi util/VRAM/power/temp + latest loss/step/tok_s from train_log.jsonl + df /workspace + HF cache size + vast.ai dph_total. Writes `compute_status.txt` one-liner. Tested locally (emits valid JSONL ✓).

3. **Failsafe** — `scripts/watchdog_rf.sh`. Reads `NAN_KILL`/`VAL_KILL`/`CAP_HOURS` from env. Stops on: NaN/Inf loss (reads compute_log.jsonl), 8h cap, orchestrator absent 10 ticks. Logs but does NOT stop on: hang (util~0), disk, divergence. Calls `stop_verify.sh` on trigger. NaN kill also inline in `train_sinks.py` (immediate, doesn't wait for watchdog 60s cycle).

4. **Cloud scripts** — `scripts/run_session_rf.sh` (full RF session: exports NAN_KILL=1/VAL_KILL=0/CAP_HOURS=8, launches monitor+watchdog bg, runs locked recipe, persist+push, HF upload, verified stop). `scripts/disk_dryrun.sh` (Gate-2 pre-flight: HF token check, fake smoke, real stream smoke limit_per_subset=50 max_steps=3, cache footprint assert).

TESTS PASSED (offline):
- `tests/test_streaming_loader.py` — all 5 cases ✓ (schema, val drain disjoint, val drain deterministic, IterableVQADataset output, row count)
- `tests/test_invariants.py` — unchanged, still pass ✓
- compute_monitor.py smoke — valid JSONL emitted ✓

CLOUD DRY-RUN SEAMS (Gate-2 must close):
- `num_workers=0` in streaming train DataLoader: multi-worker shard split not tested locally (no disk-bounded eviction to test). Cloud box must verify throughput is acceptable; if bottlenecked, add per-worker IterableDataset.shard() in IterableVQADataset.__iter__.
- Real FineVision shard eviction behavior on 80GB disk (1.49TB pool): only testable with real HF pull on the box. `disk_dryrun.sh` covers this with `limit_per_subset=50`.
- `stop_verify.sh` actual-status check: must confirm stop actually takes (Session-1 lost ~$1.5 to a stop that silently no-op'd). `disk_dryrun.sh` does NOT test this — cloud agent must verify manually.

RETURN: loader offline tests pass ✓; monitor emits valid JSONL on local stub ✓; failsafe env-var contract: `NAN_KILL=1 VAL_KILL=0 CAP_HOURS=8` exported by run_session_rf.sh, watchdog reads them ✓. Cloud dry-run seams: num_workers shard split + disk eviction + stop_verify — all must be validated by cloud agent in disk_dryrun.sh gate.
