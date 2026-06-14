TO: Engineering Manager   FROM: Director   RE: build + launch RF (random-fresh 1B Gate-A retest), overnight

CONTEXT (only what you lack — rest in repo):
- Decisions locked this session: `CONTEXT.md` (RF/Comparator/Gate A terms) + `docs/adr/0001-fresh-vs-repeated-accepts-domain-shift.md`. Background: `archive/session3/rf_blocked.md`, `references/plan_when-sinks-emerge-in-vlms.md`.
- Scope = RF ONLY: one arm (`--arm baseline`, pretrained ViT, random LM), `--data_mode fresh`, 1B tok, single seed. No sigmoid/g1gate/textinit, no seed-3 reruns tonight.

ASK (build, then launch):
1. **Streaming loader** (the only new code; last session's blocker). `load_mix_streaming` via HF `interleave_datasets(streaming=True)` over `mixes.FRESH`, `.shuffle(buffer_size~10k, seed)`, reuse existing VQACollator/template (data-only difference — do NOT touch model-facing template). Add fresh-mode branch to `get_data` in `train_sinks.py`: `IterableDataset` has NO random access, so current `rep.select(range(...))` / `VQADataset[i]` path breaks — fresh val = deterministic first-N drained off the stream BEFORE training consumes it (`--val_size 1024`).
2. **Disk**: keep the 80GB instance. Async shard-eviction thread on idle CPU, evict only shards N behind the read cursor (over-evict → re-download, not crash). Confirm cache footprint stays bounded with a short streamed dry-run before the real launch.
3. **Recipe = byte-identical to the Comparator.** FIRST locate the ~855M–1B *repeated* Gate-A run's `run_config.json` on HF (`nemesismaniac/vlm-sink-emergence-ckpts`) and clone its recipe + `max_steps` (≈105k for 1B @ ~9.5k tok/step) so cosine LR spans the full budget. **If that config can't be found, STOP and report — RF is not a clean swap without it.**
4. **Probing**: every 100 to step ~5k then every 500. bf16 ckpts at 0,250,1k,2k,4k,8k,16k,32k,64k,105k → HF.
5. **Overnight policy**: checkpoint-only, NO divergence/val auto-kill (Director judges in AM). KEEP crash/NaN/idle-tail backstop (waste rule, `docs/cloud-agent.md`) — last session lost ~$1.5 to a backstop that didn't fire; verify `vastai stop` actually takes effect.

CONSTRAINTS: vast.ai billed GPU; ~5–6h GPU expected; self-stop on waste; commit raw results (`probes.jsonl`, `train_log.jsonl`, `run_config.json`) to git after the run survives. Director oversees every step — surface go/no-go points, don't auto-advance scope.

RETURN (compacted, no raw logs):
- Pre-launch: confirmation the streamed dry-run holds disk bounded + comparator config found (or STOP).
- Post-run: Sink^0.3_1 trajectory + final, fresh val_loss trajectory (overfit y/n), 3-signature summary at 1B, HF paths. Gate-A verdict: Sink^0.3_1 < ~5% AND val healthy = clean CONFIRM.
