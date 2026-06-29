TO: Engineering Manager   FROM: Director   RE: Stage-2 infra hardening — foolproof loader, monitoring, babysit

**STATUS: DEFERRED — do NOT action until (a) Gate B passes at n=3 AND (b) user has acquired a new instance.** This is the ready-to-pull spec for scaled fresh runs; it is not live work. The immediate work (Session-4 seed-3 + exit) is `handoff-em-session4-exit.md`. Exception already pulled forward: `generate.py` is built now for the exit quality-check (step 5 below / exit-runbook step 1).

CONTEXT (rest in repo — don't restate):
- Decisions locked this session: `docs/adr/0002-litdata-streaming-loader.md`, `docs/exit-runbook.md`, `CONTEXT.md` (fresh-shards, cloud-agent role). Loader leak post-mortem: `data/mixes.py:151-154` (decoded images in shuffle buffer) + git `af661d2`/`c5efca7`/`3fed3d7` (12→8 worker creep).
- This is Stage-2 prep. The pending **seed-3 g1gate/textinit reruns are 60M-tok repeated/map-style — NO streaming, NO leak — run them on the existing path in parallel; do not block them on the loader rebuild.**

ASK — route to local-agent (zero-cost first) then cloud-agent:

1. **LEAK REPRO (local-agent, HARD GATE before any spend).** RSS-vs-time profile of the current HF streaming loader on a small subset → attribute creep: HF-internal-state vs glibc fragmentation. Then prove the **litdata** path stays flat on the same subset. No billed box until the leak is provably dead locally.

2. **FOOLPROOF LOADER (local-agent build).** litdata StreamingDataset, capped local cache ~20GB. Convert FineVision→litdata **off-box** (user device or cheap CPU instance), push shards to **HF** (NOT GDrive — no range support; GDrive = cold mirror). Allocator → **jemalloc/tcmalloc** via `LD_PRELOAD`. Per `docs/adr/0002`.

3. **MONITORING (W&B).** Wire `train_sinks.py` to log train/val loss, **steps/s**, 3 sink signatures per probe; W&B auto-captures gpu%/vram/RAM/disk. Add a **W&B alert at system-RAM > 90GB** (the leak canary — ping before OOM). Needs `WANDB_API_KEY` on the box.

4. **BABYSIT LOOP (cloud-agent self-loops).** Tightening interval 2→5→15 min. Graduate 2→5 only when held 3 ticks: **RAM slope flat (Δ<~0.5GB/10min) AND gpu-util≥90% AND steps/s within ~10% of dry-run target**; re-tighten to 2 min if RAM slope returns. Per-tick: RAM(abs+slope), VRAM, gpu%, steps/s, disk, loss finite, instance alive → breach = alert + drop to 2 min. W&B alerts = independent backstop.

5. **EXIT (cloud-agent, ordered — `docs/exit-runbook.md`).** Build+commit `generate.py` → quality spot-check ALL arms (~5 held-out imgs each, dump to md, user picks keepers; RF baseline kept regardless) → `save_pretrained` export keepers to HF (storage-disciplined; public repo if quota; GDrive cold mirror) → box-only sweep → **verify load on another machine** → `vastai stop` → **independently confirm actual_status=stopped**. **STOP THERE. Agents NEVER destroy — the user destroys the instance manually.** Hand back with stop confirmed + backups verified.

CONSTRAINTS: HF storage is FINITE — keep only dense-early (emergence repro) + final (inference) ckpts per kept arm; mirror bulk to GDrive. vast.ai billed; self-stop on waste. Director holds go/no-go. **Destroy is user-only — no agent destroys the box.**

RETURN (compacted):
- Pre-spend: local-agent leak-attribution + litdata-flat proof (or STOP). 
- Loader: dry-run RSS-flat + steps/s on the new GPU, max stable workers.
- Exit: arm quality-check md for user's keep decision; confirmation each kept model loads via from_pretrained on a clean machine; instance destroyed + final compute log.
