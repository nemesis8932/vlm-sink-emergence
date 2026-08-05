# Exit runbook — vast.ai instance teardown (migrating to a new GPU)

We are leaving the current vast.ai box for a more training-friendly GPU (more usable VRAM).
Goal: lose nothing, and keep the trained nanoVLM(s) **loadable + runnable** forever.
Stop is the LAST agent step — nothing is stopped until backups are verified on another machine.

Authority: cloud-agent executes up to **stop only**. **Agents NEVER destroy the instance —
the user destroys it manually.** Go/no-go stays with the Director/user.

## Ordered steps

1. **Build + commit `generate.py`** (local-agent, zero-cost). Minimal: load a ckpt
   (raw bf16 state-dict via VLMConfig from `run_config.json`, or `from_pretrained` after
   export) + generate on (image, prompt). Double duty — quality check now, "play with it" later.

2. **Quality spot-check all arms** (cloud-agent, on-box while alive). Run `generate.py` on
   ~5 fixed held-out images per arm's final ckpt; dump outputs to a markdown. User judges which
   arms are worth keeping. **RF baseline is kept regardless** (the headline run); others kept
   only if outputs are meaningful (most are weak from-scratch baselines).

3. **Export keepers** to HF-loadable format. `save_pretrained()` → `config.json` + safetensors
   → push to HF so `from_pretrained('repo')` just works. **HF storage is finite** — export only
   keepers, only the final ckpt per kept arm; use a **public repo** if quota is hit; mirror the
   bulk to the user's ~2TB **GDrive** as cold backup.

4. **Box-only sweep.** Enumerate anything that exists ONLY on the box — run scripts, watchdog,
   wandb local dir, any uncommitted loader/config edits, logs/configs not yet pushed. Code → git;
   data/ckpts → HF. Cross-check `git status` is clean and HF has every referenced artifact.

5. **Verify before destroy.** On a DIFFERENT machine: fresh clone, `from_pretrained` (or
   `generate.py`) smoke-test that each kept model loads and generates; confirm RF
   `runs/rf_fresh_baseline/` data is fully on git + HF. No green here → do not proceed.

6. **Stop → confirm. DO NOT destroy.** Only after step-5 backups are confirmed: `vastai stop`,
   then **independently query `actual_status=stopped`** (Session-1 lost ~$1.5 to a silent stop
   no-op; never trust "stopping now"). **Agents stop here.** Destroying the instance is the
   **user's** manual action — hand back with stop confirmed and backups verified.

## Storage discipline

Keep: dense-early ckpts (emergence reproduction) + final ckpt (inference) per kept arm. Prune
redundant intermediates. Bulk/cold → GDrive mirror; HF holds the minimal loadable set.
