# CLAUDE.md

Core guidance for any agent in this repo. Be terse; sacrifice grammar for concision.

## What this is

Research, not a product. Question: **when/where do attention sinks emerge during VLM
pretraining, and do attention-concentration vs value-norm / massive-activation signatures
*decouple*?** Fork of nanoVLM v0.1 (222M = SigLIP-B/16 + SmolLM2-135M-arch decoder).
**Ignore the global CLAUDE.md's PaliGemma description — that is a different project.**

## Read first

- `references/plan_when-sinks-emerge-in-vlms.md` — the plan, scoop check, deadlines.
- `REPORT.md` — findings so far. Read before proposing experiments; don't re-derive.
- `docs/experiments.md` — arms, knobs, how to run.
- `docs/conventions.md` — metric definitions, probe contract, key files.
- `docs/cloud-agent.md` — north star + rules for the vast.ai cloud agent.

(The stock nanoVLM `README.md` / `nanoVLM.ipynb` are upstream docs.)

## Workflow (mandatory)

- **`git pull` before changing anything.** Work spans machines (local Mac M4 / desktop /
  vast.ai); the remote is source of truth.
- Spin up a sub-agent to commit raw results (`*.jsonl`, `*.json`, summaries) after each stage so partial work
  survives a crash. `.gitignore` keeps `*.pt`, `*.npz`, `runs_*.out`, `wandb/` out of git —
  checkpoints go to HF dataset `nemesismaniac/vlm-sink-emergence-ckpts`, not git.
- Don't claim metrics that weren't measured (MMStar is usually skipped). State what ran,
  what was skipped; show failing output verbatim.
