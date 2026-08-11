# AGENTS.md

Be extremely concise and sacrifice grammar for the sake of concision.

## What this is

Research, not a product. Question: **when/where do attention sinks emerge during VLM
pretraining, and do attention-concentration vs value-norm / massive-activation signatures
*decouple*?** Fork of nanoVLM v0.1 (222M = SigLIP-B/16 + SmolLM2-135M-arch decoder).
**Ignore the global AGENTS.md's PaliGemma description — that is a different project.**

## Read first

- `technical-reference-manual.md` - core research document
- `references/plan_when-sinks-emerge-in-vlms.md` — the plan, scoop check, deadlines.
- `REPORT.md` — findings so far. Read before proposing experiments; don't re-derive.
- `docs/open-questions.md` — known soft spots / reviewer-critique pre-empts + cheap fixes (read before writeup).
- `docs/experiments.md` — arms, knobs, how to run.
- `docs/conventions.md` — metric definitions, probe contract, key files.
- `docs/cloud-agent.md` — north star + rules for the vast.ai cloud agent.
- `docs/handoff-contract.md` — one format for all role-to-role messages.

(The stock nanoVLM `README.md` / `nanoVLM.ipynb` are upstream docs.)

## Agent roles (fleet)

Skills in `.Codex/skills/` — invoke one to adopt that role in a fresh thread (you spawn each agent yourself, across Mac / desktop / vast.ai). Delegation = `docs/handoff-contract.md`: try `SendMessage`, else a lean `.md` to carry across devices.

```
Director (1, Opus) ── orchestrates, holds context, go/no-go
├─ Engineering Manager (2, Sonnet) ── compacts eng → Director; commands the fleet
│   ├─ Cloud Agent (3) ── billed GPU; self-stops on waste (docs/cloud-agent.md)
│   └─ local Agent (4) ── builds/tests infra; zero-cost verifications
├─ Auditor (2, Opus) ── audits data, kills bias, publication viz
├─ Researcher (2, Desktop) ── prior-art scout, no code access
└─ Drafter (2, Opus) ── writes project docs, outline-first; fact-checks peer-to-peer w/ Auditor/Researcher
```

Director never ingests raw logs — EM/Auditor compact first. Rank-2s report straight to Director.

## Workflow (mandatory)

- **`git pull` before changing anything.** Work spans machines (local Mac M4 / desktop /
  vast.ai); the remote is source of truth.
- Spin up a sub-agent to commit raw results (`*.jsonl`, `*.json`, summaries) after each stage so partial work
  survives a crash. `.gitignore` keeps `*.pt`, `*.npz`, `runs_*.out`, `wandb/` out of git —
  checkpoints go to HF dataset `nemesismaniac/vlm-sink-emergence-ckpts`, not git.
- Don't claim metrics that weren't measured (MMStar is usually skipped). State what ran,
  what was skipped; show failing output verbatim.
