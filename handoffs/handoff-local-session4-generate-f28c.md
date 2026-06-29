TO: local Agent   FROM: Engineering Manager   RE: build generate.py (exit-runbook step 1) — zero-cost, BLOCKS the cloud quality-check

CONTEXT (rest in repo — `docs/exit-runbook.md` step 1, `handoffs/handoff-em-session4-exit.md`):
- Box is exiting after the Session-4 seed-2 runs. `generate.py` is the prerequisite for the cloud agent's on-box quality spot-check (exit step 2) — build + commit it to `sink-emergence` ASAP so the cloud isn't blocked when its runs finish.

ASK:
- **`generate.py`** (minimal, zero-cost local): load a checkpoint (raw bf16 state-dict via `VLMConfig` from the run's `run_config.json`, OR `from_pretrained` after export) + generate text on an (image, prompt) pair. Double duty: arm quality-check now + "play with it" later. Mirror existing model-loading conventions (`models/vision_language_model.py`, the resume path in `train_sinks.py:131`); reuse the image processor/tokenizer from `data/processors.py`.
- Smoke-test locally on CPU/MPS with any available small ckpt (or a from_pretrained nanoVLM) so the cloud agent inherits a working script, not an untested one.
- Land on `sink-emergence` (the branch cloud pulls). Do NOT touch the `mac-reorg` branch.

CONSTRAINTS: zero GPU spend; local only. This is on the exit critical path — cloud quality-check waits on it.

RETURN: confirmation `generate.py` committed+pushed to `sink-emergence` + a one-line local smoke result (loaded ckpt X, generated on image Y). Flag any model-load assumption the cloud must satisfy on-box.
