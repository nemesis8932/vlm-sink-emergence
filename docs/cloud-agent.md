# Cloud agent (vast.ai)

This is the agent that runs on a rented vast.ai GPU instance. It also follows the root
`CLAUDE.md`; this file is its operating contract.

## North star

**Every paid GPU-minute produces results, and no result is ever lost.** Two failure modes to
design against, both cost money or data: (1) the GPU sits idle while you do CPU/upload/git
work; (2) the instance keeps billing after the work is done, or a crash takes uncommitted
data with it.

## Rules

1. **Keep the GPU busy back-to-back.** Start the longest-priority run *first*. Slot
   CPU-bound work (reprobes that fit, HF uploads, git) into gaps where the GPU would idle —
   never block training on them. Run uploads in the background (`&`), `wait` only before
   freeing disk.
2. **Persist before you free.** After each stage: commit raw results (`*.jsonl`/`*.json`/
   summaries) and push, then upload checkpoints to HF dataset
   `nemesismaniac/vlm-sink-emergence-ckpts`. Delete local `*.pt` **only after** the upload
   has landed (`wait $PID`). A crash mid-stage must never lose a completed run.
3. **Verified stop — the idle tail is the #1 waste.** When all work is done, stop the
   instance and *confirm it actually stopped* (`stop_verify.sh`); `vastai stop` can silently
   no-op. Keep a backstop timer (`watchdog.sh`) but don't trust it alone — the worst Session-1
   waste was ~2.2h idle ($1.5) after the backstop failed to take.
4. **Send everything reproducible off-box.** Checkpoints → HF; jsonl/json/figures → git.
   The instance is disposable; assume it's wiped the moment work ends. Nothing important
   should live only on `/workspace`.
5. **Measure before you spend.** `measure_vram.py` to pick batch size; smoke-test the recipe
   at tiny `--max_steps` before launching a multi-hour run.
6. **Budget honestly.** Log GPU-hours and $ per arm in the report (rate × wall). Call out
   avoidable waste (OOM retries, idle tail, relaunches) so the next session improves.

## Orchestration

`run_session2.sh` encodes rules 1–3 for a multi-arm session on one GPU (priority run first,
background HF uploads, reprobes in the gaps, commit+push after every stage, `stop_verify.sh`
at the end). `watchdog.sh` = independent backstop stop. Edit `run_session2.sh` to define a
new session; keep the persist-then-free and verified-stop structure intact.

Env on the instance: `source /workspace/venv/bin/activate`;
`HF_HOME=/workspace/.hf_home`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
