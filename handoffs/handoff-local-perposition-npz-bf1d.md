TO: local Agent   FROM: Engineering Manager   RE: close per-position open-Q #1 on the MAC (zero-cost) — reprobe npz + attention-location for seed-1/seed-2/RF

CONTEXT (rest in repo — Auditor brief `handoffs/handoff-em-perposition-fetch.md`, `deliverables/preprint_readiness_audit.md` item #1):
- Auditor closed seed-0 (pos0 = max-mass all 4 arms). Remaining = the runs HF has NO reprobe npz for: **seed-1, seed-2, RF**. Headline question: does textinit's sink position **migrate off pos0** at seed-2 (live-probe argmax says s2 → pos1/pos13)? And is RF's argmax-pos1 a real mass-sink or noise?
- **Re-routed cloud→local: the billed box is STOPPED/torn down — do NOT resurrect it.** This is an inference re-walk on a 32-sample probe batch over a few ckpts; it is NOT training. EM already patched `reprobe.py:181` to auto-select mps/cpu → runs on the Mac in minutes.

TASK (all on Mac, zero GPU cost):
1. **Fetch ckpts from HF** (use `hf download`, `huggingface-cli` is deprecated). Pull the per-run ckpt dirs you lack into their `runs/<run>/` so `reprobe.py --run_dir` finds `ckpt_step{N}.pt`:
   `hf download nemesismaniac/vlm-sink-emergence-ckpts --repo-type dataset --include "runs/textinit_seed1/*" --include "runs/textinit_seed2/*" --include "runs/g1gate_seed1/*" --include "runs/g1gate_seed2/*" --include "runs/rf_fresh_baseline/*" --local-dir .`
   (textinit_seed2 + RF are the priority pair; add baseline_seed1/sigmoid_seed1 only if their ckpts are on HF and you want full completeness.)
2. **Regen reprobe npz** — `python reprobe.py --arm <arm> --run_dir runs/<run> --ckpts <latest_step>` for each run (latest ckpt per run is enough; the per-position script reads `latest_npz`). Keep `--subsets` at the default the_cauldron probe batch for ALL runs **including RF** — apples-to-apples vs the seed-0 npz; the question is attention *location*, not data domain. Flag this choice in the output. `*.npz` is gitignored → do NOT commit it.
3. **Edit RUNS map** in `analysis/per_position_attention_from_npz.py` to point at the seed-1/seed-2/RF run dirs (currently seed-0 only), then `python analysis/per_position_attention_from_npz.py`.
4. **Commit** `analysis/per_position_attention.json` (+ any figure) to `sink-emergence`.

CONSTRAINTS: zero GPU spend; Mac mps/cpu only; no cloud box. reprobe device-patch already landed (verify it's on your tree). `*.npz` stays out of git (gitignored) — commit only the analysis json/svg.

RETURN: one line — "pos0 is max-mass for {arms/seeds}"; explicitly flag **textinit-s2** (migrates off pos0? to which pos?) and **RF** (argmax-pos1 real mass-sink or noise?). Auditor folds into the v1 limitations.
