# Smoke-test log — freshdata-arm (Mac M4, CPU, $0)

Authored + smoke-tested on the M4 with `python3` (torch 2.8.0, datasets 4.5.0, transformers
4.57.6; torchvision absent → pure-PIL image-processor fallback). No GPU training on the Mac.

## Acceptance checklist (from the directive)

- [x] **5-step run, bs=2, 1 probe call — completes on both `data_mode`s.**
  - `baseline --data_mode repeated --fake_data` → finished step 5, 3 probe calls, dual val. ✔
  - `baseline --data_mode fresh --fake_data` → finished step 4 (exercises the FRESH-spec probe
    branch: probe still built from the repeated tail). ✔
  - `g1gate --data_mode repeated --fake_data` (232M params, gate path) → finished step 4. ✔
- [x] **Loader prints unique-image count + projected epochs; fresh-mode assert.**
  - repeated: 147,755 unique → 52.87 (nominal) / 75.20 (eff.) epochs @ 1B.
  - fresh: **4,644,331 unique → 1.68 (nominal) / 2.39 (eff.) epochs @ 1B** — warns (>2, ≤4), no fail.
  - assert-guard: a 50k-image fresh pool (222 epochs) raises AssertionError. ✔
- [x] **Probe validates against the model forward (<1e-2)** (in-probe assert active, no drift)
      and the **image-swap self-check returns "varies"** (pos0 v_cv 0.068–0.075 > 0.02). ✔
- [x] **Per-head outputs are [layers, heads] (NOT pre-pooled):** per-layer dict has
      `v_norm_pos0_perhead` / `v_norm_rest_perhead` (H=9); summary adds
      `v_ratio_perhead_{min,mean,max}` and raw `raw_{mean,max}_attn_pos0`. Back-compat keys
      (`sink_eps*`, `v_ratio_pos0`, `h_ratio_pos0`, `argmax_key`, `v_norm_rest`) retained so
      `analyze_sinks.py` / `gate_summary.py` keep working. ✔
- [x] **Auto-stop wired; `--dry-run` prints stop/shutdown without executing.**
      `bash run_session3.sh --dry-run` prints all train calls + both stop mechanisms
      (`stop_verify.sh` and `shutdown -h +30`), executes nothing. ✔
- [x] **`tests/test_invariants.py` (D5 gate 0.5×-at-init equivalence) passes** (max diff 0.0). ✔
- [x] `py_compile` + `bash -n` clean on all changed files.

## Commands used

```bash
python3 tests/test_invariants.py
python3 train_sinks.py --arm baseline --data_mode repeated --fake_data --vit_init random \
  --device cpu --batch_size 2 --workers 0 --max_steps 5 --probe_every 2 --val_every 2 \
  --val_size 16 --probe_n 8 --out_dir /tmp/smoke_baseline
python3 train_sinks.py --arm baseline --data_mode fresh  --fake_data --vit_init random --device cpu ...
python3 train_sinks.py --arm g1gate   --data_mode repeated --fake_data --vit_init random --device cpu ...
bash run_session3.sh --dry-run
```

## Notes / limits
- Smoke used `--fake_data` (synthetic in-memory pool) + `--vit_init random` so the run is fully
  offline and $0. The real-data streamed path (`--limit_per_subset`) and pretrained ViT need
  network + GPU and are exercised on the remote, not here.
- For full-fidelity local data tests, install matching torchvision: `pip install torchvision==0.23.0`
  (blocked in this session by policy; the pure-PIL fallback covered the smoke run).
