# RF (fresh-data baseline 1B) — BLOCKED on instance 40436103, not run this session

**Why:** the FRESH FineVision pool (11 natural-image configs in `data/mixes.py:FRESH`) totals
**~1.49 TB** to download (measured via `load_dataset_builder(...).info.download_size`), and the
loader (`mixes.load_mix`, `limit_per_subset=None`) does a **non-streaming full `load_dataset`**.
This instance has an 80 GB disk (~13 GB free), so RF would fill the disk and crash, wasting the
~$7/10h. A disk-trimmed fresh pool can't reach the ≤2–4 visual-epoch "diversity" regime either
(≥2.75M unique images ≈ ~290 GB). G0.2 (streamed smoke) validates the fresh loader *parses* real
data, but does NOT exercise the full-load disk path.

**To run RF, pick one (Mac/director decision):**
1. Add a **streaming training path** (HF `IterableDataset` + shuffle buffer) so fresh data is
   consumed on-the-fly without materializing 1.49 TB. Smallest change; keeps this hardware.
2. Provision a **multi-TB / volume-backed** instance and run the existing non-streaming loader.
3. Use a smaller already-local fresh shard only for a *lower-token* pilot (loses the ≤2-epoch
   property — not a clean Gate-A retest).

Everything else this session (narrow Gate A at 1B, G0.2 loader parse-check, R2 Gate-B seeds,
R3 reprobe) completed and is on HF + git.
