# RF recipe stays byte-identical to the Comparator; speed comes from the GPU, not the batch

On the Session-3 box the RF run showed **GPU compute-util 100% but VRAM only ~25/48 GB**. The
obvious "fill the spare VRAM" move — raising `--batch_size` above the Comparator's 128 — is
**rejected**. At 100% compute-util the cores are already saturated, so a larger batch yields
~0 throughput gain (more compute/step exactly offsets more samples/step → tok/s flat →
same wall-clock for the fixed 1B-token budget). It also breaks the experiment: bigger batch +
the same cosine LR schedule = fewer optimizer steps and different gradient noise, so RF would
no longer be the byte-identical Comparator clone Gate-A requires (see [[0001]]).

Decision: **recipe locked** (batch 128, `grad_accum=1`, comparator `max_steps`). The 23 GB of
idle VRAM is documented as **unrecoverable for this run** — it is not a throughput lever when
the run is compute-bound.

## Consequences

- The only real speed lever is a **higher-compute card** (e.g. H100-class vs the A40/L40-class
  48 GB box), which is a cost/hr ↔ wall-clock trade, orthogonal to VRAM. **Action for the next
  run: source a faster GPU rather than a bigger one**; measure tok/s per $ before committing.
- VRAM slack does *not* justify a smaller (24 GB) card here either: measured peak ~25 GB > 24 GB,
  so bs128 would OOM without activation-checkpointing (which adds recompute → slower on an
  already compute-bound run). Not worth it for one ~5–6 h run.
