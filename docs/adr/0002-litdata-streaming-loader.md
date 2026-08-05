# Stream fresh data via litdata shards, not HF `datasets` streaming

The HF `datasets` streaming loader used for the RF run leaked memory two ways: a large
**static base** (its `shuffle(buffer_size)` holds RAW PIL-decoded rows, and each DataLoader
worker keeps its own buffer → `workers × buffer × img`, e.g. 12×10k decoded images), and a
**monotonic creep** on top (persistent workers holding HF streaming state + glibc malloc
fragmentation; `MALLOC_ARENA_MAX=2` only dented it). This forced workers 12→10→8 and caused
the step-4000 OOM crash that nearly confounded RF. We are replacing it with **litdata**
StreamingDataset: shards hold **encoded bytes**, decoded **on-demand** per sample, with a
**capped local cache** (~20GB) — O(1) memory by construction. Allocator swapped to
**jemalloc/tcmalloc** (`LD_PRELOAD`) as the library-agnostic kill for residual glibc creep.

## Pipeline

- FineVision → litdata chunks via a **one-time, off-box** conversion (user device or a cheap
  CPU instance) — the billed GPU box never converts.
- Shards **served from HF** (CDN-backed HTTPS with byte-range requests). **Not Google Drive**:
  no S3 API, no range support, throttles large sequential reads → would stall the GPU.
  GDrive (~2TB) is a **cold backup mirror** only.

## Considered options

- **litdata + HF-served shards (chosen).** Structurally kills the static base + HF-internal
  creep; capped cache fits the 80GB box; bonus deterministic resume. Cost: one-time conversion
  + new dep.
- **Patch HF streaming path** (persistent_workers off, encoded smaller buffer, allocator). No
  new dep, but HF streaming creep may be intrinsic — not provably killed. Rejected as
  not-foolproof for the longer Stage-2 runs.
- **Materialize a ~290GB shard** to a volume, train map-style. Impossible on the 80GB disk.

## Consequences

Hard gate before any spend: local-agent reproduces the creep with an RSS-vs-time profile to
**attribute** it (HF-internal vs glibc) and proves the litdata path stays flat on a small
subset. No paid box until the leak is provably dead locally. HF storage is finite — store only
what's needed; mirror bulk to GDrive.
