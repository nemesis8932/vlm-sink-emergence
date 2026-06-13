"""Data-pool presets + unique-image accounting for the fresh-vs-repeated baseline.

The fresh-vs-repeated comparison is only valid if data is the ONLY thing that differs:
identical sequence layout, collator/label-mask, tokenizer and "Question: .. Answer:"
template across both. This module therefore only chooses *which rows* are loaded; it never
touches the model-facing template (that stays in data/datasets.py + data/collators.py).

  repeated : the original 4 the_cauldron subsets, EXACTLY as Session-1 (the comparison
             baseline). ~1.5M QA over 146,731 unique images -> ~40 visual epochs at 1B tok.
  fresh    : a curated FineVision natural-image pool (~4.6M unique images) so 1B tokens is
             ~1-2 epochs of *fresh* data. FineVision shares the_cauldron's
             {images, texts:[{user,assistant}]} schema, so it is drop-in through the existing
             VQADataset/VQACollator -- no template change.

Why FineVision and not "more the_cauldron subsets": the entire the_cauldron is only ~1.88M
unique images (all 47 subsets), so it cannot reach <=2 epochs at 1B. FineVision (the pool
nanoVLM-450M used) is ~5M+ rows; the natural-image configs below were chosen to (a) clear
~4.6M unique images and (b) stay close to the COCO-heavy repeated distribution so the
comparison isolates *repetition*, not domain shift. Synthetic / document / chart / OCR /
math / text-only configs are deliberately excluded.
"""

from itertools import islice

from datasets import Dataset, concatenate_datasets, load_dataset

CAULDRON = 'HuggingFaceM4/the_cauldron'
FINEVISION = 'HuggingFaceM4/FineVision'

# repeated baseline -- DO NOT CHANGE (Session-1 comparison baseline)
REPEATED = dict(dataset=CAULDRON, subsets=['vqav2', 'cocoqa', 'aokvqa', 'vsr'])

# fresh natural-image pool. Excludes the repeated subsets so the pools are (mostly) disjoint;
# some COCO-derived configs share COCO photos with the repeated set (<3% of the fresh pool).
FRESH = dict(dataset=FINEVISION, subsets=[
    'objects365_qa',                 # 1,665,847  Objects365 natural scenes
    'densefusion_1m',                # 1,058,751  dense captions, natural web images
    'allava_laion',                  #   468,664  LAION natural images
    'lnqa',                          #   302,780  Localized-Narratives QA
    'google_landmarks',              #   299,993  landmark photos
    'lvis_instruct4v',               #   222,711  LVIS (COCO) images
    'localized_narratives',          #   199,998  natural
    'LLaVA_Instruct_150K',           #   157,710  COCO
    'coco_colors',                   #   118,287  COCO
    'image_textualization(filtered)',#    99,573  natural dense captions
    'sharegpt4v(coco)',              #    50,017  COCO
])

# Fixed, versioned, held-out probe set, shared across ALL arms/runs for comparability and
# IDENTICAL to Session-1 (so already-collected S1 curves stay comparable): the first
# probe_n samples of the repeated subsets' shuffled val tail, under random.seed(0).
PROBE_VERSION = 'v1-repeatedtail-32'

# Offline fallback for unique-image counts (HuggingFace datasets-server, 2026-06-13). Lets the
# stats/assert run without a metadata fetch; the loader recomputes the true count from len(ds).
_KNOWN_COUNTS = {
    ('HuggingFaceM4/the_cauldron', 'vqav2'): 82772,
    ('HuggingFaceM4/the_cauldron', 'cocoqa'): 46287,
    ('HuggingFaceM4/the_cauldron', 'aokvqa'): 16539,
    ('HuggingFaceM4/the_cauldron', 'vsr'): 2157,
    ('HuggingFaceM4/FineVision', 'objects365_qa'): 1665847,
    ('HuggingFaceM4/FineVision', 'densefusion_1m'): 1058751,
    ('HuggingFaceM4/FineVision', 'allava_laion'): 468664,
    ('HuggingFaceM4/FineVision', 'lnqa'): 302780,
    ('HuggingFaceM4/FineVision', 'google_landmarks'): 299993,
    ('HuggingFaceM4/FineVision', 'lvis_instruct4v'): 222711,
    ('HuggingFaceM4/FineVision', 'localized_narratives'): 199998,
    ('HuggingFaceM4/FineVision', 'LLaVA_Instruct_150K'): 157710,
    ('HuggingFaceM4/FineVision', 'coco_colors'): 118287,
    ('HuggingFaceM4/FineVision', 'image_textualization(filtered)'): 99573,
    ('HuggingFaceM4/FineVision', 'sharegpt4v(coco)'): 50017,
}


def spec_for(data_mode):
    if data_mode == 'fresh':
        return FRESH
    if data_mode == 'repeated':
        return REPEATED
    raise ValueError(f"data_mode must be 'repeated' or 'fresh', got {data_mode!r}")


def _subset_count(dataset, name):
    """Unique-image count for one subset (rows == unique images in cauldron/FineVision).
    Metadata only -- never downloads images."""
    key = (dataset, name)
    if key in _KNOWN_COUNTS:
        return _KNOWN_COUNTS[key]
    from datasets import load_dataset_builder
    info = load_dataset_builder(dataset, name).info
    n = info.splits['train'].num_examples
    assert n, f"no train num_examples for {dataset}:{name}"
    return int(n)


def unique_image_count(spec):
    return sum(_subset_count(spec['dataset'], s) for s in spec['subsets'])


def projected_epochs(unique_images, target_tokens, tok_per_sample):
    """Visual epochs at the token budget: one image == one visit == one sample."""
    visits = target_tokens / tok_per_sample
    return visits / max(unique_images, 1)


def report_data_stats(spec, data_mode, target_tokens=1e9, tok_per_sample=90.0,
                      warn_epochs=2.0, hard_epochs=4.0, unique=None):
    """Log unique-image count + projected visual-epochs at the token budget. For fresh mode,
    assert the pool is large enough: hard-fail above `hard_epochs`, warn loudly above
    `warn_epochs`. tok_per_sample is the *effective* token accounting train_sinks uses
    (49 image tokens + real text tokens); 128 is also reported as the nominal upper bound."""
    unique = unique_image_count(spec) if unique is None else unique
    ep_eff = projected_epochs(unique, target_tokens, tok_per_sample)
    ep_nom = projected_epochs(unique, target_tokens, 128.0)
    print(f"[data] mode={data_mode} dataset={spec['dataset']} subsets={len(spec['subsets'])} "
          f"unique_images={unique:,}", flush=True)
    print(f"[data] projected visual-epochs @ {target_tokens/1e9:.2f}B tokens: "
          f"{ep_eff:.2f} (eff. {tok_per_sample:.0f} tok/sample) / "
          f"{ep_nom:.2f} (nominal 128)", flush=True)
    if data_mode == 'fresh':
        assert ep_eff <= hard_epochs, (
            f"fresh pool too small: {unique:,} unique images -> {ep_eff:.2f} epochs @ "
            f"{target_tokens/1e9:.2f}B (> hard limit {hard_epochs}); add more unique-image subsets")
        if ep_eff > warn_epochs:
            print(f"[data] WARNING: fresh pool gives {ep_eff:.2f} effective epochs @ "
                  f"{target_tokens/1e9:.2f}B (> target {warn_epochs}); nominal {ep_nom:.2f}. "
                  f"Acceptable but not the <=2 ideal -- note in report.", flush=True)
    return dict(unique_images=unique, epochs_eff=ep_eff, epochs_nominal=ep_nom,
                tok_per_sample=tok_per_sample, target_tokens=target_tokens)


def _fake_dataset(n=64, size=32):
    """Offline synthetic pool (random images + dummy QA) so the train/probe machinery can be
    smoke-tested with zero network. Same {images, texts:[{user,assistant}]} schema."""
    from PIL import Image
    rows = []
    for i in range(n):
        img = Image.new('RGB', (size, size), (i % 256, (2 * i) % 256, (3 * i) % 256))
        rows.append({'images': [img],
                     'texts': [{'user': f'What color is region {i}?', 'assistant': f'Color {i % 7}.'}]})
    return Dataset.from_list(rows)


def load_mix(spec, limit_per_subset=None, seed=0, fake=False):
    """Concatenate the spec's subsets into one shuffled Dataset (rows == unique images).

    fake=True            -> offline synthetic pool (smoke test, no network).
    limit_per_subset=N   -> stream only N rows/subset (cheap smoke test on real schema).
    otherwise            -> full load (the remote training path).
    Keeps only ['images','texts'] so configs with differing extra columns concat cleanly.
    Deterministic shuffle keyed to `seed` (train/val split fixed; data *order* varies via the
    DataLoader generator in train_sinks)."""
    if fake:
        return _fake_dataset()
    cols = ['images', 'texts']
    parts = []
    for name in spec['subsets']:
        if limit_per_subset:
            stream = load_dataset(spec['dataset'], name, split='train', streaming=True)
            rows = list(islice((r for r in stream), limit_per_subset))
            parts.append(Dataset.from_list([{c: r[c] for c in cols} for r in rows]))
        else:
            ds = load_dataset(spec['dataset'], name, split='train')
            parts.append(ds.select_columns(cols))
    ds = concatenate_datasets(parts)
    return ds.shuffle(seed=seed)
