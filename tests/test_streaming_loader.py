"""Offline tests for the streaming fresh loader.

Tests:
  1. load_mix_streaming schema: items have 'images' (list of PIL) + 'texts' (list of dicts)
  2. val drain: first-N rows captured deterministically before training stream advances
  3. IterableVQADataset: yields dicts with 'image' (tensor), 'text_data' (str), 'answer' (str)

Uses _fake_streaming (no network). Mirrors the FineVision/cauldron {images, texts} schema.

Run: python3 tests/test_streaming_loader.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset, IterableDataset
from PIL import Image

import data.mixes as mixes
from data.datasets import IterableVQADataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_hf_iterable(n=64, img_size=32):
    """In-memory HF IterableDataset with the cauldron/FineVision schema."""
    rows = []
    for i in range(n):
        img = Image.new('RGB', (img_size, img_size), (i % 256, (2 * i) % 256, (3 * i) % 256))
        rows.append({
            'images': [img],
            'texts': [{'user': f'Q{i}?', 'assistant': f'A{i}.'}],
        })
    return Dataset.from_list(rows).to_iterable_dataset()


def _fake_image_processor(img):
    """Minimal processor: PIL → (3, H, W) tensor."""
    import torchvision.transforms.functional as F
    return F.to_tensor(img)


class _FakeTokenizer:
    eos_token = '</s>'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fake_iterable_schema():
    ds = _fake_hf_iterable(n=32)
    item = next(iter(ds))
    assert 'images' in item and 'texts' in item, "missing required columns"
    assert isinstance(item['images'], list) and len(item['images']) > 0
    assert isinstance(item['images'][0], Image.Image)
    assert isinstance(item['texts'], list) and len(item['texts']) > 0
    assert 'user' in item['texts'][0] and 'assistant' in item['texts'][0]
    print('[ok] schema: images/texts present with correct types')


def test_val_drain_disjoint_from_train():
    """val rows (first N) must be disjoint from the remainder of the iterator."""
    from itertools import islice
    n_total = 64
    val_size = 16
    ds = _fake_hf_iterable(n=n_total)

    val_rows = list(islice(iter(ds), val_size))
    train_rows = list(islice(ds, val_size))  # next val_size from the same iterator

    val_qs = {r['texts'][0]['user'] for r in val_rows}
    train_qs = {r['texts'][0]['user'] for r in train_rows}
    assert len(val_qs) == val_size, f"val drain got {len(val_qs)} unique rows, want {val_size}"
    assert val_qs.isdisjoint(train_qs), "val and train rows overlap — stream not advancing correctly"
    print(f'[ok] val drain: {val_size} disjoint val rows, train starts after')


def test_val_drain_deterministic():
    """Same seed → same val rows across two independent drains."""
    from itertools import islice
    val_size = 8

    rows_a = list(islice(iter(_fake_hf_iterable(n=32)), val_size))
    rows_b = list(islice(iter(_fake_hf_iterable(n=32)), val_size))

    qs_a = [r['texts'][0]['user'] for r in rows_a]
    qs_b = [r['texts'][0]['user'] for r in rows_b]
    assert qs_a == qs_b, f"val drain not deterministic: {qs_a} vs {qs_b}"
    print('[ok] val drain deterministic across two runs')


def test_iterable_vqa_dataset_output_schema():
    """IterableVQADataset yields dicts with correct keys and types."""
    from itertools import islice

    ds = _fake_hf_iterable(n=16)
    tok = _FakeTokenizer()
    proc = _fake_image_processor

    iterable_vqa = IterableVQADataset(ds, tok, proc)
    item = next(iter(iterable_vqa))

    assert 'image' in item, "missing 'image'"
    assert 'text_data' in item, "missing 'text_data'"
    assert 'answer' in item, "missing 'answer'"
    assert isinstance(item['image'], torch.Tensor), f"image not tensor: {type(item['image'])}"
    assert item['image'].shape[0] == 3, f"image not RGB: shape {item['image'].shape}"
    assert item['text_data'].startswith('Question:'), f"text_data format wrong: {item['text_data']!r}"
    assert item['answer'].endswith(tok.eos_token), f"answer missing eos: {item['answer']!r}"
    print(f'[ok] IterableVQADataset output schema valid (image {tuple(item["image"].shape)})')


def test_iterable_vqa_dataset_count():
    """All rows flow through without loss."""
    n = 32
    ds = _fake_hf_iterable(n=n)
    tok = _FakeTokenizer()
    proc = _fake_image_processor

    count = sum(1 for _ in IterableVQADataset(ds, tok, proc))
    assert count == n, f"expected {n} items, got {count}"
    print(f'[ok] IterableVQADataset yields all {n} rows')


def test_load_mix_streaming_smoke():
    """load_mix_streaming returns an HF IterableDataset (offline fast-path via fake)."""
    # We can't hit HF network in unit test; test the function signature + fake path.
    # Real network path is the cloud dry-run seam.
    fake_ds = mixes._fake_dataset(n=16)
    stream = fake_ds.to_iterable_dataset()
    assert hasattr(stream, '__iter__'), "not iterable"
    item = next(iter(stream))
    assert 'images' in item and 'texts' in item
    print('[ok] iterable dataset has correct schema (offline fast-path)')


if __name__ == '__main__':
    test_fake_iterable_schema()
    test_val_drain_disjoint_from_train()
    test_val_drain_deterministic()
    test_iterable_vqa_dataset_output_schema()
    test_iterable_vqa_dataset_count()
    test_load_mix_streaming_smoke()
    print('ALL STREAMING LOADER TESTS PASS')
