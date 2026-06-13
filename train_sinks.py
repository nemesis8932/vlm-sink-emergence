"""From-scratch VLM pretraining with dense attention-sink probing.

One process = one arm. Arms differ in attention variant and init:
  baseline   : softmax attention, random-init LM, pretrained ViT
  g1gate     : + elementwise sigmoid output gate (zero-init), random-init LM
  sigmoid    : sigmoid (unnormalized) attention, random-init LM
  textinit   : softmax attention, pretrained SmolLM2-135M LM (sinks inherited?)
Plus --vit_init random for the fully-from-scratch variant of any arm.
"""

import argparse
import contextlib
import json
import math
import os
import random
import time

import numpy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Seeding happens in main() from --seed. The *probe batch* is always rebuilt under a
# fixed random.seed(0) in get_data, so it is byte-identical across seeds/runs (metrics
# are then comparable seed-to-seed), while model init and data order follow --seed.

from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
from models.vision_transformer import ViT
from models.language_model import LanguageModel
import models.config as config
import data.mixes as mixes
from sink_probe import probe_sinks, image_swap_self_check

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ARMS = {
    'baseline': dict(gate=False, attn='softmax', lm_init='random'),
    'g1gate':   dict(gate=True,  attn='softmax', lm_init='random'),
    'sigmoid':  dict(gate=False, attn='sigmoid', lm_init='random'),
    'textinit': dict(gate=False, attn='softmax', lm_init='pretrained'),
}

CKPT_STEPS = {0, 250, 1000, 2000, 4000, 8000, 12000, 16000, 24000}


def get_lr(it, max_lr, max_steps, warmup_steps):
    min_lr = max_lr * 0.1
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def _shared_probe(collator, repeated_val_dataset, probe_n):
    """Fixed, versioned, held-out probe shared across ALL arms/runs (mixes.PROBE_VERSION).
    Always the repeated subsets' val tail under random.seed(0) -> byte-identical to Session-1
    and identical for the fresh arm, so every arm/ckpt is measured on the same stimulus.
    Nothing consumes the global `random` stream before this, so the 32 QA choices are fixed."""
    random.seed(0)
    return collator([repeated_val_dataset[i] for i in range(probe_n)])


def get_data(args, vlm_cfg):
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    spec = mixes.spec_for(args.data_mode)

    # unique-image count + projected visual-epochs; fresh mode asserts the pool is big enough
    target_tokens = args.max_tokens_M * 1e6 if math.isfinite(args.max_tokens_M) else 1e9
    if not args.fake_data:
        mixes.report_data_stats(spec, args.data_mode, target_tokens=target_tokens,
                                tok_per_sample=args.tok_per_sample)

    ds = mixes.load_mix(spec, limit_per_subset=args.limit_per_subset, seed=0, fake=args.fake_data)
    val_size = min(args.val_size, len(ds) // 4)
    train_rows = ds.select(range(len(ds) - val_size))
    unseen_rows = ds.select(range(len(ds) - val_size, len(ds)))      # held-out images (primary val)
    seen_rows = train_rows.select(range(min(val_size, len(train_rows))))  # seen images, fresh QA

    train_dataset = VQADataset(train_rows, tokenizer, image_processor)
    val_unseen = VQADataset(unseen_rows, tokenizer, image_processor)   # PRIMARY: true generalization
    val_seen = VQADataset(seen_rows, tokenizer, image_processor)       # SECONDARY: memorization proxy
    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)  # data order varies with --seed; train/val split fixed (shuffle seed=0)
    mw = dict(persistent_workers=True, prefetch_factor=4) if args.workers > 0 else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker, generator=g, **mw)
    val_kw = dict(batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                  num_workers=2, pin_memory=True, drop_last=True)
    val_unseen_loader = DataLoader(val_unseen, **val_kw)
    val_seen_loader = DataLoader(val_seen, **val_kw)

    # shared probe: repeated val tail (reuse it if this IS the repeated arm, else build it)
    if spec is mixes.REPEATED:
        probe_val = val_unseen
    else:
        rep = mixes.load_mix(mixes.REPEATED, limit_per_subset=args.limit_per_subset, seed=0,
                             fake=args.fake_data)
        probe_val = VQADataset(rep.select(range(len(rep) - min(1024, len(rep) // 4), len(rep))),
                               tokenizer, image_processor)
    probe = _shared_probe(collator, probe_val, args.probe_n)
    return train_loader, val_unseen_loader, val_seen_loader, probe


def build_model(args, vlm_cfg):
    arm = ARMS[args.arm]
    vlm_cfg.lm_attn_gate = arm['gate']
    vlm_cfg.lm_attn_impl = arm['attn']

    model = VisionLanguageModel(vlm_cfg, load_backbone=False)  # all random
    if args.vit_init == 'pretrained':
        model.vision_encoder = ViT.from_pretrained(vlm_cfg)
    if arm['lm_init'] == 'pretrained':
        model.decoder = LanguageModel.from_pretrained(vlm_cfg)
    if args.resume:
        # Resume from a saved (bf16) checkpoint. Note: only model weights are saved, not
        # optimizer moments — Adam restarts cold, so expect a brief loss transient that
        # recovers within a few hundred steps. The LR schedule continues from --resume_step.
        sd = torch.load(args.resume, map_location='cpu')
        sd = {k: v.float() for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"resume mismatch missing={missing} unexpected={unexpected}"
        print(f"[resume] loaded {args.resume}", flush=True)
    return model


def save_ckpt(model, out_dir, step):
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    torch.save(sd, os.path.join(out_dir, f'ckpt_step{step}.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--vit_init', choices=['pretrained', 'random'], default='pretrained')
    p.add_argument('--max_steps', type=int, default=20000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--lr_lm', type=float, default=4e-4)      # Gu et al. from-scratch LR
    p.add_argument('--lr_mp', type=float, default=2e-3)
    p.add_argument('--lr_vit', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--probe_every', type=int, default=100)
    p.add_argument('--probe_n', type=int, default=32)
    p.add_argument('--val_every', type=int, default=500)
    p.add_argument('--data_mode', choices=['repeated', 'fresh'], default='repeated',
                   help="repeated = 4 cauldron subsets (S1 baseline); fresh = FineVision natural pool")
    p.add_argument('--val_size', type=int, default=1024, help='held-out images for primary val')
    p.add_argument('--tok_per_sample', type=float, default=90.0,
                   help='effective tokens/sample for the epoch projection (49 image + text)')
    p.add_argument('--limit_per_subset', type=int, default=None,
                   help='stream only N rows/subset (cheap smoke test on real data)')
    p.add_argument('--fake_data', action='store_true',
                   help='offline synthetic pool for smoke testing (no network)')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--max_hours', type=float, default=100.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto', help="auto|cuda|mps|cpu (CPU/MPS for Mac smoke tests)")
    p.add_argument('--max_tokens_M', type=float, default=float('inf'),
                   help='stop when tokens_seen >= this * 1e6 (same unit as the report Mtok)')
    p.add_argument('--resume', type=str, default=None, help='ckpt .pt to resume model weights from')
    p.add_argument('--resume_step', type=int, default=0, help='step counter to resume at (LR schedule position)')
    p.add_argument('--resume_tokens', type=int, default=0, help='tokens_seen to resume at')
    p.add_argument('--ckpt_steps', type=str, default=None, help='comma-sep step numbers to checkpoint at')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt_set = set(int(x) for x in args.ckpt_steps.split(',')) if args.ckpt_steps else CKPT_STEPS

    out_dir = args.out_dir or f'runs/{args.arm}_vit-{args.vit_init}'
    os.makedirs(out_dir, exist_ok=True)

    vlm_cfg = config.VLMConfig()
    vlm_cfg.data_mode = args.data_mode
    train_loader, val_unseen_loader, val_seen_loader, probe = get_data(args, vlm_cfg)
    model = build_model(args, vlm_cfg)

    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump({'args': vars(args), 'vlm_cfg': vlm_cfg.__dict__}, f, indent=2, default=str)

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[{args.arm}] {n_params:,} params | {len(train_loader.dataset)} train samples | bs {args.batch_size}", flush=True)

    if args.device != 'auto':
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    use_amp = device.type == 'cuda'   # bf16 autocast only on CUDA; CPU/MPS smoke runs fp32

    def amp():
        return (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                if use_amp else contextlib.nullcontext())

    print(f"[{args.arm}] device={device} amp={use_amp}", flush=True)
    model.to(device)

    param_groups = [
        {'params': list(model.MP.parameters()), 'lr': args.lr_mp},
        {'params': list(model.decoder.parameters()), 'lr': args.lr_lm},
        {'params': list(model.vision_encoder.parameters()), 'lr': args.lr_vit},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    base_lrs = [args.lr_mp, args.lr_lm, args.lr_vit]
    warmup = max(100, int(0.03 * args.max_steps))

    fwd_model = torch.compile(model) if (args.compile and device.type == 'cuda') else model

    log_f = open(os.path.join(out_dir, 'train_log.jsonl'), 'a')
    probe_f = open(os.path.join(out_dir, 'probes.jsonl'), 'a')

    def run_probe(step, tokens_seen):
        t0 = time.time()
        res = probe_sinks(model, probe['image'], probe['input_ids'], probe['attention_mask'])
        res.update(step=step, tokens_seen=tokens_seen, wall=time.time())
        probe_f.write(json.dumps(res) + '\n')
        probe_f.flush()
        s = res['summary']
        print(f"[probe {args.arm}] step {step} tok {tokens_seen/1e6:.1f}M | "
              f"sink.2/.3/.4 {s['sink_eps0.2']:.3f}/{s['sink_eps0.3']:.3f}/{s['sink_eps0.4']:.3f} | "
              f"max_a0 {s['max_attn_pos0']:.3f} mean_a0 {s['mean_attn_pos0']:.4f} | "
              f"v_ratio {s['v_ratio_pos0']:.3f} h_ratio {s['h_ratio_pos0']:.3f} | {time.time()-t0:.1f}s", flush=True)

    def run_val(loader):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 4:
                    break
                with amp():
                    _, loss = model(batch['input_ids'].to(device), batch['image'].to(device),
                                    attention_mask=batch['attention_mask'].to(device),
                                    targets=batch['labels'].to(device))
                tot += loss.item(); n += 1
        model.train()
        return tot / max(n, 1)

    img_tokens = (vlm_cfg.vit_img_size // vlm_cfg.vit_patch_size) ** 2 // vlm_cfg.mp_pixel_shuffle_factor ** 2

    # harness self-check: pos0 is a content-bearing image token, so it MUST vary across images
    v_cv, a0_std = image_swap_self_check(model, probe['image'], probe['input_ids'],
                                         probe['attention_mask'])
    print(f"[{args.arm}] image-swap self-check OK: pos0 v_cv {v_cv:.3f} attn0_std {a0_std:.4f} "
          f"(probe {mixes.PROBE_VERSION})", flush=True)

    step = args.resume_step
    tokens_seen = args.resume_tokens
    t_start = time.time()
    run_probe(step, tokens_seen)          # re-measure at the resume point (or step 0 fresh)
    save_ckpt(model, out_dir, step)
    model.train()

    done = False
    last_t = time.time()
    while not done:
        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            with amp():
                _, loss = fwd_model(input_ids, images, attention_mask=attention_mask, targets=labels)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0 or args.grad_accum == 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for gi, g_ in enumerate(optimizer.param_groups):
                    g_['lr'] = get_lr(step, base_lrs[gi], args.max_steps, warmup)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            tokens_seen += int(attention_mask.sum().item()) + images.size(0) * img_tokens
            step += 1

            if step % 25 == 0:
                dt = time.time() - last_t
                last_t = time.time()
                rec = {'step': step, 'loss': round(loss.item(), 4),
                       'lr': optimizer.param_groups[1]['lr'],
                       'tok_s': round(tokens_seen and (25 * args.batch_size * 128) / dt, 1),
                       'tokens_seen': tokens_seen, 'wall': round(time.time() - t_start, 1)}
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                if step % 100 == 0:
                    print(f"[{args.arm}] step {step} loss {rec['loss']} tok/s {rec['tok_s']}", flush=True)

            if step % args.probe_every == 0:
                run_probe(step, tokens_seen)
                last_t = time.time()
            if step % args.val_every == 0:
                vl = run_val(val_unseen_loader)       # PRIMARY: held-out (unseen) images
                vs = run_val(val_seen_loader)         # SECONDARY: seen images / fresh QA
                log_f.write(json.dumps({'step': step, 'val_loss': round(vl, 4),
                                        'val_unseen': round(vl, 4), 'val_seen': round(vs, 4)}) + '\n')
                log_f.flush()
                print(f"[{args.arm}] step {step} VAL unseen {vl:.4f} seen {vs:.4f}", flush=True)
                last_t = time.time()
            if step in ckpt_set:
                save_ckpt(model, out_dir, step)
                last_t = time.time()

            if (step >= args.max_steps or (time.time() - t_start) > args.max_hours * 3600
                    or tokens_seen >= args.max_tokens_M * 1e6):
                done = True
                break

    run_probe(step, tokens_seen)
    save_ckpt(model, out_dir, step)
    print(f"[{args.arm}] finished at step {step}, {tokens_seen/1e6:.1f}M tokens, "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
