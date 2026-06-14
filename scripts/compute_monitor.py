#!/usr/bin/env python3
"""Append one JSON record to compute_log.jsonl every ~interval seconds.

Also writes compute_status.txt for a quick human-readable glance.

Usage (background, on the GPU instance):
    python3 scripts/compute_monitor.py \
        --train_log /workspace/vlm-sink-emergence/runs/<arm>/train_log.jsonl \
        --out_dir   /workspace/vlm-sink-emergence \
        --interval  45 &

Consumers:
  - watchdog_rf.sh   reads compute_log.jsonl to detect NaN loss
  - EM               tails compute_log.jsonl for the session report
  - human            reads compute_status.txt (no JSON parsing)

Reads from env: CONTAINER_ID, CONTAINER_API_KEY (both required for cost tracking).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=10)
        return out.decode().strip()
    except Exception:
        return ''


def gpu_stats():
    raw = _run(
        'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,'
        'power.draw,temperature.gpu --format=csv,noheader,nounits'
    )
    if not raw:
        return {}
    parts = [p.strip() for p in raw.split(',')]
    if len(parts) < 5:
        return {}
    try:
        return {
            'gpu_util_pct': float(parts[0]),
            'vram_used_mb': float(parts[1]),
            'vram_total_mb': float(parts[2]),
            'power_w': float(parts[3]),
            'temp_c': float(parts[4]),
        }
    except ValueError:
        return {}


def train_log_latest(log_path):
    """Return the latest loss/step/tok_s from train_log.jsonl (tail 80 lines)."""
    if not log_path or not os.path.exists(log_path):
        return {}
    try:
        lines = _run(f'tail -n 80 {log_path}').splitlines()
    except Exception:
        return {}
    result = {}
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ('loss', 'step', 'tok_s', 'tokens_seen', 'val_loss', 'val_unseen', 'val_seen'):
            if key in r and key not in result:
                result[key] = r[key]
        if len(result) >= 7:
            break
    return result


def disk_stats(workspace='/workspace'):
    out = {}
    # df on /workspace
    raw = _run(f'df -BM {workspace} 2>/dev/null | tail -1')
    if raw:
        parts = raw.split()
        try:
            out['disk_used_mb'] = int(parts[2].rstrip('M'))
            out['disk_avail_mb'] = int(parts[3].rstrip('M'))
        except (IndexError, ValueError):
            pass
    # shard cache footprint (HF_HOME)
    hf_home = os.environ.get('HF_HOME', os.path.join(workspace, '.hf_home'))
    raw2 = _run(f'du -sm {hf_home} 2>/dev/null | cut -f1')
    if raw2:
        try:
            out['hf_cache_mb'] = int(raw2)
        except ValueError:
            pass
    return out


def vast_cost():
    cid = os.environ.get('CONTAINER_ID', '')
    key = os.environ.get('CONTAINER_API_KEY', '')
    if not cid or not key:
        return {}
    raw = _run(f'vastai show instance {cid} --api-key {key} --raw 2>/dev/null')
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {
            'dph_total': data.get('dph_total'),
            'actual_status': data.get('actual_status'),
        }
    except (json.JSONDecodeError, AttributeError):
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_log', default='', help='path to train_log.jsonl')
    p.add_argument('--out_dir', default='/workspace/vlm-sink-emergence',
                   help='directory for compute_log.jsonl + compute_status.txt')
    p.add_argument('--interval', type=int, default=45, help='seconds between ticks')
    args = p.parse_args()

    log_path = os.path.join(args.out_dir, 'compute_log.jsonl')
    status_path = os.path.join(args.out_dir, 'compute_status.txt')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[monitor] writing to {log_path} every {args.interval}s', flush=True)
    t0 = time.time()

    while True:
        rec = {'wall': round(time.time() - t0, 1), 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        rec.update(gpu_stats())
        rec.update(train_log_latest(args.train_log))
        rec.update(disk_stats())
        rec.update(vast_cost())

        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

        # terse human status line
        gpu_u = int(rec['gpu_util_pct']) if 'gpu_util_pct' in rec else '?'
        vram = int(rec['vram_used_mb']) if 'vram_used_mb' in rec else '?'
        vtot = int(rec['vram_total_mb']) if 'vram_total_mb' in rec else '?'
        tok_s = rec.get('tok_s', '?')
        loss = rec.get('loss', '?')
        step = rec.get('step', '?')
        disk_a = rec.get('disk_avail_mb', '?')
        dph = round(rec['dph_total'], 4) if 'dph_total' in rec else '?'
        status_line = (
            f"{rec['ts']}  gpu={gpu_u}%  vram={vram}/{vtot}MB  "
            f"step={step}  loss={loss}  tok/s={tok_s}  "
            f"disk_avail={disk_a}MB  $/h={dph}\n"
        )
        with open(status_path, 'w') as f:
            f.write(status_line)
        print(f'[monitor] {status_line.strip()}', flush=True)

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
