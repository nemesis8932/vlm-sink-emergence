"""Build emergence figures + summary table from runs/*/probes.jsonl and train_log.jsonl."""

import json
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARM_COLORS = {'baseline': 'tab:blue', 'textinit': 'tab:orange', 'g1gate': 'tab:green', 'sigmoid': 'tab:red'}


def load_run(run_dir):
    probes = []
    with open(os.path.join(run_dir, 'probes.jsonl')) as f:
        for line in f:
            probes.append(json.loads(line))
    train, val = [], []
    lp = os.path.join(run_dir, 'train_log.jsonl')
    if os.path.exists(lp):
        with open(lp) as f:
            for line in f:
                r = json.loads(line)
                (val if 'val_loss' in r else train).append(r)
    return probes, train, val


def main(out_dir='analysis'):
    os.makedirs(out_dir, exist_ok=True)
    runs = {}
    for d in sorted(glob.glob('runs/*/')):
        name = d.rstrip('/').split('/')[-1]
        if os.path.exists(os.path.join(d, 'probes.jsonl')):
            runs[name] = load_run(d)
    if not runs:
        sys.exit('no runs found')

    # de-duplicate probes by step (restarts append)
    for name, (probes, t, v) in runs.items():
        seen = {}
        for p in probes:
            seen[p['step']] = p
        runs[name] = ([seen[s] for s in sorted(seen)], t, v)

    # 1) sink fraction + attn-to-pos0 curves
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for name, (probes, _, _) in runs.items():
        c = ARM_COLORS.get(name, None)
        toks = [p['tokens_seen'] / 1e6 for p in probes]
        for eps, ls in [('0.2', ':'), ('0.3', '-'), ('0.4', '--')]:
            axes[0, 0].plot(toks, [p['summary'][f'sink_eps{eps}'] for p in probes],
                            ls, color=c, label=f'{name} ε={eps}' if eps == '0.3' else None, alpha=0.8)
        axes[0, 1].plot(toks, [p['summary']['mean_attn_pos0'] for p in probes], color=c, label=name)
        axes[0, 1].plot(toks, [p['summary']['max_attn_pos0'] for p in probes], '--', color=c, alpha=0.6)
        axes[1, 0].plot(toks, [p['summary']['v_ratio_pos0'] for p in probes], color=c, label=name)
        axes[1, 1].plot(toks, [p['summary']['h_ratio_pos0'] for p in probes], color=c, label=name)
    axes[0, 0].set_title('Sink$^{ε}_1$: frac of (layer,head) with mean attn→pos0 > ε  ([:]=0.2, [-]=0.3, [--]=0.4)')
    axes[0, 1].set_title('mean (solid) / max (dashed) attention → pos 0')
    axes[1, 0].set_title('value-norm ratio: ‖v(pos0)‖ / ‖v(rest)‖  (drain < 1)')
    axes[1, 1].set_title('residual-stream norm ratio pos0/rest (massive activations)')
    for ax in axes.flat:
        ax.set_xlabel('tokens seen (M)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/emergence_curves.png', dpi=130)

    # 2) per-layer heatmaps of max-over-heads attn->pos0 (normalized view)
    fig, axs = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), squeeze=False)
    for i, (name, (probes, _, _)) in enumerate(runs.items()):
        M = np.array([[max(l['attn_to_pos0_norm']) for l in p['layers']] for p in probes]).T  # L x time
        im = axs[0, i].imshow(M, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1,
                              extent=[probes[0]['tokens_seen'] / 1e6, probes[-1]['tokens_seen'] / 1e6, 0, M.shape[0]])
        axs[0, i].set_title(f'{name}: max-head attn→pos0 per layer')
        axs[0, i].set_xlabel('tokens (M)'); axs[0, i].set_ylabel('layer')
        plt.colorbar(im, ax=axs[0, i])
    fig.tight_layout()
    fig.savefig(f'{out_dir}/layer_heatmaps.png', dpi=130)

    # 3) where does the sink sit: argmax-key segment fractions over time
    fig, axs = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4), squeeze=False)
    n_img = 49
    for i, (name, (probes, _, _)) in enumerate(runs.items()):
        toks = [p['tokens_seen'] / 1e6 for p in probes]
        f0, fimg, ftxt = [], [], []
        for p in probes:
            am = [k for l in p['layers'] for k in l['argmax_key']]
            n = len(am)
            f0.append(sum(k == 0 for k in am) / n)
            fimg.append(sum(0 < k < n_img for k in am) / n)
            ftxt.append(sum(k >= n_img for k in am) / n)
        axs[0, i].stackplot(toks, f0, fimg, ftxt, labels=['pos 0', 'image 1..48', 'text'], alpha=0.8)
        axs[0, i].set_title(f'{name}: argmax attention key location')
        axs[0, i].set_xlabel('tokens (M)'); axs[0, i].legend(fontsize=8, loc='center right')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/sink_location.png', dpi=130)

    # 4) loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (_, train, val) in runs.items():
        c = ARM_COLORS.get(name, None)
        if train:
            ax.plot([r['tokens_seen'] / 1e6 for r in train], [r['loss'] for r in train], color=c, alpha=0.35)
        if val:
            steps2tok = {r['step']: r['tokens_seen'] for r in train}
            vt = [steps2tok.get(r['step'], 0) / 1e6 for r in val]
            ax.plot(vt, [r['val_loss'] for r in val], 'o-', color=c, label=f'{name} val')
    ax.set_xlabel('tokens (M)'); ax.set_ylabel('loss'); ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 8)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/loss_curves.png', dpi=130)

    # lead-lag table: first probe step where each sink signature crosses its threshold
    print("\nFirst-crossing steps (lead-lag of sink signatures):")
    print(f"{'arm':10s} {'h_ratio>2':>10s} {'h_ratio>4':>10s} {'v_ratio<0.8':>12s} {'maxA0>0.2':>10s} {'sink.2>0':>10s} {'sink.3>0':>10s}")
    for name, (probes, _, _) in runs.items():
        def first(cond):
            for p in probes:
                if cond(p['summary']):
                    return str(p['step'])
            return '-'
        print(f"{name:10s} {first(lambda s: s['h_ratio_pos0'] > 2):>10s} "
              f"{first(lambda s: s['h_ratio_pos0'] > 4):>10s} "
              f"{first(lambda s: s['v_ratio_pos0'] < 0.8):>12s} "
              f"{first(lambda s: s['max_attn_pos0'] > 0.2):>10s} "
              f"{first(lambda s: s['sink_eps0.2'] > 0):>10s} "
              f"{first(lambda s: s['sink_eps0.3'] > 0):>10s}")

    # summary table
    print(f"{'arm':10s} {'steps':>7s} {'Mtok':>7s} {'sink.2':>7s} {'sink.3':>7s} {'max_a0':>7s} "
          f"{'v_ratio':>8s} {'h_ratio':>8s} {'val_loss':>9s}")
    for name, (probes, train, val) in runs.items():
        p = probes[-1]; s = p['summary']
        vl = val[-1]['val_loss'] if val else float('nan')
        print(f"{name:10s} {p['step']:7d} {p['tokens_seen']/1e6:7.1f} {s['sink_eps0.2']:7.3f} "
              f"{s['sink_eps0.3']:7.3f} {s['max_attn_pos0']:7.3f} {s['v_ratio_pos0']:8.3f} "
              f"{s['h_ratio_pos0']:8.3f} {vl:9.4f}")


if __name__ == '__main__':
    main(*sys.argv[1:])
