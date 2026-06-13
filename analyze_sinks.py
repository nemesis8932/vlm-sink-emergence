"""Build emergence figures + summary table from runs/*/probes.jsonl and train_log.jsonl.

Run from the repo root:  python analyze_sinks.py [out_dir]

baseline + baseline_ext are merged into a single 'baseline' trace.
"""

import json
import os
import sys
import glob
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ARM_COLORS  = {'baseline': 'tab:blue', 'textinit': 'tab:orange',
               'g1gate': 'tab:green', 'sigmoid': 'tab:red'}
ARM_ORDER   = ['baseline', 'textinit', 'g1gate', 'sigmoid']
S1_BOUNDARY = 174.4   # M tokens — end of Session 1 baseline; start of extension
N_IMG       = 49      # image tokens before text


# ── loaders ──────────────────────────────────────────────────────────────────

def load_probes(run_dir):
    path = os.path.join(run_dir, 'probes.jsonl')
    probes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))
    return probes


def load_train_log(run_dir):
    train, val = [], []
    lp = os.path.join(run_dir, 'train_log.jsonl')
    if not os.path.exists(lp):
        return train, val
    with open(lp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (val if 'val_loss' in r else train).append(r)
    return train, val


def dedup_probes(probes):
    seen = {}
    for p in probes:
        seen[p['step']] = p
    return [seen[s] for s in sorted(seen)]


def load_all_runs(base='runs'):
    runs = {}
    for d in sorted(glob.glob(f'{base}/*/')):
        name = d.rstrip('/').split('/')[-1]
        probe_path = os.path.join(d, 'probes.jsonl')
        if os.path.exists(probe_path):
            runs[name] = (load_probes(d), *load_train_log(d))
    # merge baseline_ext into baseline
    if 'baseline' in runs and 'baseline_ext' in runs:
        bp, bt, bv = runs['baseline']
        ep, et, ev = runs['baseline_ext']
        merged_probes = dedup_probes(bp + ep)
        # for train log: baseline_ext starts at resume_step so concat is fine
        merged_train = dedup_by_step(bt + et)
        merged_val   = dedup_by_step(bv + ev)
        runs['baseline'] = (merged_probes, merged_train, merged_val)
        del runs['baseline_ext']
    # dedup all
    for name in list(runs.keys()):
        p, t, v = runs[name]
        runs[name] = (dedup_probes(p), dedup_by_step(t), dedup_by_step(v))
    # return in canonical order
    ordered = {}
    for name in ARM_ORDER:
        if name in runs:
            ordered[name] = runs[name]
    for name in runs:  # any extras
        if name not in ordered:
            ordered[name] = runs[name]
    return ordered


def dedup_by_step(records):
    seen = {}
    for r in records:
        seen[r.get('step', id(r))] = r
    return [seen[k] for k in sorted(seen)]


# ── helpers ───────────────────────────────────────────────────────────────────

def toks(probes):
    return [p['tokens_seen'] / 1e6 for p in probes]


def summ(probes, key):
    return [p['summary'][key] for p in probes]


def vline(ax, x=S1_BOUNDARY, **kw):
    defaults = dict(color='black', ls='--', lw=0.8, alpha=0.5)
    defaults.update(kw)
    ax.axvline(x, **defaults)


def annotate_s1(ax, y_frac=0.97, fontsize=7):
    ylim = ax.get_ylim()
    y = ylim[0] + (ylim[1] - ylim[0]) * y_frac
    ax.text(S1_BOUNDARY + 2, y, 'S1 end', fontsize=fontsize,
            va='top', ha='left', color='black', alpha=0.6)


def finish(ax, xlabel='tokens seen (M)', legend=True, grid=True):
    ax.set_xlabel(xlabel)
    if legend:
        ax.legend(fontsize=8)
    if grid:
        ax.grid(alpha=0.3)


# ── figures ───────────────────────────────────────────────────────────────────

def fig_emergence_curves(runs, out_dir, current_tok=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_sink, ax_attn, ax_vr, ax_hr = axes.flat

    for name, (probes, _, _) in runs.items():
        c   = ARM_COLORS.get(name, 'gray')
        t   = toks(probes)
        # panel A — Sink fraction
        for eps, ls, a in [('0.2', ':', 0.7), ('0.3', '-', 1.0), ('0.4', '--', 0.6)]:
            lbl = f'{name}  ε={eps}' if eps == '0.3' else None
            ax_sink.plot(t, summ(probes, f'sink_eps{eps}'), ls, color=c, label=lbl, alpha=a, lw=1.4)
        # panel B — mean + max attn
        ax_attn.plot(t, summ(probes, 'mean_attn_pos0'), color=c, label=f'{name} mean', lw=1.4)
        ax_attn.plot(t, summ(probes, 'max_attn_pos0'),  color=c, ls='--', alpha=0.55, lw=1.0)
        # panel C — v_ratio
        ax_vr.plot(t, summ(probes, 'v_ratio_pos0'), color=c, label=name, lw=1.4)
        # panel D — h_ratio (log scale — textinit hits 42×)
        ax_hr.plot(t, summ(probes, 'h_ratio_pos0'), color=c, label=name, lw=1.4)

    for ax in axes.flat:
        vline(ax)

    ax_sink.set_title('Sink$^{ε}_1$  — fraction of (layer, head) pairs with mean attn→pos0 > ε\n'
                      '(dotted=0.2, solid=0.3, dashed=0.4)', fontsize=9)
    ax_attn.set_title('Attention → pos0 :  mean (solid) / max-head (dashed)', fontsize=9)
    ax_vr.set_title('Value-norm ratio  ‖v(pos0)‖ / ‖v(rest)‖  (drain < 1,  anti-drain > 1)', fontsize=9)
    ax_hr.set_title('Residual-norm ratio pos0/rest  —  massive activations  (log scale)', fontsize=9)
    ax_hr.set_yscale('log')
    ax_hr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.1f}×'))
    ax_vr.axhline(1.0, color='black', lw=0.6, alpha=0.4)

    for ax in axes.flat:
        finish(ax)

    if current_tok is not None:
        for ax in axes.flat:
            ax.axvline(current_tok, color='purple', ls=':', lw=1.0, alpha=0.7)

    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    cur_str = f'  ·  baseline at {current_tok:.0f}M tok' if current_tok else ''
    fig.suptitle(f'VLM Sink Emergence — interim  ({now}{cur_str})', fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/emergence_curves.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved emergence_curves.png')


def fig_layer_heatmaps(runs, out_dir):
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, (name, (probes, _, _)) in enumerate(runs.items()):
        # max-over-heads attn→pos0 per layer, over time
        M = np.array([[max(l['attn_to_pos0']) for l in p['layers']] for p in probes]).T
        t = toks(probes)
        extent = [t[0], t[-1], 0, M.shape[0]]
        im = axs[0, i].imshow(M, aspect='auto', origin='lower', cmap='viridis',
                               vmin=0, vmax=1, extent=extent)
        axs[0, i].set_title(f'{name}: max-head attn→pos0 per layer', fontsize=9)
        axs[0, i].set_xlabel('tokens (M)')
        axs[0, i].set_ylabel('layer')
        if t[-1] > S1_BOUNDARY + 10:
            axs[0, i].axvline(S1_BOUNDARY, color='white', ls='--', lw=0.8, alpha=0.7)
        plt.colorbar(im, ax=axs[0, i])
    fig.tight_layout()
    fig.savefig(f'{out_dir}/layer_heatmaps.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved layer_heatmaps.png')


def fig_v_heatmaps(runs, out_dir):
    """Per-layer mean v_ratio over training time."""
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, (name, (probes, _, _)) in enumerate(runs.items()):
        M = np.array([[l['v_norm_pos0'] / max(l['v_norm_rest'], 1e-8)
                       for l in p['layers']] for p in probes]).T
        t = toks(probes)
        vmax = min(np.percentile(M, 98), 3.0)
        extent = [t[0], t[-1], 0, M.shape[0]]
        im = axs[0, i].imshow(M, aspect='auto', origin='lower', cmap='RdBu_r',
                               vmin=0, vmax=vmax, extent=extent)
        axs[0, i].set_title(f'{name}: v_ratio per layer', fontsize=9)
        axs[0, i].set_xlabel('tokens (M)')
        axs[0, i].set_ylabel('layer')
        if t[-1] > S1_BOUNDARY + 10:
            axs[0, i].axvline(S1_BOUNDARY, color='black', ls='--', lw=0.8, alpha=0.6)
        plt.colorbar(im, ax=axs[0, i])
    fig.tight_layout()
    fig.savefig(f'{out_dir}/v_heatmaps.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved v_heatmaps.png')


def fig_sink_location(runs, out_dir):
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for i, (name, (probes, _, _)) in enumerate(runs.items()):
        t = toks(probes)
        f0, fimg, ftxt = [], [], []
        for p in probes:
            am = [k for l in p['layers'] for k in l['argmax_key']]
            N = len(am)
            f0.append(sum(k == 0 for k in am) / N)
            fimg.append(sum(0 < k < N_IMG for k in am) / N)
            ftxt.append(sum(k >= N_IMG for k in am) / N)
        axs[0, i].stackplot(t, f0, fimg, ftxt,
                             labels=['pos 0 (img token 0)', 'image 1..48', 'text'],
                             colors=['tab:red', 'tab:blue', 'tab:gray'], alpha=0.75)
        axs[0, i].set_title(f'{name}: argmax-key location fraction', fontsize=9)
        axs[0, i].set_xlabel('tokens (M)')
        axs[0, i].legend(fontsize=7, loc='upper right')
        if t[-1] > S1_BOUNDARY + 10:
            axs[0, i].axvline(S1_BOUNDARY, color='black', ls='--', lw=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/sink_location.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved sink_location.png')


def fig_loss(runs, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (_, train, val) in runs.items():
        c = ARM_COLORS.get(name, 'gray')
        if train:
            ax.plot([r['tokens_seen'] / 1e6 for r in train],
                    [r['loss'] for r in train], color=c, alpha=0.25, lw=0.7)
        if val:
            tok_map = {r['step']: r['tokens_seen'] for r in train}
            vt  = [tok_map.get(r['step'], 0) / 1e6 for r in val]
            vls = [r['val_loss'] for r in val]
            ax.plot(vt, vls, 'o-', color=c, label=f'{name}  val={vls[-1]:.3f}', lw=1.4, ms=4)
    vline(ax)
    ax.set_xlabel('tokens (M)')
    ax.set_ylabel('loss')
    ax.set_ylim(0, 8)
    ax.set_title('Training (faint) + validation loss')
    finish(ax)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/loss_curves.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved loss_curves.png')


def fig_baseline_zoom(runs, out_dir):
    """Detailed view of the extended baseline — all 3 signatures on one plot."""
    if 'baseline' not in runs:
        return
    probes, _, _ = runs['baseline']
    t  = toks(probes)
    s2 = [p['summary'] for p in probes]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, [s['sink_eps0.2'] for s in s2], ':', label='ε=0.2', lw=1.3)
    axes[0].plot(t, [s['sink_eps0.3'] for s in s2], '-', label='ε=0.3', lw=1.3)
    axes[0].plot(t, [s['sink_eps0.4'] for s in s2], '--', label='ε=0.4', lw=1.3)
    axes[0].plot(t, [s['mean_attn_pos0'] for s in s2], color='black', lw=1.0,
                 alpha=0.5, label='mean attn→pos0')
    axes[0].set_ylabel('Sink fraction / mean attn')
    axes[0].set_title('Baseline (softmax, scratch) — extended to 1B  [INTERIM]', fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(t, [s['v_ratio_pos0'] for s in s2], color='tab:blue', lw=1.4)
    axes[1].axhline(1.0, color='black', lw=0.6, alpha=0.4)
    axes[1].set_ylabel('v_ratio  ‖v(pos0)‖/‖v(rest)‖')

    axes[2].plot(t, [s['h_ratio_pos0'] for s in s2], color='tab:red', lw=1.4)
    axes[2].axhline(1.0, color='black', lw=0.6, alpha=0.4)
    axes[2].set_ylabel('h_ratio  (massive activations)')
    axes[2].set_xlabel('tokens seen (M)')

    for ax in axes:
        vline(ax)
        ax.grid(alpha=0.3)

    # annotate where Session 1 ended
    axes[0].text(S1_BOUNDARY + 3, axes[0].get_ylim()[1] * 0.9,
                 '← S1 end', fontsize=8, va='top', color='black', alpha=0.6)

    fig.tight_layout()
    fig.savefig(f'{out_dir}/baseline_zoom.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved baseline_zoom.png')


# ── tables ────────────────────────────────────────────────────────────────────

def print_summary(runs):
    print('\n' + '='*90)
    print('SUMMARY TABLE  (last probe per arm)')
    print('='*90)
    hdr = f"{'arm':12s} {'steps':>7s} {'Mtok':>8s} {'sink.2':>7s} {'sink.3':>7s} " \
          f"{'mean_a0':>8s} {'max_a0':>8s} {'v_ratio':>8s} {'h_ratio':>8s} {'val_loss':>9s}"
    print(hdr)
    print('-'*90)
    for name, (probes, train, val) in runs.items():
        p   = probes[-1]
        s   = p['summary']
        vl  = val[-1]['val_loss'] if val else float('nan')
        print(f"{name:12s} {p['step']:7d} {p['tokens_seen']/1e6:8.1f} "
              f"{s['sink_eps0.2']:7.3f} {s['sink_eps0.3']:7.3f} "
              f"{s['mean_attn_pos0']:8.3f} {s['max_attn_pos0']:8.3f} "
              f"{s['v_ratio_pos0']:8.3f} {s['h_ratio_pos0']:8.3f} {vl:9.4f}")

    print('\n' + '='*90)
    print('FIRST-CROSSING TABLE  (first step/Mtok where signature exceeds threshold)')
    print('='*90)
    hdr2 = f"{'arm':12s} {'h>2×':>10s} {'h>4×':>10s} {'v<0.8':>10s} {'maxA>0.2':>10s} " \
           f"{'sink.2>0':>10s} {'sink.3>0':>10s}"
    print(hdr2)
    print('-'*90)
    for name, (probes, _, _) in runs.items():
        def first(cond):
            for p in probes:
                if cond(p['summary']):
                    return f"{p['step']}  ({p['tokens_seen']/1e6:.0f}M)"
            return '-'
        print(f"{name:12s} "
              f"{first(lambda s: s['h_ratio_pos0'] > 2):>10s}  "
              f"{first(lambda s: s['h_ratio_pos0'] > 4):>10s}  "
              f"{first(lambda s: s['v_ratio_pos0'] < 0.8):>10s}  "
              f"{first(lambda s: s['max_attn_pos0'] > 0.2):>10s}  "
              f"{first(lambda s: s['sink_eps0.2'] > 0):>10s}  "
              f"{first(lambda s: s['sink_eps0.3'] > 0):>10s}")
    print('='*90)


# ── main ──────────────────────────────────────────────────────────────────────

def main(out_dir='analysis'):
    os.makedirs(out_dir, exist_ok=True)
    runs = load_all_runs()
    if not runs:
        sys.exit('no runs found under runs/')

    baseline_tok = None
    if 'baseline' in runs:
        p = runs['baseline'][0][-1]
        baseline_tok = p['tokens_seen'] / 1e6

    print(f'Arms loaded: {list(runs.keys())}')
    for name, (probes, _, _) in runs.items():
        print(f'  {name}: {len(probes)} probes  '
              f'steps {probes[0]["step"]}→{probes[-1]["step"]}  '
              f'({probes[-1]["tokens_seen"]/1e6:.1f}M tok)')
    if baseline_tok:
        print(f'  baseline current: {baseline_tok:.1f}M / 1000M  '
              f'({baseline_tok/10:.1f}% of target)')

    print('\nGenerating figures...')
    fig_emergence_curves(runs, out_dir, current_tok=baseline_tok)
    fig_layer_heatmaps(runs, out_dir)
    fig_v_heatmaps(runs, out_dir)
    fig_sink_location(runs, out_dir)
    fig_loss(runs, out_dir)
    fig_baseline_zoom(runs, out_dir)

    print_summary(runs)
    print(f'\nAll figures written to {out_dir}/')


if __name__ == '__main__':
    main(*sys.argv[1:])
