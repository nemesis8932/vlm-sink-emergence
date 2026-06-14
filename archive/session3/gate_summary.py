"""Machine-readable acceptance-gate numbers for Session 2. Prose REPORT is written by hand."""
import json, os, glob

def load(run):
    p = os.path.join('runs', run, 'probes.jsonl')
    if not os.path.exists(p): return []
    seen = {}
    for line in open(p):
        r = json.loads(line); seen[r['step']] = r
    return [seen[s] for s in sorted(seen)]

def at_tokens(probes, target):
    """probe nearest to (and <=) target tokens, else the last available."""
    cand = [p for p in probes if p['tokens_seen'] <= target * 1.02]
    return (cand or probes)[-1] if probes else None

def first_cross(probes, key, thr, lt=False):
    for p in probes:
        v = p['summary'][key]
        if (v < thr) if lt else (v > thr):
            return p['tokens_seen']
    return None

print("="*78); print("SESSION-2 ACCEPTANCE GATES"); print("="*78)

# ---- Gate A: does the baseline concentration sink ever appear by 1.0B? ----
base = load('baseline') + load('baseline_ext')          # full seed-0 curve, 0 -> 1.0B
base = sorted({p['tokens_seen']: p for p in base}.values(), key=lambda p: p['tokens_seen']) \
       if base else []
print("\n[Gate A] baseline seed-0, full curve to 1.0B")
if base:
    last = base[-1]; s = last['summary']
    fc = first_cross(base, 'max_attn_pos0', 0.2)
    print(f"  final tokens: {last['tokens_seen']/1e6:.1f}M  step {last['step']}")
    print(f"  Sink^0.2_1 @final: {s['sink_eps0.2']:.4f}   Sink^0.3_1 @final: {s['sink_eps0.3']:.4f}")
    print(f"  max mean-attn->pos0 @final: {s['max_attn_pos0']:.4f}")
    print(f"  first tokens any head crosses mean-attn->pos0 > 0.2: "
          f"{(str(fc/1e6)+'M') if fc else 'NEVER (no concentration sink by 1.0B)'}")
    print(f"  v_ratio @final: {s['v_ratio_pos0']:.3f}   h_ratio @final: {s['h_ratio_pos0']:.3f}")
else:
    print("  (no baseline probes found)")

# ---- Gate B: seed-0 vs seed-1 agreement at matched 100M budget ----
print("\n[Gate B] seed-0 (Session-1) vs seed-1, matched ~100M tokens")
hdr = f"  {'arm':9s} {'seed':4s} {'Mtok':>6s} {'sink.2':>7s} {'mean_a0':>8s} {'v_ratio':>8s} {'h_ratio':>8s}"
print(hdr)
for arm in ['baseline', 'textinit', 'g1gate', 'sigmoid']:
    s0 = load(arm); s1 = load(f'{arm}_seed1')
    # match at the smaller of the two arms' final token counts
    tgt = min([p['tokens_seen'] for p in (s0[-1:]+s1[-1:])] or [100e6])
    for tag, pr in [('s0', s0), ('s1', s1)]:
        p = at_tokens(pr, tgt) if pr else None
        if not p: print(f"  {arm:9s} {tag:4s}  (missing)"); continue
        z = p['summary']
        print(f"  {arm:9s} {tag:4s} {p['tokens_seen']/1e6:6.1f} {z['sink_eps0.2']:7.3f} "
              f"{z['mean_attn_pos0']:8.4f} {z['v_ratio_pos0']:8.3f} {z['h_ratio_pos0']:8.3f}")
print("\nGate B reading: sigmoid expected large+robust (sink.2~0.8 vs ~0); g1gate is the fragile "
      "one (sink.2~0.004) -> if seeds disagree, do NOT trust it (needs seed-3). n=2 bar = "
      "tight agreement + large effect.")
