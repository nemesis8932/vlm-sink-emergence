# Session-3 cost log (approx)

Instance 40436103 RTX 4090 @ $0.7037/hr. GPU wall-clock per run (from train_log 'wall'):

- baseline            1.60 h  ~$ 1.13
- baseline_ext        6.22 h  ~$ 4.38
- baseline_seed1      0.94 h  ~$ 0.66
- g1gate              1.00 h  ~$ 0.70
- g1gate_seed1        0.98 h  ~$ 0.69
- sigmoid             1.00 h  ~$ 0.70
- sigmoid_seed1       0.99 h  ~$ 0.70
- textinit            0.55 h  ~$ 0.39
- textinit_seed1      0.94 h  ~$ 0.66

**Training wall total ≈ 14.21 GPU-h ≈ $10.00** (excludes setup/probe/idle).
RF (fresh-1B) NOT run -> ~$7 saved (see RF-BLOCKED.md).
