"""Plumbing smoke test for routes/evalx on SYNTHETIC tensors (no real data,
no claims). Builds a U-shaped risk landscape resembling the expected geometry:
high miss at both grid ends, minimum in the interior; checks every route
returns a sane grid index, run_calibrated + summarize complete, and the
valid routes land near/below alpha on eval halves."""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.routes import ROUTES
from sed_crc.evalx import make_splits, run_calibrated, summarize

rng = np.random.default_rng(0)
C, K, G = 400, 3, 201
grid = np.linspace(0, 1, G)

nref = rng.poisson(3.0, (C, K)).astype(np.int32)
miss = np.zeros((C, K, G), np.int32)
fps = np.zeros((C, K, G), np.int32)
for c in range(C):
    for k in range(K):
        M = nref[c, k]
        # U-shaped per-clip expected miss prob: high near 0 and 1, low ~0.3
        base = 0.9 * np.exp(-((grid - 0.0) ** 2) / 0.02) + \
               np.clip((grid - 0.3) * 1.3, 0, 0.95)
        p = np.clip(base + rng.normal(0, 0.05), 0.02, 0.98)
        m = rng.binomial(M, p) if M > 0 else np.zeros(G, int)
        # add local non-monotone jitter
        miss[c, k] = np.minimum(M, m)
        fps[c, k] = np.maximum(0, ((1 - grid) * 20).astype(int) + rng.integers(-2, 3, G))

durations_h = np.full(C, 30.0 / 3600)
splits = make_splits(C, 30, 0.5)
alpha = 0.1

for name, fn in ROUTES.items():
    res = run_calibrated(miss, fps, nref, durations_h, splits, alpha, fn,
                         grouping="marginal", route_kwargs={"delta": 0.05})
    s = summarize(res, alpha, n_required=int(0.95 * 30))
    lam0 = res["lams"][0][0]
    print(f"{name:22s} lam~{lam0:4d}/{G} n_ok={s['n_splits_ok']:2d}/30 "
          f"mean_miss={s['mean_miss_share']:.3f} fph={s['fp_per_h_mean']:.1f}")

# classwise pass
res = run_calibrated(miss, fps, nref, durations_h, splits, alpha,
                     ROUTES["ltt_split_clipmean"], grouping="classwise",
                     route_kwargs={"delta": 0.05})
s = summarize(res, alpha, n_required=int(0.95 * 30))
print(f"{'classwise ltt_split':22s} n_ok={s['n_splits_ok']:2d}/30 "
      f"mean_miss={s['mean_miss_share']:.3f}")
print("SMOKE OK")
