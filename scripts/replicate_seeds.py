"""Route-ranking replication across the three additional training
runs (seed caches). For each seed and each key (variant, matching, route,
alpha) configuration: 100-split gate count and mean FP/h. Answers whether the
route ranking (not just the floors) is checkpoint-stable."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES
from sed_crc.evalx import make_splits, run_calibrated, summarize
from sed_crc.gt import load_durations

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

CONFIGS = [
    ("csebb", "intersect50", "ltt_split_clipmean", "marginal", 0.2),
    ("csebb", "intersect50", "ltt_bonf_clipmean", "marginal", 0.2),
    ("csebb", "intersect50", "rcps_fixedseq", "marginal", 0.2),
    ("csebb", "intersect50", "crc_naive", "marginal", 0.2),
    ("csebb", "intersect70", "ltt_bonf_clipmean", "marginal", 0.45),
    ("csebb", "intersect70", "ltt_split_clipmean", "marginal", 0.45),
    ("csebb", "collar", "ltt_bonf_clipmean", "marginal", 0.6),
    ("medfilt", "collar", "ltt_bonf_clipmean", "marginal", 0.6),
    ("medfilt", "intersect50", "ltt_bonf_clipmean", "marginal", 0.2),
]


def main():
    dur = load_durations("test")
    out = {}
    for seed in ("seed2", "seed3", "seed4"):
        cache = os.path.expanduser(f"~/sed-crc-work/cache_{seed}")
        rows = []
        for variant, matching, route, grouping, alpha in CONFIGS:
            sfx = "" if matching == "collar" else f"__{matching}"
            p = os.path.join(cache, f"stats_test_{variant}{sfx}.npz")
            if not os.path.exists(p):
                continue
            clip_ids, miss, fps, nref, classes, grid = load_tensors(p)
            durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
            splits = make_splits(miss.shape[0], 100, 0.5)
            res = run_calibrated(miss, fps, nref, durations_h, splits, alpha,
                                 ROUTES[route], grouping=grouping,
                                 route_kwargs={"delta": 0.05})
            s = summarize(res, alpha)
            rows.append(dict(variant=variant, matching=matching, route=route,
                             alpha=alpha, n_ok=s["n_splits_ok"],
                             miss=round(s["mean_miss_share"], 4),
                             fph=round(s["fp_per_h_mean"], 1)))
            print(seed, variant, matching, route, alpha, s["n_splits_ok"],
                  round(s["fp_per_h_mean"], 1), flush=True)
        out[seed] = rows
    with open(os.path.join(RESULTS, "seed_replication.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/seed_replication.json")


if __name__ == "__main__":
    main()
