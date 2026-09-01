"""Pooled vs clip-balanced gap at the key operating configurations.

For each config, over the 100 protocol splits (same seeds and inner rngs as
run_experiment): select the marginal threshold on the calibration half with
the config's route, then record BOTH evaluation-half functionals -- the
pooled miss share (sum m_c / sum M_c) and the clip-balanced mean miss rate
(mean over clips of m_c/M_c) -- and their per-split difference. Quantifies
how far apart the two encodings of the risk sit at deployed thresholds."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES
from sed_crc.evalx import make_splits

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

CONFIGS = [
    ("i20_champion", "csebbmax", "intersect50", 0.2,  "ltt_split_clipmean"),
    ("i20_rcps",     "csebbmax", "intersect50", 0.2,  "rcps_fixedseq"),
    ("i45_champion", "csebb",    "intersect70", 0.45, "ltt_bonf_clipmean"),
    ("c60_champion", "csebb",    "collar",      0.6,  "ltt_bonf_clipmean"),
]


def main():
    out = {}
    for tag, variant, matching, alpha, route in CONFIGS:
        sfx = "" if matching == "collar" else f"__{matching}"
        clip_ids, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        C, K, G = miss.shape
        miss_k, nref_k = miss.sum(1), nref.sum(1)     # [C,G], [C]
        route_fn = ROUTES[route]

        pooled, clipbal, gaps, lams = [], [], [], []
        for si, (cal, ev) in enumerate(make_splits(C, 100, 0.5)):
            rng = np.random.default_rng(10_000 + si)
            g = route_fn(miss_k[cal], nref_k[cal], alpha, delta=0.05, rng=rng)
            evm, evn = miss_k[ev, g], nref_k[ev]
            pl = float(evm.sum() / max(evn.sum(), 1))
            pos = evn > 0
            cb = float((evm[pos] / evn[pos]).mean())
            pooled.append(pl)
            clipbal.append(cb)
            gaps.append(pl - cb)
            lams.append(float(grid[g]))
        gaps = np.array(gaps)
        out[tag] = {
            "config": dict(variant=variant, matching=matching, alpha=alpha,
                           route=route, grouping="marginal"),
            "pooled_mean": round(float(np.mean(pooled)), 4),
            "clipbal_mean": round(float(np.mean(clipbal)), 4),
            "gap_mean": round(float(gaps.mean()), 4),
            "gap_std": round(float(gaps.std()), 4),
            "gap_ci95_percentile": [round(float(np.percentile(gaps, 2.5)), 4),
                                    round(float(np.percentile(gaps, 97.5)), 4)],
            "gap_min": round(float(gaps.min()), 4),
            "gap_max": round(float(gaps.max()), 4),
            "frac_splits_gap_positive": round(float((gaps > 0).mean()), 3),
            "mean_lam": round(float(np.mean(lams)), 3),
        }
        print(tag, json.dumps(out[tag]), flush=True)

    with open(os.path.join(RESULTS, "gap_functionals.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/gap_functionals.json")


if __name__ == "__main__":
    main()
