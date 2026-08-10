"""Is CRC-C valid IN EXPECTATION here, or is its
guarantee itself broken by the missing feasible anchor?

The instrument study reports violation FRACTIONS; the in-expectation claim
E[R(lam_hat)] <= alpha is about the MEAN population risk over calibration
draws. This script measures exactly that: for CRC-C and naive CRC at the
operating points, over fresh 50/50 draws, the mean (and std) over draws of
the population clip-balanced risk and population pooled risk at the selected
threshold.

Reading: mean pop risk <= alpha  -> the route's in-expectation semantics
survive the transplant and the gate failure is purely semantics;
mean pop risk > alpha -> the missing-anchor adaptation breaks the guarantee
itself, and the row must be labeled out-of-assumption, not just in-expectation.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

N_DRAWS = 500

CONFIGS = [
    ("crcC-i20",  "csebb",   "intersect50", "crc_c_nonmono", 0.2),
    ("crcC-i45",  "csebb",   "intersect70", "crc_c_nonmono", 0.45),
    ("crcC-c60",  "csebb",   "collar",      "crc_c_nonmono", 0.6),
    ("crc-i20",   "csebb",   "intersect50", "crc_naive",     0.2),
    ("crc-c60m",  "medfilt", "collar",      "crc_naive",     0.6),
]


def main():
    out = {}
    for tag, variant, matching, route, alpha in CONFIGS:
        sfx = "" if matching == "collar" else f"__{matching}"
        _, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        C = miss.shape[0]
        miss_k, nref_k = miss.sum(1), nref.sum(1)
        tot_ref = nref_k.sum()
        pop_pooled = miss_k.sum(0) / tot_ref
        mm = nref_k > 0
        pop_clip = (miss_k[mm] / nref_k[mm][:, None]).mean(0)
        route_fn = ROUTES[route]
        half = C // 2

        t0 = time.time()
        clip_r, pooled_r = [], []
        for si in range(N_DRAWS):
            rng = np.random.default_rng(5_000_000 + si)
            cal = rng.permutation(C)[:half]
            g = route_fn(miss_k[cal], nref_k[cal], alpha, delta=0.05,
                         rng=np.random.default_rng(6_000_000 + si))
            clip_r.append(pop_clip[g])
            pooled_r.append(pop_pooled[g])
        clip_r, pooled_r = np.array(clip_r), np.array(pooled_r)
        out[tag] = {
            "config": dict(variant=variant, matching=matching, route=route,
                           alpha=alpha, n_draws=N_DRAWS),
            "mean_pop_clipbalanced_risk": round(float(clip_r.mean()), 4),
            "std_pop_clipbalanced_risk": round(float(clip_r.std()), 4),
            "mean_pop_pooled_risk": round(float(pooled_r.mean()), 4),
            "std_pop_pooled_risk": round(float(pooled_r.std()), 4),
            "viol_frac_clipbalanced": round(float((clip_r > alpha).mean()), 4),
            "viol_frac_pooled": round(float((pooled_r > alpha).mean()), 4),
            "in_expectation_holds_clipbalanced": bool(clip_r.mean() <= alpha),
            "seconds": round(time.time() - t0, 1),
        }
        print(tag, out[tag], flush=True)

    with open(os.path.join(RESULTS, "crcc_meanrisk.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/crcc_meanrisk.json")


if __name__ == "__main__":
    main()
