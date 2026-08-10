"""Paired split-level CIs for the key
efficiency comparisons. Routes share split seeds, so the honest statistic for
"A is cheaper than B" is the per-split FP/h difference, not two marginal
means over correlated splits.
"""
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

PAIRS = [
    ("i20-200cand", "csebb", "intersect50", 0.2,
     "ltt_split_clipmean", "ltt_bonf_cm200"),
    ("i20-50cand", "csebb", "intersect50", 0.2,
     "ltt_split_cm50", "ltt_bonf_clipmean"),
    ("i20-max", "csebbmax", "intersect50", 0.2,
     "ltt_split_clipmean", "ltt_bonf_clipmean"),
]


def main():
    out = {}
    for tag, variant, matching, alpha, route_a, route_b in PAIRS:
        sfx = "" if matching == "collar" else f"__{matching}"
        clip_ids, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        dur = load_durations("test")
        durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
        C = miss.shape[0]
        miss_k, fps_k, nref_k = miss.sum(1), fps.sum(1), nref.sum(1)
        fa, fb = ROUTES[route_a], ROUTES[route_b]

        diffs, fpa, fpb = [], [], []
        for si, (cal, ev) in enumerate(make_splits(C, 100, 0.5)):
            ga = fa(miss_k[cal], nref_k[cal], alpha, delta=0.05,
                    rng=np.random.default_rng(10_000 + si))
            gb = fb(miss_k[cal], nref_k[cal], alpha, delta=0.05,
                    rng=np.random.default_rng(10_000 + si))
            ha = fps_k[ev, ga].sum() / durations_h[ev].sum()
            hb = fps_k[ev, gb].sum() / durations_h[ev].sum()
            fpa.append(ha)
            fpb.append(hb)
            diffs.append(ha - hb)
        d = np.array(diffs)
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        out[tag] = {
            "config": dict(variant=variant, matching=matching, alpha=alpha,
                           route_a=route_a, route_b=route_b, n_splits=n),
            "fph_a_mean": round(float(np.mean(fpa)), 1),
            "fph_b_mean": round(float(np.mean(fpb)), 1),
            "mean_diff_a_minus_b": round(float(d.mean()), 1),
            "ci95_normal": [round(float(d.mean() - 1.96 * se), 1),
                            round(float(d.mean() + 1.96 * se), 1)],
            "ci95_percentile": [round(float(np.percentile(d, 2.5)), 1),
                                round(float(np.percentile(d, 97.5)), 1)],
            "frac_splits_a_cheaper": round(float((d < 0).mean()), 3),
        }
        print(tag, out[tag], flush=True)

    with open(os.path.join(RESULTS, "paired_diffs.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/paired_diffs.json")


if __name__ == "__main__":
    main()
