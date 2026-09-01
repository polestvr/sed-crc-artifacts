"""Explicit arithmetic behind the pooled-cap vacuity claim.

Reports, per operating point: the mean events per clip on the test split,
the best feasibility floor for the point's matching rule, the largest
attainable mean pooled excess |X-bar| = (alpha - floor) * mean(M), the mean
cap kappa and capped-subpopulation size n_kappa over the protocol splits
(replicating the route's inner A/B rng), and the resulting Hoeffding
rejection requirement kappa * sqrt(ln(1/delta_test) / (2 n_kappa))."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors
from sed_crc.routes import _split
from sed_crc.evalx import make_splits

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

CONFIGS = [
    ("i20", "csebbmax", "intersect50", 0.2,  "intersect50/csebb"),
    ("i45", "csebb",    "intersect70", 0.45, "intersect70/csebb"),
    ("c60", "csebb",    "collar",      0.6,  "collar/csebb"),
]
DELTA = 0.05
TAIL_FRAC = 0.2


def main():
    floors = json.load(open(os.path.join(RESULTS, "feasibility_floors.json")))
    out = {}
    for tag, variant, matching, alpha, floor_key in CONFIGS:
        sfx = "" if matching == "collar" else f"__{matching}"
        clip_ids, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        C = miss.shape[0]
        nref_k = nref.sum(1)                       # [C]
        mean_M = float(nref_k.sum() / C)
        floor = float(floors[floor_key]["pooled_floor"])

        kappas, n_kappas = [], []
        for si, (cal, ev) in enumerate(make_splits(C, 100, 0.5)):
            rng = np.random.default_rng(10_000 + si)
            A, Bh = _split(rng, len(cal))
            kappa = max(int(nref_k[cal][A].max()), 1)
            n_kappa = int((nref_k[cal][Bh] <= kappa).sum())
            kappas.append(kappa)
            n_kappas.append(n_kappa)
        kbar = float(np.mean(kappas))
        nkbar = float(np.mean(n_kappas))
        delta_test = DELTA * (1.0 - TAIL_FRAC)
        required = kbar * np.sqrt(np.log(1.0 / delta_test) / (2.0 * nkbar))
        out[tag] = {
            "config": dict(variant=variant, matching=matching, alpha=alpha),
            "n_clips": C,
            "n_events": int(nref_k.sum()),
            "mean_events_per_clip": round(mean_M, 3),
            "best_floor": round(floor, 4),
            "attainable_signal": round((alpha - floor) * mean_M, 3),
            "kappa_mean": round(kbar, 1),
            "n_kappa_mean": round(nkbar, 1),
            "delta_test": delta_test,
            "required_excess": round(float(required), 3),
        }
        print(tag, json.dumps(out[tag]), flush=True)

    with open(os.path.join(RESULTS, "vacuity_numbers.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/vacuity_numbers.json")


if __name__ == "__main__":
    main()
