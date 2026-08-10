"""Feasibility floors OUTSIDE the resampled test data.

Two checks:
  1. Floors on the VALIDATION split (independent of every test-split resample,
     though spent on decoder tuning -- disclosed) for every cached
     validation tensor.
  2. Floors inside the 60% SELECTION pool of the holdout partition
     (rng 777, identical to holdout_confirm.py), i.e. the data a
     selection-pool-only protocol would have used to pick levels.

If both reproduce the test-split floor ranking and leave the chosen levels
(0.6/0.45/0.2) above the respective floors, the level choice would have been
the same under either audit; that closes the within-corpus half of the
"level selection is never exercised" concern.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

LEVELS = {"collar": 0.6, "intersect70": 0.45, "intersect50": 0.2}


def floors(miss, nref, idx=None):
    if idx is not None:
        miss, nref = miss[idx], nref[idx]
    tot = max(int(nref.sum()), 1)
    return {
        "common_floor": round(float(miss.sum(0).sum(0).min() / tot), 4),
        "classwise_floor": round(float(miss.sum(0).min(axis=1).sum() / tot), 4),
    }


def main():
    out = {"validation": {}, "selection_pool": {}, "level_check": {}}

    # --- 1. validation floors, every cached tensor
    for p in sorted(glob.glob(os.path.join(CACHE, "stats_validation_*.npz"))):
        stem = os.path.basename(p)[len("stats_validation_"):-len(".npz")]
        parts = stem.split("__")
        variant = parts[0]
        matching = parts[1] if len(parts) > 1 else "collar"
        if len(parts) > 2:
            continue
        _, miss, fps, nref, classes, grid = load_tensors(p)
        out["validation"][f"{matching}/{variant}"] = floors(miss, nref)
        print(f"val {matching}/{variant}:",
              out["validation"][f"{matching}/{variant}"], flush=True)

    # --- 2. selection-pool floors (holdout partition, rng 777, 60/40)
    canon = os.path.join(CACHE, "stats_test_medfilt.npz")
    _, miss0, _, _, _, _ = load_tensors(canon)
    C = miss0.shape[0]
    rng = np.random.default_rng(777)
    perm = rng.permutation(C)
    n_sel = int(round(0.6 * C))
    sel = np.sort(perm[:n_sel])
    for p in sorted(glob.glob(os.path.join(CACHE, "stats_test_*.npz"))):
        stem = os.path.basename(p)[len("stats_test_"):-len(".npz")]
        parts = stem.split("__")
        variant = parts[0]
        matching = parts[1] if len(parts) > 1 else "collar"
        if len(parts) > 2:
            continue
        _, miss, fps, nref, classes, grid = load_tensors(p)
        if miss.shape[0] != C:
            continue
        out["selection_pool"][f"{matching}/{variant}"] = floors(miss, nref, sel)
        print(f"pool {matching}/{variant}:",
              out["selection_pool"][f"{matching}/{variant}"], flush=True)

    # --- 3. do the chosen levels survive each audit?
    for src in ("validation", "selection_pool"):
        known = {k: v for k, v in out[src].items() if k.split("/")[0] in LEVELS}
        if not known:
            continue
        chk = {}
        for key, fl in known.items():
            matching = key.split("/")[0]
            lvl = LEVELS[matching]
            chk[key] = {"level": lvl,
                        "feasible": bool(fl["common_floor"] < lvl)}
        # alpha=0.1 infeasibility: is any floor below 0.1?
        min_floor = min(fl["common_floor"] for fl in known.values())
        out["level_check"][src] = {
            "per_config": chk,
            "min_common_floor": min_floor,
            "alpha01_infeasible": bool(min_floor > 0.1),
        }
        print(f"level check [{src}]: min floor {min_floor}, "
              f"alpha=0.1 infeasible: {min_floor > 0.1}", flush=True)

    with open(os.path.join(RESULTS, "floors_valpool.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/floors_valpool.json")


if __name__ == "__main__":
    main()
