"""DCASE-style operating-point baseline: per-class thresholds maximizing
event-level F1 on the VALIDATION split, frozen, then scored on the test
split under the standard 100-split protocol (evaluation halves only, no
calibration-half use) -- the common practice of tuning class-wise
thresholds for a benchmark score, with no statistical guarantee.

Writes results/exp_<EXP-ID>.json in the run_experiment format so the
table generator picks the rows up as uncertified references."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import load_tensors
from sed_crc.evalx import make_splits, summarize

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

RUNS = [
    ("F1V-c60-med", "medfilt",  "collar",      0.6),
    ("F1V-c60-seb", "csebb",    "collar",      0.6),
    ("F1V-i45-med", "medfilt",  "intersect70", 0.45),
    ("F1V-i45-seb", "csebb",    "intersect70", 0.45),
    ("F1V-i20-med", "medfilt",  "intersect50", 0.2),
    ("F1V-i20-max", "csebbmax", "intersect50", 0.2),
]


def f1_thresholds(variant, matching):
    """Per-class grid indices maximizing event-level F1 on validation
    (ties resolved toward the larger threshold)."""
    sfx = "" if matching == "collar" else f"__{matching}"
    _, vmiss, vfps, vnref, classes, grid = load_tensors(
        os.path.join(CACHE, f"stats_validation_{variant}{sfx}.npz"))
    K = vmiss.shape[1]
    lam_idx = np.zeros(K, np.int64)
    for k in range(K):
        fn = vmiss[:, k, :].sum(0)                 # [G]
        fp = vfps[:, k, :].sum(0)                  # [G]
        tp = vnref[:, k].sum() - fn                # [G]
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(2 * tp + fp + fn > 0,
                          2 * tp / (2 * tp + fp + fn), 0.0)
        lam_idx[k] = int(np.flatnonzero(f1 == f1.max())[-1])
    return lam_idx


def main():
    for exp_id, variant, matching, alpha in RUNS:
        t0 = time.time()
        lam_idx = f1_thresholds(variant, matching)
        sfx = "" if matching == "collar" else f"__{matching}"
        clip_ids, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        dur = load_durations("test")
        durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
        C, K, G = miss.shape
        res = {"miss_share": [], "fp_per_h": [], "lams": []}
        for cal_idx, ev_idx in make_splits(C, 100, 0.5):
            ev_miss = miss[ev_idx][:, np.arange(K), lam_idx]
            ev_fps = fps[ev_idx][:, np.arange(K), lam_idx]
            ev_nref = nref[ev_idx]
            res["miss_share"].append(float(ev_miss.sum() / max(ev_nref.sum(), 1)))
            res["fp_per_h"].append(float(ev_fps.sum() / durations_h[ev_idx].sum()))
            res["lams"].append(lam_idx.copy())
        s = summarize(res, alpha)
        out = {
            "exp_id": exp_id,
            "config": {"exp_id": exp_id, "variant": variant,
                       "route": "valtuned_f1", "margin": 0.0,
                       "grouping": "classwise", "matching": matching,
                       "alpha": alpha, "delta": 0.05, "cal_frac": 0.5,
                       "n_splits": 100, "seed_base": 0},
            "summary": s,
            "val_f1_lams": {classes[k]: float(grid[lam_idx[k]])
                            for k in range(K)},
            "runtime_s": round(time.time() - t0, 1),
        }
        with open(os.path.join(RESULTS, f"exp_{exp_id}.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps({"exp_id": exp_id, **s}), flush=True)


if __name__ == "__main__":
    main()
