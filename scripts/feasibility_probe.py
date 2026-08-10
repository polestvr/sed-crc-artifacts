"""Feasibility floors: minimal achievable pooled event miss share over the
threshold grid, for different event-matching rules, on the full test split.

Rules probed (all via untouched sed_scores_eval machinery):
  collar      - sed_eval rule: onset 0.2 s, offset max(0.2 s, 0.5*dur) [cached]
  onset-only  - onset 0.2 s only (offset collar effectively infinite)
  intersect   - intersection-based TP criterion (PSDS-style dtc=gtc=0.7)
  intersect-lo- intersection with dtc=gtc=0.5
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_gt, load_scores
from sed_crc.stats import GRID, load_tensors
from sed_scores_eval import collar_based, intersection_based

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def floors_from_cached(variant):
    _, miss, fps, nref, classes, grid = load_tensors(
        os.path.join(CACHE, f"stats_test_{variant}.npz"))
    pooled = miss.sum(0).sum(0)  # [G]
    tot = nref.sum()
    g = int(np.argmin(pooled))
    per_class = {}
    for k, c in enumerate(classes):
        pk = miss[:, k, :].sum(0)
        nk = max(nref[:, k].sum(), 1)
        per_class[c] = round(float(pk.min() / nk), 3)
    return {"pooled_floor": float(pooled.min() / tot),
            "argmin_lam": float(grid[g]),
            "per_class_floor": per_class}


def floors_from_stats(stats):
    """stats: {class: (cp_scores, {'tps':..., 'n_ref':...})} accumulated over test."""
    total_ref, min_miss_sum = 0, 0.0
    per_class = {}
    # global grid evaluation: use each class's own curve; pooled floor needs a
    # COMMON threshold; evaluate all classes on GRID and sum
    miss_on_grid = []
    for c, (cp, st) in stats.items():
        n_ref = int(st["n_ref"])
        tps = np.asarray(st["tps"], dtype=float)
        cp = np.asarray(cp)
        order = np.argsort(cp, kind="stable")
        # library convention: arrays aligned with sorted unique cp scores,
        # value applies when threshold falls below cp; build step function
        cp_s = cp[order]
        tp_s = tps[order]
        idx = np.searchsorted(cp_s, GRID, side="right")
        # tps array from accumulated_intermediate_statistics is cumulative
        # counts per change point (not deltas) in sed_scores_eval; index into it
        idx = np.clip(idx, 0, len(tp_s) - 1)
        tp_g = tp_s[idx]
        miss_g = n_ref - tp_g
        miss_on_grid.append(miss_g)
        per_class[c] = round(float(miss_g.min() / max(n_ref, 1)), 3)
        total_ref += n_ref
    pooled = np.sum(miss_on_grid, axis=0)
    g = int(np.argmin(pooled))
    return {"pooled_floor": float(pooled[g] / total_ref),
            "argmin_lam": float(GRID[g]),
            "per_class_floor": per_class}


def main():
    out = {}
    test_gt = load_gt("test")

    for variant in ("medfilt", "csebb"):
        out[f"collar/{variant}"] = floors_from_cached(variant)

        if variant == "csebb":
            with open(os.path.join(CACHE, "test_csebb_scores.pkl"), "rb") as f:
                scores = pickle.load(f)
        else:
            scores = load_scores("test")
            first = next(iter(scores.values()))
            classes = [c for c in first.columns if c not in ("onset", "offset")]
            for cid, df in scores.items():
                d = df.copy()
                d[classes] = d[classes].rolling(window=9, center=True).median().bfill().ffill()
                scores[cid] = d
        gt = {k: test_gt[k] for k in scores}

        st, _ = collar_based.accumulated_intermediate_statistics(
            scores, gt, onset_collar=0.2, offset_collar=1e6,
            offset_collar_rate=0.0, num_jobs=8)
        out[f"onset-only/{variant}"] = floors_from_stats(st)

        for name, dtc, gtc in (("intersect70", 0.7, 0.7), ("intersect50", 0.5, 0.5)):
            sti, _ = intersection_based.accumulated_intermediate_statistics(
                scores, gt, dtc_threshold=dtc, gtc_threshold=gtc,
                cttc_threshold=None, num_jobs=8)
            # intersection stats: use 'tps' and 'n_ref'
            st2 = {c: (cp, {"tps": s["tps"], "n_ref": s["n_ref"]})
                   for c, (cp, s) in sti.items()}
            out[f"{name}/{variant}"] = floors_from_stats(st2)
        print(f"{variant} done", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "feasibility_floors.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k:22s} pooled_floor={v['pooled_floor']:.3f} @lam={v['argmin_lam']:.3f}")


if __name__ == "__main__":
    main()
