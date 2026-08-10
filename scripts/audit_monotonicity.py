"""E0: monotonicity audit of event-level miss counts vs threshold (validation).

For each class and each decoding variant: pooled miss curve M_k(lam) over the
grid must be non-decreasing in lam if losses were monotone. We count and size
the violations (negative steps), at dataset level and clip level.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def audit_variant(variant, subgrid=None, matching="collar"):
    sfx = "" if matching == "collar" else f"__{matching}"
    path = os.path.join(CACHE, f"stats_validation_{variant}{sfx}.npz")
    clip_ids, miss, fps, nref, classes, grid = load_tensors(path)
    if subgrid is not None:
        sel = np.linspace(0, len(grid) - 1, subgrid).astype(int)
        miss = miss[:, :, sel]
    out = {}
    pooled = miss.sum(0)  # [K,G]
    for ki, cls in enumerate(classes):
        d = np.diff(pooled[ki].astype(np.int64))
        neg = d < 0
        # clip-level: count clips whose own miss curve has any negative step
        dclip = np.diff(miss[:, ki, :].astype(np.int64), axis=1)
        clips_nonmono = int((dclip < 0).any(axis=1).sum())
        out[cls] = {
            "n_ref": int(nref[:, ki].sum()),
            "n_neg_steps": int(neg.sum()),
            "sum_neg_magnitude": int(-d[neg].sum()) if neg.any() else 0,
            "max_neg_step": int(-d[neg].min()) if neg.any() else 0,
            "clips_with_nonmono": clips_nonmono,
            "n_clips_with_refs": int((nref[:, ki] > 0).sum()),
        }
    agg = {
        "classes_with_violations": sum(1 for v in out.values() if v["n_neg_steps"] > 0),
        "total_neg_steps": sum(v["n_neg_steps"] for v in out.values()),
        "total_neg_magnitude": sum(v["sum_neg_magnitude"] for v in out.values()),
        "total_refs": sum(v["n_ref"] for v in out.values()),
    }
    return {"per_class": out, "aggregate": agg}


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for matching in ("collar", "intersect70", "intersect50"):
        sfx = "" if matching == "collar" else f"__{matching}"
        for variant in ("raw", "medfilt", "csebb"):
            path = os.path.join(CACHE, f"stats_validation_{variant}{sfx}.npz")
            if not os.path.exists(path):
                continue
            key = f"{matching}/{variant}"
            report[key] = {
                "grid1001": audit_variant(variant, matching=matching),
                "grid200": audit_variant(variant, subgrid=200, matching=matching),
            }
            a = report[key]["grid200"]["aggregate"]
            print(f"{key:22s} grid200: {a['classes_with_violations']}/15 classes violate, "
                  f"{a['total_neg_steps']} neg steps, magnitude {a['total_neg_magnitude']} "
                  f"of {a['total_refs']} refs")
    with open(os.path.join(OUT, "audit_monotonicity.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("saved results/audit_monotonicity.json")


if __name__ == "__main__":
    main()
