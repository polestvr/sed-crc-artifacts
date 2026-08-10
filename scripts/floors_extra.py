"""Feasibility floors for the remaining decodings (raw, cSEBB-max,
cSEBB-top3) wherever test tensors exist, plus the a-priori event cap taken
from the VALIDATION split (fixed before test data under the study's own
ordering) and whether any test clip exceeds it."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    out = {}
    for m in ("collar", "intersect70", "intersect50"):
        sfx = "" if m == "collar" else f"__{m}"
        for v in ("raw", "csebbmax", "csebbtopk3"):
            p = os.path.join(CACHE, f"stats_test_{v}{sfx}.npz")
            if not os.path.exists(p):
                continue
            _, miss, fps, nref, classes, grid = load_tensors(p)
            tot = max(int(nref.sum()), 1)
            out[f"{m}/{v}"] = {
                "common_floor": round(float(miss.sum(0).sum(0).min() / tot), 4),
                "classwise_floor": round(float(miss.sum(0).min(axis=1).sum() / tot), 4),
            }
            print(f"{m}/{v}:", out[f"{m}/{v}"], flush=True)

    # a-priori event cap from validation
    _, vmiss, vfps, vnref, _, _ = load_tensors(
        os.path.join(CACHE, "stats_validation_medfilt.npz"))
    _, tmiss, tfps, tnref, _, _ = load_tensors(
        os.path.join(CACHE, "stats_test_medfilt.npz"))
    cap_val = int(vnref.sum(1).max())
    max_test = int(tnref.sum(1).max())
    out["event_cap"] = {"validation_max_events_per_clip": cap_val,
                       "test_max_events_per_clip": max_test,
                       "test_exceeds_val_cap": bool(max_test > cap_val)}
    print("event cap:", out["event_cap"])

    with open(os.path.join(RESULTS, "floors_extra.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/floors_extra.json")


if __name__ == "__main__":
    main()
