"""Per-seed feasibility floors: common and class-wise pooled miss floors for
medfilt/csebb x {collar, intersect70, intersect50} from this seed's cache
(SED_CRC_CACHE env). Appends one JSON line to results/seed_floors.jsonl."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    seed_tag = sys.argv[1] if len(sys.argv) > 1 else "seed?"
    out = {"seed": seed_tag, "cache": CACHE}
    for m in ("collar", "intersect70", "intersect50"):
        sfx = "" if m == "collar" else f"__{m}"
        for v in ("medfilt", "csebb"):
            p = os.path.join(CACHE, f"stats_test_{v}{sfx}.npz")
            if not os.path.exists(p):
                continue
            _, miss, fps, nref, classes, grid = load_tensors(p)
            tot = max(int(nref.sum()), 1)
            pooled = miss.sum(0).sum(0)
            cw = miss.sum(0).min(axis=1).sum()
            out[f"{m}/{v}"] = {"common_floor": round(float(pooled.min() / tot), 4),
                               "classwise_floor": round(float(cw / tot), 4)}
    gr = os.path.join(CACHE, "gate_result.json")
    if os.path.exists(gr):
        g = json.load(open(gr))
        out["psds1m"] = {k: round(g[k]["psds1_macro"], 4) for k in ("raw", "medfilt", "csebb") if k in g}
    with open(os.path.join(RESULTS, "seed_floors.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
