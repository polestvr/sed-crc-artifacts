"""Stress test: calibrate thresholds on decoding variant A, deploy on variant B
(e.g. medfilt -> csebb): does the guarantee survive a post-processing swap?

Both variants share clip ids and the threshold grid, so this is pure tensor
work. Expected: catastrophic violation when score scales differ (the paper's
'guarantee is attached to the decoder, not the model' message).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES
from sed_crc.evalx import make_splits, summarize

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--cal-variant", default="medfilt")
    ap.add_argument("--deploy-variant", default="csebb")
    ap.add_argument("--route", default="ltt_bonf_clipmean", choices=list(ROUTES))
    ap.add_argument("--grouping", default="classwise", choices=["marginal", "classwise"])
    ap.add_argument("--matching", default="collar",
                    choices=["collar", "onset", "intersect70", "intersect50"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-splits", type=int, default=100)
    args = ap.parse_args()

    sfx = "" if args.matching == "collar" else f"__{args.matching}"
    cidsA, missA, fpsA, nrefA, classes, grid = load_tensors(
        os.path.join(CACHE, f"stats_test_{args.cal_variant}{sfx}.npz"))
    cidsB, missB, fpsB, nrefB, _, _ = load_tensors(
        os.path.join(CACHE, f"stats_test_{args.deploy_variant}{sfx}.npz"))
    assert cidsA == cidsB
    dur = load_durations("test")
    durations_h = np.array([dur[c] for c in cidsA]) / 3600.0
    C, K, G = missA.shape
    splits = make_splits(C, args.n_splits, 0.5)
    route_fn = ROUTES[args.route]

    res = {"miss_share": [], "fp_per_h": [], "lams": []}
    for si, (cal_idx, ev_idx) in enumerate(splits):
        rng = np.random.default_rng(10_000 + si)
        kw = {"delta": args.delta, "rng": rng}
        if args.grouping == "marginal":
            g = route_fn(missA[cal_idx].sum(1), nrefA[cal_idx].sum(1), args.alpha, **kw)
            lam_idx = np.full(K, g, np.int64)
        else:
            lam_idx = np.zeros(K, np.int64)
            for k in range(K):
                lam_idx[k] = route_fn(missA[cal_idx, k, :], nrefA[cal_idx, k],
                                      args.alpha, **kw)
        ev_miss = missB[ev_idx][:, np.arange(K), lam_idx]
        ev_fps = fpsB[ev_idx][:, np.arange(K), lam_idx]
        res["miss_share"].append(float(ev_miss.sum() / max(nrefB[ev_idx].sum(), 1)))
        res["fp_per_h"].append(float(ev_fps.sum() / durations_h[ev_idx].sum()))
        res["lams"].append(lam_idx.copy())

    s = summarize(res, args.alpha)
    out = {"exp_id": args.exp_id, "config": vars(args), "summary": s}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"stress_{args.exp_id}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"exp_id": args.exp_id, **s}))


if __name__ == "__main__":
    main()
