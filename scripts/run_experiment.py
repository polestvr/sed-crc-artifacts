"""Run one calibration experiment over 100 cal/test splits.

Examples:
  python run_experiment.py --exp-id E1 --variant medfilt --route crc_naive --grouping marginal
  python run_experiment.py --exp-id E3 --variant medfilt --route ltt_pooled --grouping marginal --delta 0.05
  python run_experiment.py --exp-id E6a --variant medfilt --route fixed05
  python run_experiment.py --exp-id E6b --variant medfilt --route valtuned
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES
from sed_crc.evalx import make_splits, run_calibrated, summarize

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def valtuned_route_factory(variant, alpha, matching="collar"):
    """No-guarantee reference: per-group smallest miss-share lam on VALIDATION
    with pooled miss <= alpha (largest such lam), no finite-sample correction."""
    sfx = "" if matching == "collar" else f"__{matching}"
    clip_ids, vmiss, vfps, vnref, classes, grid = load_tensors(
        os.path.join(CACHE, f"stats_validation_{variant}{sfx}.npz"))

    def route(ms_cal, nr_cal, alpha_, k=None, **kw):
        raise RuntimeError("valtuned ignores calibration; handled in main()")
    return vmiss, vnref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--variant", default="medfilt",
                    choices=["raw", "medfilt", "csebb", "csebbmax", "csebbtopk3"])
    ap.add_argument("--route", required=True,
                    choices=list(ROUTES) + ["fixed05", "valtuned", "valtuned_alpha",
                                            "oracle_marg", "oracle_cw"])
    ap.add_argument("--margin", type=float, default=0.0,
                    help="safety margin for valtuned_alpha (tune to alpha-margin)")
    ap.add_argument("--grouping", default="marginal",
                    choices=["marginal", "classwise", "clustered"])
    ap.add_argument("--matching", default="collar",
                    choices=["collar", "onset", "intersect70", "intersect50"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    sfx = "" if args.matching == "collar" else f"__{args.matching}"
    clip_ids, miss, fps, nref, classes, grid = load_tensors(
        os.path.join(CACHE, f"stats_test_{args.variant}{sfx}.npz"))
    dur = load_durations("test")
    durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
    C, K, G = miss.shape
    splits = make_splits(C, args.n_splits, args.cal_frac, seed_base=args.seed_base)

    if args.route in ROUTES:
        route_fn = ROUTES[args.route]
        res = run_calibrated(miss, fps, nref, durations_h, splits, args.alpha,
                             route_fn, grouping=args.grouping,
                             route_kwargs={"delta": args.delta})
    elif args.route in ("oracle_marg", "oracle_cw"):
        # ORACLE reference: thresholds chosen ON THE EVAL HALF subject to
        # empirical miss<=alpha there. Efficiency upper bound; never a
        # leaderboard candidate.
        res = {"miss_share": [], "fp_per_h": [], "lams": [], "classwise_miss": []}
        for cal_idx, ev_idx in splits:
            em = miss[ev_idx]     # [Cev,K,G]
            en = nref[ev_idx]     # [Cev,K]
            if args.route == "oracle_marg":
                pooled = em.sum(0).sum(0)               # [G]
                tot = max(en.sum(), 1)
                ok = pooled <= args.alpha * tot
                g = int(np.where(ok)[0][-1]) if ok.any() else int(np.argmin(pooled))
                lam_idx = np.full(K, g, np.int64)
            else:
                lam_idx = np.zeros(K, np.int64)
                for k in range(K):
                    pk = em[:, k, :].sum(0)
                    tk = en[:, k].sum()
                    if tk == 0:
                        lam_idx[k] = G - 1
                        continue
                    ok = pk <= args.alpha * tk
                    lam_idx[k] = int(np.where(ok)[0][-1]) if ok.any() else int(np.argmin(pk))
            ev_miss = em[:, np.arange(K), lam_idx]
            ev_fps = fps[ev_idx][:, np.arange(K), lam_idx]
            res["miss_share"].append(float(ev_miss.sum() / max(en.sum(), 1)))
            res["fp_per_h"].append(float(ev_fps.sum() / durations_h[ev_idx].sum()))
            res["lams"].append(lam_idx.copy())
    else:
        # reference rows without calibration-half use
        if args.route == "fixed05":
            g = int(np.argmin(np.abs(grid - 0.5)))
            lam_idx = np.full(K, g, np.int64)
        elif args.route == "valtuned_alpha":
            # practitioner baseline: tune the MARGINAL threshold
            # to hit alpha-margin on the validation split, deploy uncertified
            vmiss, vnref = valtuned_route_factory(args.variant, args.alpha, args.matching)
            pooled = vmiss.sum(0).sum(0)
            tot = max(vnref.sum(), 1)
            ok = pooled <= (args.alpha - args.margin) * tot
            gsel = int(np.where(ok)[0][-1]) if ok.any() else int(np.argmin(pooled))
            lam_idx = np.full(K, gsel, np.int64)
        else:  # valtuned
            vmiss, vnref = valtuned_route_factory(args.variant, args.alpha, args.matching)
            lam_idx = np.zeros(K, np.int64)
            if args.grouping == "marginal":
                pooled = vmiss.sum(0).sum(0)  # [G]
                tot = vnref.sum()
                ok = pooled <= args.alpha * tot
                gsel = int(np.where(ok)[0][-1]) if ok.any() else 0
                lam_idx[:] = gsel
            else:
                for k in range(K):
                    pk = vmiss[:, k, :].sum(0)
                    tk = vnref[:, k].sum()
                    ok = pk <= args.alpha * max(tk, 1)
                    lam_idx[k] = int(np.where(ok)[0][-1]) if ok.any() else 0
        res = {"miss_share": [], "fp_per_h": [], "lams": [], "classwise_miss": []}
        for cal_idx, ev_idx in splits:
            ev_miss = miss[ev_idx][:, np.arange(K), lam_idx]
            ev_fps = fps[ev_idx][:, np.arange(K), lam_idx]
            ev_nref = nref[ev_idx]
            res["miss_share"].append(float(ev_miss.sum() / max(ev_nref.sum(), 1)))
            res["fp_per_h"].append(float(ev_fps.sum() / durations_h[ev_idx].sum()))
            res["lams"].append(lam_idx.copy())

    s = summarize(res, args.alpha)
    lams = np.stack(res["lams"])
    out = {
        "exp_id": args.exp_id,
        "config": vars(args),
        "summary": s,
        "mean_lam_per_class": {classes[k]: float(grid[int(round(lams[:, k].mean()))])
                               for k in range(K)},
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"exp_{args.exp_id}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"exp_id": args.exp_id, **s, "runtime_s": out["runtime_s"]}))


if __name__ == "__main__":
    main()
