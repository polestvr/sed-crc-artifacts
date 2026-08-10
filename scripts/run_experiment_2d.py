"""2-D lambda experiment: (median-filter window W, threshold g) selected jointly
by Bonferroni-LTT. Under Bonferroni over a fixed candidate set, ALL candidates
are tested simultaneously at delta/N, so ANY selection rule among the passing
ones is valid -> we pick the one with the best (lowest) calibration FP/h.

Requires stats tensors for each window: stats_test_raw.npz (W=1 frame, i.e. no
filter) and stats_test_medfilt[W<w>].npz built by build_stats.py --window.
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
from sed_crc.routes import P_VALUES
from sed_crc.evalx import make_splits, summarize

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def stats_path(split, w):
    if w == 1:
        return os.path.join(CACHE, f"stats_{split}_raw.npz")
    if w == 9:
        return os.path.join(CACHE, f"stats_{split}_medfilt.npz")
    return os.path.join(CACHE, f"stats_{split}_medfiltW{w}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--windows", default="1,5,9,13,21")
    ap.add_argument("--loss", default="clipmean", choices=["clipmean", "pooled"])
    ap.add_argument("--grouping", default="marginal", choices=["marginal", "classwise"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--cands-per-window", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()
    windows = [int(w) for w in args.windows.split(",")]
    misses, fpss = [], []
    nref = classes = grid = clip_ids = None
    for w in windows:
        cids, m, f, nr, cls, g = load_tensors(stats_path("test", w))
        if clip_ids is None:
            clip_ids, nref, classes, grid = cids, nr, cls, g
        else:
            assert cids == clip_ids
        misses.append(m)
        fpss.append(f)
    miss = np.stack(misses, axis=2)  # [C,K,W,G]
    fps = np.stack(fpss, axis=2)
    C, K, W, G = miss.shape
    dur = load_durations("test")
    durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0

    gsub = np.unique(np.linspace(0, G - 1, args.cands_per_window).astype(int))
    cands = [(wi, gi) for wi in range(W) for gi in gsub]
    thr = args.delta / len(cands)
    pfun = P_VALUES[args.loss]
    splits = make_splits(C, args.n_splits, 0.5)

    def select(cal_idx, k_sel=None):
        """returns (wi, gi) for the pooled group or one class."""
        if k_sel is None:
            m_cal = miss[cal_idx].sum(1)   # [Ccal,W,G]
            f_cal = fps[cal_idx].sum(1)
            n_cal = nref[cal_idx].sum(1)
        else:
            m_cal = miss[cal_idx, k_sel]
            f_cal = fps[cal_idx, k_sel]
            n_cal = nref[cal_idx, k_sel]
        hours_cal = durations_h[cal_idx].sum()
        best, best_fph = None, np.inf
        fallback, fb_risk = None, np.inf
        for wi, gi in cands:
            mg = m_cal[:, wi, gi]
            risk = mg.sum() / max(n_cal.sum(), 1)
            if risk < fb_risk:
                fb_risk, fallback = risk, (wi, gi)
            if pfun(mg, n_cal, args.alpha) <= thr:
                fph = f_cal[:, wi, gi].sum() / hours_cal
                if fph < best_fph:
                    best_fph, best = fph, (wi, gi)
        return best if best is not None else fallback

    res = {"miss_share": [], "fp_per_h": [], "lams": []}
    for cal_idx, ev_idx in splits:
        if args.grouping == "marginal":
            wi, gi = select(cal_idx)
            sel = [(wi, gi)] * K
        else:
            sel = [select(cal_idx, k) for k in range(K)]
        ev_m = sum(miss[ev_idx, :, :, :][:, k, wi, gi].sum() for k, (wi, gi) in enumerate(sel))
        ev_f = sum(fps[ev_idx, :, :, :][:, k, wi, gi].sum() for k, (wi, gi) in enumerate(sel))
        tot = max(nref[ev_idx].sum(), 1)
        res["miss_share"].append(float(ev_m / tot))
        res["fp_per_h"].append(float(ev_f / durations_h[ev_idx].sum()))
        res["lams"].append([(windows[wi], float(grid[gi])) for wi, gi in sel])

    s = summarize(res, args.alpha)
    out = {"exp_id": args.exp_id, "config": vars(args), "summary": s,
           "runtime_s": round(time.time() - t0, 1),
           "example_selection": res["lams"][0]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"exp_{args.exp_id}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"exp_id": args.exp_id, **s, "runtime_s": out["runtime_s"]}))


if __name__ == "__main__":
    main()
