"""Build per-clip collar-statistics tensors for a (split, variant) pair.

Variants:
  raw      - cached sigmoid score curves as-is
  medfilt  - official 360 ms (9-frame) centered rolling median, bfill/ffill
  csebb    - cSEBB confidence curves produced by reproduce_gate.py (cached pkl)

Includes a differential test of the grid tensors against
sed_scores_eval.collar_based.fscore at fixed thresholds.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, COLLAR, load_gt, load_scores
from sed_crc.stats import GRID, clip_stats_tensors, save_tensors


def medfilt_scores(scores, classes, window=9):
    out = {}
    for cid, df in scores.items():
        d = df.copy()
        d[classes] = d[classes].rolling(window=window, center=True).median().bfill().ffill()
        out[cid] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--variant", required=True,
                    choices=["raw", "medfilt", "csebb", "csebbmax", "csebbtopk3"])
    ap.add_argument("--window", type=int, default=9,
                    help="median filter window in frames (medfilt variant only); "
                         "window!=9 is cached as variant medfiltW<window>")
    ap.add_argument("--matching", default="collar",
                    choices=["collar", "onset", "intersect70", "intersect50"])
    ap.add_argument("--num_jobs", type=int, default=8)
    ap.add_argument("--skip_check", action="store_true")
    args = ap.parse_args()

    if args.variant.startswith("csebb"):
        with open(os.path.join(CACHE, f"{args.split}_{args.variant}_scores.pkl"), "rb") as f:
            scores = pickle.load(f)
    else:
        scores = load_scores(args.split)
    first = next(iter(scores.values()))
    classes = [c for c in first.columns if c not in ("onset", "offset")]
    variant_name = args.variant
    if args.variant == "medfilt":
        scores = medfilt_scores(scores, classes, window=args.window)
        if args.window != 9:
            variant_name = f"medfiltW{args.window}"

    gt = load_gt(args.split)
    gt = {cid: gt[cid] for cid in scores}

    clip_ids, miss, fps, nref, = clip_stats_tensors(
        scores, gt, classes, num_jobs=args.num_jobs, matching=args.matching)
    suffix = "" if args.matching == "collar" else f"__{args.matching}"
    out = os.path.join(CACHE, f"stats_{args.split}_{variant_name}{suffix}.npz")
    save_tensors(out, clip_ids, miss, fps, nref, classes)
    print("saved", out, "miss shape", miss.shape, "total ref", int(nref.sum()))

    if not args.skip_check:
        from sed_scores_eval import collar_based, intersection_based
        from sed_crc.stats import MATCHINGS
        _, mkw = MATCHINGS[args.matching]
        for thr in (0.3, 0.5, 0.7):
            g = int(np.argmin(np.abs(GRID - thr)))
            tgrid = float(GRID[g])
            if args.matching in ("collar", "onset"):
                f, p, r, stats = collar_based.fscore(
                    scores, gt, threshold=tgrid, num_jobs=args.num_jobs,
                    return_onset_offset_dist_sum=False, **mkw,
                )
            else:
                ikw = {k: v for k, v in mkw.items()
                       if k in ("dtc_threshold", "gtc_threshold")}
                f, p, r, stats = intersection_based.fscore(
                    scores, gt, threshold=tgrid, num_jobs=args.num_jobs, **ikw,
                )
            for ki, cls in enumerate(classes):
                lib_tp = stats[cls]["tps"]
                lib_fp = stats[cls]["fps"]
                my_tp = int(nref[:, ki].sum() - miss[:, ki, g].sum())
                my_fp = int(fps[:, ki, g].sum())
                assert my_tp == lib_tp, (cls, tgrid, my_tp, lib_tp)
                assert my_fp == lib_fp, (cls, tgrid, my_fp, lib_fp)
        print("differential check vs collar_based.fscore: PASS (thr 0.3/0.5/0.7)")


if __name__ == "__main__":
    main()
