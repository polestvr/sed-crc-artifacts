"""Cross-corpus stress: calibrate per-class thresholds on RealDESED test,
deploy on DESED public eval (and report the reverse direction), for the
classes with a defensible label mapping:

    RealDESED            DESED public eval
    running_water    <-> Running_water
    vacuum_cleaner   <-> Vacuum_cleaner
    cutlery_dishes   <-> Dishes

DESED gt labels are renamed to RealDESED class names so the frozen model's
score columns match. Median-filter decoding on both sides.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import clip_stats_tensors, load_tensors, save_tensors
from sed_crc.routes import ROUTES
from build_stats import medfilt_scores

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

MAPPING = {
    "Running_water": "running_water",
    "Vacuum_cleaner": "vacuum_cleaner",
    "Dishes": "cutlery_dishes",
}
MAPPED = list(MAPPING.values())


def load_desed_gt(tsv=os.path.expanduser(
        "~/data/desed_public_eval/dataset/metadata/eval/public.tsv")):
    df = pd.read_csv(tsv, sep="\t")
    gt = {}
    for fn, on, off, lbl in zip(df["filename"], df["onset"], df["offset"], df["event_label"]):
        cid = os.path.splitext(str(fn))[0]
        gt.setdefault(cid, [])
        if isinstance(lbl, str) and lbl in MAPPING and float(off) > float(on):
            gt[cid].append((float(on), float(off), MAPPING[lbl]))
    return gt


def build_desed_stats(matching="collar"):
    sfx = "" if matching == "collar" else f"__{matching}"
    path = os.path.join(CACHE, f"stats_desed_medfilt{sfx}.npz")
    if os.path.exists(path):
        return path
    with open(os.path.join(CACHE, "desed_scores.pkl"), "rb") as f:
        scores = pickle.load(f)
    # keep only the mapped class columns: sed_scores_eval asserts every score
    # column has >=1 gt event in the dataset
    scores = {cid: df[["onset", "offset"] + MAPPED].copy() for cid, df in scores.items()}
    scores = medfilt_scores(scores, MAPPED)
    gt = load_desed_gt()
    gt = {cid: gt.get(cid, []) for cid in scores}
    clip_ids, miss, fps, nref = clip_stats_tensors(scores, gt, MAPPED, num_jobs=8,
                                                   matching=matching)
    save_tensors(path, clip_ids, miss, fps, nref, MAPPED)
    return path


def eval_at(miss, fps, nref, durations_h, lam_idx, k_indices):
    m = sum(miss[:, k, lam_idx[k]].sum() for k in k_indices)
    f = sum(fps[:, k, lam_idx[k]].sum() for k in k_indices)
    n = sum(nref[:, k].sum() for k in k_indices)
    hours = durations_h.sum()
    return (m / max(n, 1), f / hours, int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="ltt_bonf_clipmean", choices=list(ROUTES))
    ap.add_argument("--matching", default="collar",
                    choices=["collar", "onset", "intersect70", "intersect50"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()

    sfx = "" if args.matching == "collar" else f"__{args.matching}"
    dpath = build_desed_stats(args.matching)
    dcids, dmiss, dfps, dnref, dclasses, grid = load_tensors(dpath)
    ddur = json.load(open(os.path.join(CACHE, "desed_durations.json")))
    ddur_h = np.array([ddur[c] for c in dcids]) / 3600.0

    rcids, rmiss, rfps, rnref, rclasses, _ = load_tensors(
        os.path.join(CACHE, f"stats_test_medfilt{sfx}.npz"))
    rdur = load_durations("test")
    rdur_h = np.array([rdur[c] for c in rcids]) / 3600.0

    rk = {c: rclasses.index(c) for c in MAPPED}   # RealDESED-side indices
    dk = {c: dclasses.index(c) for c in MAPPED}   # DESED-side indices (K=3)
    route_fn = ROUTES[args.route]
    rng = np.random.default_rng(777)

    out = {"config": vars(args), "mapping": MAPPING}

    # Direction 1: calibrate on FULL RealDESED test -> deploy on DESED
    lam_r = {}
    for c in MAPPED:
        k = rk[c]
        lam_r[c] = route_fn(rmiss[:, k, :], rnref[:, k], args.alpha,
                            delta=args.delta, rng=rng)
    lam_d_idx = np.zeros(len(dclasses), np.int64)
    for c in MAPPED:
        lam_d_idx[dk[c]] = lam_r[c]
    m, f, n = eval_at(dmiss, dfps, dnref, ddur_h, lam_d_idx, [dk[c] for c in MAPPED])
    lam_r_idx = np.zeros(len(rclasses), np.int64)
    for c in MAPPED:
        lam_r_idx[rk[c]] = lam_r[c]
    m_in, f_in, n_in = eval_at(rmiss, rfps, rnref, rdur_h, lam_r_idx, [rk[c] for c in MAPPED])
    out["realdesed_to_desed"] = {
        "lams": {c: float(grid[lam_r[c]]) for c in MAPPED},
        "desed_miss_share": float(m), "desed_fp_per_h": float(f), "desed_n_events": n,
        "indomain_resub_miss": float(m_in), "indomain_n_events": n_in,
    }

    # Direction 2: calibrate on DESED -> deploy on RealDESED test
    lam_d = {}
    for c in MAPPED:
        k = dk[c]
        lam_d[c] = route_fn(dmiss[:, k, :], dnref[:, k], args.alpha,
                            delta=args.delta, rng=rng)
    lam_r2 = np.zeros(len(rclasses), np.int64)
    for c in MAPPED:
        lam_r2[rk[c]] = lam_d[c]
    m2, f2, n2 = eval_at(rmiss, rfps, rnref, rdur_h, lam_r2, [rk[c] for c in MAPPED])
    out["desed_to_realdesed"] = {
        "lams": {c: float(grid[lam_d[c]]) for c in MAPPED},
        "realdesed_miss_share": float(m2), "realdesed_fp_per_h": float(f2),
        "realdesed_n_events": n2,
        "desed_cal_events": {c: int(dnref[:, dk[c]].sum()) for c in MAPPED},
    }

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"stress_crosscorpus__{args.matching}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    main()
