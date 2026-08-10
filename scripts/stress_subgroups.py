"""Stress test: does the (marginal/classwise) guarantee calibrated on random
calibration halves hold inside metadata subgroups of the eval half?

Subgroups: device category (ios/android/other), device placement
(static/mobile), and the most frequent recording environments.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/sed-crc-work/RealDESED"))
from sed_crc.gt import CACHE, load_durations, load_meta
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES
from sed_crc.evalx import make_splits
from utils.evaluation import categorize_recording_device  # official grouping logic

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def env_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(";") if x.strip()]
    return list(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--variant", default="medfilt",
                    choices=["raw", "medfilt", "csebb", "csebbmax", "csebbtopk3"])
    ap.add_argument("--route", required=True, choices=list(ROUTES))
    ap.add_argument("--grouping", default="marginal", choices=["marginal", "classwise"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--top-envs", type=int, default=6)
    ap.add_argument("--matching", default="collar",
                    choices=["collar", "onset", "intersect70", "intersect50"])
    args = ap.parse_args()

    sfx = "" if args.matching == "collar" else f"__{args.matching}"
    clip_ids, miss, fps, nref, classes, grid = load_tensors(
        os.path.join(CACHE, f"stats_test_{args.variant}{sfx}.npz"))
    dur = load_durations("test")
    meta = load_meta("test")
    durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
    C, K, G = miss.shape

    groups = {}
    dev = np.array([categorize_recording_device(meta[c].get("recording_device")) for c in clip_ids])
    for cat in ("ios", "android", "other"):
        groups[f"device/{cat}"] = dev == cat
    plc = np.array([(meta[c].get("device_placement") or "").strip().lower() for c in clip_ids])
    for cat in ("static", "mobile"):
        groups[f"placement/{cat}"] = plc == cat
    env_lists = [env_list(meta[c].get("recording_environment")) for c in clip_ids]
    env_counts = Counter(e for lst in env_lists for e in lst)
    for env, _ in env_counts.most_common(args.top_envs):
        groups[f"env/{env}"] = np.array([env in lst for lst in env_lists])

    splits = make_splits(C, args.n_splits, 0.5)
    route_fn = ROUTES[args.route]
    per_group = {g: {"miss": [], "fph": [], "n_ev_events": []} for g in groups}
    overall = {"miss": [], "fph": []}

    for si, (cal_idx, ev_idx) in enumerate(splits):
        rng = np.random.default_rng(10_000 + si)
        if args.grouping == "marginal":
            ms = miss[cal_idx].sum(1)
            nr = nref[cal_idx].sum(1)
            gsel = route_fn(ms, nr, args.alpha, delta=args.delta, rng=rng)
            lam_idx = np.full(K, gsel, np.int64)
        else:
            lam_idx = np.zeros(K, np.int64)
            for k in range(K):
                lam_idx[k] = route_fn(miss[cal_idx, k, :], nref[cal_idx, k],
                                      args.alpha, delta=args.delta, rng=rng)
        ev_mask = np.zeros(C, bool)
        ev_mask[ev_idx] = True
        m_at = miss[:, np.arange(K), lam_idx]
        f_at = fps[:, np.arange(K), lam_idx]
        overall["miss"].append(float(m_at[ev_mask].sum() / max(nref[ev_mask].sum(), 1)))
        overall["fph"].append(float(f_at[ev_mask].sum() / durations_h[ev_mask].sum()))
        for gname, gmask in groups.items():
            mask = ev_mask & gmask
            n_events = int(nref[mask].sum())
            if n_events == 0:
                continue
            per_group[gname]["miss"].append(float(m_at[mask].sum() / n_events))
            per_group[gname]["fph"].append(float(f_at[mask].sum() / max(durations_h[mask].sum(), 1e-9)))
            per_group[gname]["n_ev_events"].append(n_events)

    def summ(miss_list, alpha):
        a = np.array(miss_list)
        return {"n_splits": len(a), "n_ok": int((a <= alpha).sum()),
                "mean_miss": float(a.mean()), "q95_miss": float(np.quantile(a, 0.95))}

    out = {
        "config": vars(args),
        "overall": {**summ(overall["miss"], args.alpha),
                    "fp_per_h_mean": float(np.mean(overall["fph"]))},
        "groups": {},
    }
    for gname, d in per_group.items():
        if not d["miss"]:
            continue
        out["groups"][gname] = {
            **summ(d["miss"], args.alpha),
            "fp_per_h_mean": float(np.mean(d["fph"])),
            "mean_events_per_split": float(np.mean(d["n_ev_events"])),
            "n_clips_total": int(groups[gname].sum()),
        }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"stress_{args.exp_id}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({g: {"n_ok": v["n_ok"], "mean_miss": round(v["mean_miss"], 4)}
                      for g, v in out["groups"].items()}, indent=1))


if __name__ == "__main__":
    main()
