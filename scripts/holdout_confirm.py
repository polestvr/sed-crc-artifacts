"""Held-out confirmation.

The 1,007 test clips are deterministically partitioned ONCE (rng 777) into a
60% SELECTION pool and a 40% CONFIRMATION pool. The whole champion search is
re-run inside the selection pool (100 splits of it, same gate); the resulting
champions are then FROZEN and evaluated in confirmation mode: 100 calibration
draws from the selection pool, each evaluated on the untouched confirmation
pool. The confirmation pool never participates in floors, search, or
selection, so its numbers are free of the selection effects disclosed in the
paper (within-corpus; corpus-level transfer remains a separate question).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

POINTS = [
    ("c60", "collar", 0.6, ["medfilt", "csebb"]),
    ("i45", "intersect70", 0.45, ["medfilt", "csebb"]),
    ("i20", "intersect50", 0.2, ["medfilt", "csebb", "csebbmax", "csebbtopk3"]),
]
ROUTE_SET = ["crc_naive", "crc_split_mono", "ltt_split_clipmean",
             "ltt_bonf_clipmean", "ltt_bonf_pooled", "rcps_fixedseq"]
GROUPINGS = ["marginal", "classwise"]
N_SPLITS = 100
GATE = 95


def load(variant, matching):
    sfx = "" if matching == "collar" else f"__{matching}"
    return load_tensors(os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))


def select_lam(route_fn, miss_k, nref_k, miss_full, nref_full, grouping, alpha, cal, si):
    rng = np.random.default_rng(50_000 + si)
    if grouping == "marginal":
        g = route_fn(miss_k[cal], nref_k[cal], alpha, delta=0.05, rng=rng)
        return np.full(miss_full.shape[1], g, np.int64)
    K = miss_full.shape[1]
    lam = np.zeros(K, np.int64)
    for k in range(K):
        lam[k] = route_fn(miss_full[cal, k, :], nref_full[cal, k], alpha, delta=0.05, rng=rng)
    return lam


def main():
    rng = np.random.default_rng(777)
    # partition indices over the canonical clip order of any tensor file
    clip_ids, miss0, _, _, _, _ = load("medfilt", "collar")
    C = miss0.shape[0]
    perm = rng.permutation(C)
    n_sel = int(round(0.6 * C))
    sel, conf = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
    dur = load_durations("test")
    durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0

    out = {"n_sel": int(n_sel), "n_conf": int(C - n_sel), "points": {}}
    for tag, matching, alpha, variants in POINTS:
        rows = []
        for variant in variants:
            try:
                cids, miss, fps, nref, classes, grid = load(variant, matching)
            except FileNotFoundError:
                continue
            assert cids == clip_ids
            K = miss.shape[1]
            miss_k, fps_k, nref_k = miss.sum(1), fps.sum(1), nref.sum(1)
            for route in ROUTE_SET:
                if route == "rcps_fixedseq" and not variant.startswith("csebb"):
                    continue  # only licensed for monotone box decodings
                route_fn = ROUTES[route]
                for grouping in GROUPINGS:
                    if route == "rcps_fixedseq" and grouping == "classwise":
                        continue
                    # ---- selection phase: splits WITHIN the selection pool
                    n_ok, fphs = 0, []
                    for si in range(N_SPLITS):
                        r2 = np.random.default_rng(60_000 + si)
                        p = r2.permutation(n_sel)
                        cal = sel[p[: n_sel // 2]]
                        ev = sel[p[n_sel // 2:]]
                        lam = select_lam(route_fn, miss_k, nref_k, miss, nref,
                                         grouping, alpha, cal, si)
                        ev_m = miss[ev][:, np.arange(K), lam].sum()
                        ev_r = nref[ev].sum()
                        if ev_m <= alpha * ev_r:
                            n_ok += 1
                        fphs.append(fps[ev][:, np.arange(K), lam].sum()
                                    / durations_h[ev].sum())
                    rows.append(dict(variant=variant, route=route, grouping=grouping,
                                     sel_n_ok=n_ok, sel_fph=float(np.mean(fphs))))
        passing = [r for r in rows if r["sel_n_ok"] >= GATE]
        champ = min(passing, key=lambda r: r["sel_fph"]) if passing else None
        entry = {"search_rows": len(rows), "n_passing": len(passing),
                 "selection_champion": champ}
        # ---- confirmation phase for the frozen champion
        if champ:
            cids, miss, fps, nref, classes, grid = load(champ["variant"], matching)
            K = miss.shape[1]
            miss_k, nref_k = miss.sum(1), nref.sum(1)
            route_fn = ROUTES[champ["route"]]
            cal_n = int(os.environ.get("HOLDOUT_CAL_N", n_sel // 2))
            n_ok, misses, fphs = 0, [], []
            for si in range(N_SPLITS):
                r2 = np.random.default_rng(70_000 + si)
                cal = sel[r2.permutation(n_sel)[: cal_n]]
                lam = select_lam(route_fn, miss_k, nref_k, miss, nref,
                                 champ["grouping"], alpha, cal, si)
                cm = miss[conf][:, np.arange(K), lam].sum()
                cr = nref[conf].sum()
                misses.append(float(cm / cr))
                if cm <= alpha * cr:
                    n_ok += 1
                fphs.append(fps[conf][:, np.arange(K), lam].sum()
                            / durations_h[conf].sum())
            entry["confirmation"] = {
                "n_ok": n_ok, "mean_miss": round(float(np.mean(misses)), 4),
                "q95_miss": round(float(np.quantile(misses, 0.95)), 4),
                "fp_per_h_mean": round(float(np.mean(fphs)), 1),
                "fp_per_h_std": round(float(np.std(fphs)), 1),
            }
        out["points"][tag] = entry
        print(tag, json.dumps(entry), flush=True)

    suffix = os.environ.get("HOLDOUT_OUT_SUFFIX", "")
    with open(os.path.join(RESULTS, f"holdout_confirmation{suffix}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved results/holdout_confirmation{suffix}.json")


if __name__ == "__main__":
    main()
