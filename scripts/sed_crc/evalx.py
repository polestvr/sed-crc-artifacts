"""Split machinery + gate + efficiency evaluation.

100 seeded 50/50 splits of test clips into calibration/evaluation halves.
Gate: pooled empirical miss share <= alpha on the eval half in >= 95/100 splits.
Metric: FP/h on the eval half, mean over splits.
"""
import numpy as np


def make_splits(n_clips, n_splits=100, cal_frac=0.5, seed_base=0):
    splits = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed_base + s)
        perm = rng.permutation(n_clips)
        n_cal = int(round(cal_frac * n_clips))
        splits.append((perm[:n_cal], perm[n_cal:]))
    return splits


def run_calibrated(
    miss, fps, nref, durations_h, splits, alpha, route_fn,
    grouping="classwise", route_kwargs=None,
):
    """miss/fps [C,K,G], nref [C,K], durations_h [C] hours.

    grouping:
      - "marginal": one lam for ALL classes jointly (miss/nref summed over K).
      - "classwise": per-class lam_k, each calibrated on that class's events
        (route applied per class; guarantee per class => pooled also <= alpha
        in expectation terms; empirically checked by the gate).
    Returns dict with per-split pooled miss share, fp/h, chosen lams.
    """
    route_kwargs = route_kwargs or {}
    C, K, G = miss.shape
    res = {"miss_share": [], "fp_per_h": [], "lams": [], "classwise_miss": []}
    for si, (cal_idx, ev_idx) in enumerate(splits):
        rng = np.random.default_rng(10_000 + si)  # deterministic inner split
        kw = {**route_kwargs, "rng": rng}
        if grouping == "marginal":
            ms = miss[cal_idx].sum(1)          # [Ccal, G]
            nr = nref[cal_idx].sum(1)          # [Ccal]
            g = route_fn(ms, nr, alpha, **kw)
            lam_idx = np.full(K, g, np.int64)
        elif grouping == "clustered":
            # 3 clusters by calibration-half event count (rare/mid/frequent);
            # one shared lam per cluster, calibrated on the cluster's pooled events
            counts = nref[cal_idx].sum(0)      # [K]
            order = np.argsort(counts)
            k3 = K // 3
            clusters = [order[:k3], order[k3:2 * k3], order[2 * k3:]]
            lam_idx = np.zeros(K, np.int64)
            for cl in clusters:
                ms = miss[cal_idx][:, cl, :].sum(1)
                nr = nref[cal_idx][:, cl].sum(1)
                g = route_fn(ms, nr, alpha, **kw)
                lam_idx[cl] = g
        else:
            lam_idx = np.zeros(K, np.int64)
            for k in range(K):
                g = route_fn(miss[cal_idx, k, :], nref[cal_idx, k], alpha, **kw)
                lam_idx[k] = g
        ev_miss = miss[ev_idx][:, np.arange(K), lam_idx]   # [Cev, K]
        ev_fps = fps[ev_idx][:, np.arange(K), lam_idx]
        ev_nref = nref[ev_idx]
        tot_ref = ev_nref.sum()
        share = ev_miss.sum() / max(tot_ref, 1)
        hours = durations_h[ev_idx].sum()
        res["miss_share"].append(float(share))
        res["fp_per_h"].append(float(ev_fps.sum() / hours))
        res["lams"].append(lam_idx.copy())
        with np.errstate(invalid="ignore", divide="ignore"):
            cw = ev_miss.sum(0) / np.maximum(ev_nref.sum(0), 1)
        res["classwise_miss"].append(cw)
    return res


def summarize(res, alpha, n_required=95):
    miss = np.array(res["miss_share"])
    fph = np.array(res["fp_per_h"])
    n_ok = int((miss <= alpha).sum())
    return {
        "gate_pass": bool(n_ok >= n_required),
        "n_splits_ok": n_ok,
        "mean_miss_share": float(miss.mean()),
        "q95_miss_share": float(np.quantile(miss, 0.95)),
        "fp_per_h_mean": float(fph.mean()),
        "fp_per_h_std": float(fph.std()),
    }
