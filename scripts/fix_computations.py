"""Class-wise feasibility floors, clip-level
bootstrap CIs on floors, miss-vs-event-count correlation at the champion
operating point, and the U-curve figure. All from cached tensors.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sed-conformal-risk-paper")


def tensors(variant, matching):
    sfx = "" if matching == "collar" else f"__{matching}"
    return load_tensors(os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))


def pooled_curve(miss, nref):
    return miss.sum(0).sum(0) / max(nref.sum(), 1)


def main():
    out = {}
    combos = [("collar", "medfilt"), ("collar", "csebb"),
              ("intersect70", "medfilt"), ("intersect70", "csebb"),
              ("intersect50", "medfilt"), ("intersect50", "csebb"),
              ("intersect50", "csebbmax")]
    rng = np.random.default_rng(0)
    B = 1000
    for m, v in combos:
        try:
            cids, miss, fps, nref, classes, grid = tensors(v, m)
        except FileNotFoundError:
            continue
        C, K, G = miss.shape
        tot = nref.sum()
        common_floor = float(pooled_curve(miss, nref).min())
        # class-wise floor: per-class minimum misses, summed
        cw_min = miss.sum(0).min(axis=1)  # [K]
        cw_floor = float(cw_min.sum() / tot)
        # clip-level bootstrap CI of the COMMON floor
        boots = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, C, C)
            boots[b] = pooled_curve(miss[idx], nref[idx]).min()
        out[f"{m}/{v}"] = {
            "common_floor": round(common_floor, 4),
            "classwise_floor": round(cw_floor, 4),
            "floor_ci95": [round(float(np.quantile(boots, .025)), 4),
                           round(float(np.quantile(boots, .975)), 4)],
            "n_events_test": int(tot),
        }
        print(f"{m}/{v}: common {common_floor:.4f} CI {out[f'{m}/{v}']['floor_ci95']} "
              f"classwise {cw_floor:.4f}", flush=True)

    # miss-vs-event-count correlation at the champion lam (csebbmax, i20)
    cids, miss, fps, nref, classes, grid = tensors("csebbmax", "intersect50")
    # champion mean lam ~ from exp json
    ch = json.load(open(os.path.join(RESULTS, "exp_i20-csebbmax-ltt_split-marg.json")))
    lam = np.mean(list(ch["mean_lam_per_class"].values()))
    g = int(np.argmin(np.abs(grid - lam)))
    m_c = miss[:, :, g].sum(1)
    M_c = nref.sum(1)
    mask = M_c > 0
    frac = m_c[mask] / M_c[mask]
    from scipy.stats import spearmanr
    rho, p = spearmanr(M_c[mask], frac)
    pooled = m_c[mask].sum() / M_c[mask].sum()
    clipmean = float(frac.mean())
    out["champion_correlation"] = {
        "lam": round(float(grid[g]), 3),
        "spearman_Mc_vs_missfrac": round(float(rho), 3),
        "p": float(p),
        "pooled_miss_at_lam": round(float(pooled), 4),
        "clipmean_miss_at_lam": round(clipmean, 4),
    }
    print("correlation:", out["champion_correlation"], flush=True)

    with open(os.path.join(RESULTS, "reviewfix_stats.json"), "w") as f:
        json.dump(out, f, indent=1)

    # U-curve figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for m, v, lbl, style in [
        ("collar", "medfilt", "collar / median filter", "-"),
        ("collar", "csebb", "collar / cSEBB", "--"),
        ("intersect50", "medfilt", "intersection-0.5 / median filter", "-."),
        ("intersect50", "csebbmax", "intersection-0.5 / cSEBB-max", ":"),
    ]:
        cids, miss, fps, nref, classes, grid = tensors(v, m)
        ax.plot(grid, pooled_curve(miss, nref), style, label=lbl, linewidth=1.6)
    ax.axhline(0.2, color="gray", linewidth=0.7, alpha=0.7)
    ax.text(0.985, 0.205, r"$\alpha=0.2$", fontsize=8, color="gray", ha="right")
    ax.set_xlabel(r"threshold $\lambda$")
    ax.set_ylabel("pooled event miss share")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_ucurve.pdf"))
    print("figure saved")


if __name__ == "__main__":
    main()
