"""(a) Bootstrap CI for the champion-threshold Spearman
correlation between per-clip event count and miss fraction;
(b) exact binomial p-values, Bonferroni-corrected across subgroups, for the
subgroup gate exceedances."""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binom, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE
from sed_crc.stats import load_tensors

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

N_BOOT = 10_000


def main():
    out = {}

    # (a) Spearman bootstrap CI, replicating fix_computations exactly
    ch = json.load(open(os.path.join(
        RESULTS, "exp_i20-csebbmax-ltt_split-marg.json")))
    lam = np.mean(list(ch["mean_lam_per_class"].values()))
    _, miss, fps, nref, classes, grid = load_tensors(
        os.path.join(CACHE, "stats_test_csebbmax__intersect50.npz"))
    g = int(np.argmin(np.abs(grid - lam)))
    m_c = miss[:, :, g].sum(1)
    M_c = nref.sum(1)
    mask = M_c > 0
    Mv, frac = M_c[mask], m_c[mask] / M_c[mask]
    rho, p = spearmanr(Mv, frac)
    n = len(Mv)
    rng = np.random.default_rng(123)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[b] = spearmanr(Mv[idx], frac[idx]).statistic
    out["spearman"] = {
        "rho": round(float(rho), 3), "p": float(p), "n_clips": int(n),
        "ci95_bootstrap": [round(float(np.percentile(boots, 2.5)), 3),
                           round(float(np.percentile(boots, 97.5)), 3)],
    }
    print("spearman:", out["spearman"], flush=True)

    # (b) subgroup binomial p-values with Bonferroni across groups per file
    out["subgroups"] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "stress_SG*.json"))):
        d = json.load(open(path))
        groups = d.get("groups", {})
        k = len(groups)
        rows = {}
        for gname, gd in groups.items():
            n_sp, n_ok = gd["n_splits"], gd["n_ok"]
            # H0: true pass prob >= 0.95; one-sided p = P(X <= n_ok)
            pval = float(binom.cdf(n_ok, n_sp, 0.95))
            rows[gname] = {
                "n_ok": n_ok, "n_splits": n_sp,
                "p_onesided": pval,
                "p_bonferroni": min(1.0, pval * k),
                "significant_after_bonf": bool(pval * k < 0.05),
            }
        out["subgroups"][os.path.basename(path)] = {"n_groups": k, "rows": rows}
        sig = [g for g, r in rows.items() if r["significant_after_bonf"]]
        print(os.path.basename(path), "significant after Bonferroni:", sig,
              flush=True)

    with open(os.path.join(RESULTS, "misc_cis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/misc_cis.json")


if __name__ == "__main__":
    main()
