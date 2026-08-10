"""Instrument study: what does the empirical gate
actually measure?

For key configurations, over N=10,000 calibration/evaluation splits:
  - gate pass fraction under the POOLED criterion, complementary halves
    (the paper's gate) -- now with negligible Monte-Carlo error;
  - gate pass fraction under the MATCHED functional (clip-balanced criterion
    for clip-HB certificates) -- separates functional mismatch from slack;
  - POPULATION-risk certificate check: the full 1,007-clip empirical
    distribution is treated as the population; per calibration draw we record
    whether the selected threshold's full-split pooled and clip-balanced risk
    exceeds alpha -- a direct test of P(R(lam_hat)<=alpha)>=1-delta;
  - gate pass fraction with an INDEPENDENTLY drawn evaluation half
    (breaks the complementary-split negative coupling);
  - FP/h mean and std over splits.

Also: miss share at lam=0 per rule/decoder (U-shape endpoint numbers) and the
exact count of calibrated configurations that competed in the search.
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import load_tensors
from sed_crc.routes import ROUTES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

N_SPLITS = 10_000

CONFIGS = [
    ("champ-i20", "csebbmax", "intersect50", "ltt_split_clipmean", 0.2),
    ("top3-i20", "csebbtopk3", "intersect50", "ltt_split_clipmean", 0.2),
    ("bonf-i20", "csebbmax", "intersect50", "ltt_bonf_clipmean", 0.2),
    ("rcps-i20", "csebbmax", "intersect50", "rcps_fixedseq", 0.2),
    ("rcpsP-i20", "csebbmax", "intersect50", "rcps_fixedseq_pooled", 0.2),
    ("crc-i20", "csebb", "intersect50", "crc_naive", 0.2),
    ("bonf-i45", "csebb", "intersect70", "ltt_bonf_clipmean", 0.45),
    ("split-i45", "csebb", "intersect70", "ltt_split_clipmean", 0.45),
    ("bonf-c60", "csebb", "collar", "ltt_bonf_clipmean", 0.6),
    # all remaining near-boundary rows
    ("split-c60", "csebb", "collar", "ltt_split_clipmean", 0.6),
    ("splitcse-i20", "csebb", "intersect50", "ltt_split_clipmean", 0.2),
    ("crcC-i20", "csebb", "intersect50", "crc_c_nonmono", 0.2),
    ("crcNM-c60", "csebb", "collar", "crc_nm", 0.6),
]


def run_config(tag, variant, matching, route, alpha, delta=0.05):
    sfx = "" if matching == "collar" else f"__{matching}"
    clip_ids, miss, fps, nref, classes, grid = load_tensors(
        os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
    dur = load_durations("test")
    durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
    C, K, G = miss.shape
    route_fn = ROUTES[route]

    miss_k = miss.sum(1)          # [C,G] pooled over classes per clip
    fps_k = fps.sum(1)
    nref_k = nref.sum(1)          # [C]
    tot_ref = nref_k.sum()
    pop_pooled = miss_k.sum(0) / tot_ref                    # [G]
    m = nref_k > 0
    pop_clipmean = (miss_k[m] / nref_k[m][:, None]).mean(0)  # [G]

    t0 = time.time()
    res = dict(pooled_pass=0, clip_pass=0, indep_pass=0,
               pop_pooled_viol=0, pop_clip_viol=0, fph=[], g_sel=[])
    half = C // 2
    for si in range(N_SPLITS):
        rng = np.random.default_rng(1_000_000 + si)
        perm = rng.permutation(C)
        cal, ev = perm[:half], perm[half:]
        g = route_fn(miss_k[cal][:, None, :].squeeze(1)[:, :] if False else miss_k[cal],
                     nref_k[cal], alpha, delta=delta, rng=np.random.default_rng(2_000_000 + si))
        res["g_sel"].append(g)
        # complementary evaluation half, pooled criterion (the paper's gate)
        ev_miss = miss_k[ev, g].sum()
        ev_ref = nref_k[ev].sum()
        if ev_miss <= alpha * ev_ref:
            res["pooled_pass"] += 1
        # matched functional: clip-balanced criterion on the same eval half
        me = nref_k[ev] > 0
        if (miss_k[ev, g][me] / nref_k[ev][me]).mean() <= alpha:
            res["clip_pass"] += 1
        # independent evaluation half (fresh draw, may overlap calibration)
        ev2 = np.random.default_rng(3_000_000 + si).choice(C, half, replace=False)
        if miss_k[ev2, g].sum() <= alpha * nref_k[ev2].sum():
            res["indep_pass"] += 1
        # population-risk certificate check on the full split
        if pop_pooled[g] > alpha:
            res["pop_pooled_viol"] += 1
        if pop_clipmean[g] > alpha:
            res["pop_clip_viol"] += 1
        res["fph"].append(fps_k[ev, g].sum() / durations_h[ev].sum())

    out = {
        "config": dict(variant=variant, matching=matching, route=route, alpha=alpha),
        "n_splits": N_SPLITS,
        "gate_pass_frac_pooled_complementary": round(res["pooled_pass"] / N_SPLITS, 4),
        "gate_pass_frac_pooled_independent": round(res["indep_pass"] / N_SPLITS, 4),
        "gate_pass_frac_clipbalanced": round(res["clip_pass"] / N_SPLITS, 4),
        "pop_risk_violation_frac_pooled": round(res["pop_pooled_viol"] / N_SPLITS, 4),
        "pop_risk_violation_frac_clipbalanced": round(res["pop_clip_viol"] / N_SPLITS, 4),
        "fp_per_h_mean": round(float(np.mean(res["fph"])), 1),
        "fp_per_h_std": round(float(np.std(res["fph"])), 1),
        "mean_lam": round(float(grid[int(np.mean(res["g_sel"]))]), 3),
        "runtime_s": round(time.time() - t0, 1),
    }
    print(tag, json.dumps(out, indent=None), flush=True)
    return out


def main():
    out = {}
    for tag, variant, matching, route, alpha in CONFIGS:
        out[tag] = run_config(tag, variant, matching, route, alpha)

    # miss at lam=0 per rule/decoder (U-shape endpoint)
    ends = {}
    for m in ("collar", "intersect50", "intersect70"):
        for v in ("medfilt", "csebb"):
            sfx = "" if m == "collar" else f"__{m}"
            p = os.path.join(CACHE, f"stats_test_{v}{sfx}.npz")
            if not os.path.exists(p):
                continue
            _, miss, fps, nref, _, grid = load_tensors(p)
            ends[f"{m}/{v}"] = round(float(miss.sum(0).sum(0)[0] / nref.sum()), 3)
    out["miss_at_lam0"] = ends
    print("miss@lam0:", ends, flush=True)

    # exact configuration count in the shared-split search (protocol rows)
    n_cfg = 0
    for p in glob.glob(os.path.join(RESULTS, "exp_*.json")):
        d = json.load(open(p))
        c = d["config"]
        r = c.get("route", "")
        if (c.get("cal_frac", 0.5) == 0.5 and c.get("seed_base", 0) == 0
                and "oracle" not in r and r not in ("valtuned", "fixed05")
                and not d["exp_id"].startswith(("VER", "sw-", "CS-", "DL-"))):
            n_cfg += 1
    out["n_calibrated_configs_in_search"] = n_cfg
    print("configs in search:", n_cfg, flush=True)

    with open(os.path.join(RESULTS, "instrument_study.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/instrument_study.json")


if __name__ == "__main__":
    main()
