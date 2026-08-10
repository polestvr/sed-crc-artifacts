"""Leave-one-environment-out calibration.

The natural new-home simulation, distinct from the subgroup slicing of
stress_subgroups.py (where subgroup members sit in BOTH halves): calibrate the
champion route on every clip NOT in group g, deploy on the clips in g. A clip
can carry several environment tags, so "held-out environment e" means "clips
whose environment list contains e" and calibration is its complement -- stated
as such, not as a clean k-1 partition.

Champion configs at all three operating points; ltt_split repeats R times over
its internal ordering/testing split (the only stochastic element -- the
cal/deploy partition is deterministic given g), Bonferroni is deterministic.
"""
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
from utils.evaluation import categorize_recording_device

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

R_REPEATS = 50
TOP_ENVS = 6

CONFIGS = [
    ("i20", "csebbmax", "intersect50", "ltt_split_clipmean", 0.2),
    ("i45", "csebb",    "intersect70", "ltt_bonf_clipmean",  0.45),
    ("c60", "csebb",    "collar",      "ltt_bonf_clipmean",  0.6),
]


def env_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(";") if x.strip()]
    return list(v)


def main():
    out = {}
    for tag, variant, matching, route, alpha in CONFIGS:
        sfx = "" if matching == "collar" else f"__{matching}"
        clip_ids, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        dur = load_durations("test")
        meta = load_meta("test")
        durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
        C = miss.shape[0]
        miss_k, fps_k, nref_k = miss.sum(1), fps.sum(1), nref.sum(1)
        route_fn = ROUTES[route]

        groups = {}
        dev = np.array([categorize_recording_device(
            meta[c].get("recording_device")) for c in clip_ids])
        for cat in ("ios", "android", "other"):
            groups[f"device/{cat}"] = dev == cat
        plc = np.array([(meta[c].get("device_placement") or "").strip().lower()
                        for c in clip_ids])
        for cat in ("static", "mobile"):
            groups[f"placement/{cat}"] = plc == cat
        env_lists = [env_list(meta[c].get("recording_environment"))
                     for c in clip_ids]
        env_counts = Counter(e for lst in env_lists for e in lst)
        for env, _ in env_counts.most_common(TOP_ENVS):
            groups[f"env/{env}"] = np.array([env in lst for lst in env_lists])

        res = {}
        for gname, gmask in groups.items():
            held, cal = gmask, ~gmask
            n_ev = int(nref_k[held].sum())
            if n_ev == 0 or cal.sum() < 50:
                continue
            miss_shares, fphs = [], []
            for r in range(R_REPEATS):
                rng = np.random.default_rng(20_000 + r)
                g = route_fn(miss_k[cal], nref_k[cal], alpha, delta=0.05,
                             rng=rng)
                miss_shares.append(miss_k[held, g].sum() / n_ev)
                fphs.append(fps_k[held, g].sum() / durations_h[held].sum())
            a = np.array(miss_shares)
            res[gname] = {
                "n_clips_held": int(held.sum()),
                "n_events_held": n_ev,
                "n_clips_cal": int(cal.sum()),
                "mean_miss": round(float(a.mean()), 4),
                "std_miss": round(float(a.std()), 4),
                "frac_repeats_ok": round(float((a <= alpha).mean()), 3),
                "fp_per_h_mean": round(float(np.mean(fphs)), 1),
            }
            print(f"{tag} {gname}: miss {res[gname]['mean_miss']}"
                  f"±{res[gname]['std_miss']} (alpha {alpha}, "
                  f"ok {res[gname]['frac_repeats_ok']}), "
                  f"held {res[gname]['n_clips_held']} clips", flush=True)
        out[tag] = {"config": dict(variant=variant, matching=matching,
                                   route=route, alpha=alpha,
                                   repeats=R_REPEATS),
                    "groups": res}

    with open(os.path.join(RESULTS, "loo_environment.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/loo_environment.json")


if __name__ == "__main__":
    main()
