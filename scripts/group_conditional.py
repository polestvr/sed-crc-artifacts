"""Group-conditional (Mondrian-style) calibration.

Instead of one marginal threshold, calibrate a separate threshold per metadata
group (device placement; device category; primary recording environment) on
the calibration half restricted to that group, deploy group-wise, and measure
per-group validity on the evaluation half -- the direct repair candidate for
the subgroup failures in the stress tables. Small-group fallback (no rejection
or <20 calibration clips): calibration-half risk minimizer, uncertified.
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

N_SPLITS = 100
TOP_ENVS = 5


def env_primary(v):
    if v is None:
        return "other_env"
    lst = [x.strip() for x in v.split(";")] if isinstance(v, str) else list(v)
    return lst[0] if lst else "other_env"


def assignments(clip_ids, meta):
    plc = np.array([(meta[c].get("device_placement") or "unknown").strip().lower()
                    for c in clip_ids])
    dev = np.array([categorize_recording_device(meta[c].get("recording_device"))
                    for c in clip_ids])
    envs = [env_primary(meta[c].get("recording_environment")) for c in clip_ids]
    top = [e for e, _ in Counter(envs).most_common(TOP_ENVS)]
    env = np.array([e if e in top else "other_env" for e in envs])
    return {"placement": plc, "device": dev, "environment": env}


def run_scheme(tag, variant, matching, alpha, route, scheme_name, groups_vec,
               miss, fps, nref, durations_h):
    C, K, G = miss.shape
    miss_k, nref_k = miss.sum(1), nref.sum(1)
    route_fn = ROUTES[route]
    labels = sorted(set(groups_vec))
    per_group = {g: {"n_ok": 0, "miss": []} for g in labels}
    overall = {"n_ok": 0, "miss": [], "fph": [], "fallbacks": 0}
    half = C // 2
    for si in range(N_SPLITS):
        rng = np.random.default_rng(si)  # SAME splits as make_splits(seed_base=0)
        perm = rng.permutation(C)
        cal, ev = perm[:half], perm[half:]
        lam_clip = np.zeros(C, np.int64)  # per-clip lambda index via its group
        for g in labels:
            gm = groups_vec == g
            cal_g = cal[gm[cal]]
            if len(cal_g) < 20:
                gsel = int(np.argmin(miss_k[cal].sum(0))) if len(cal_g) == 0 else \
                    int(np.argmin(miss_k[cal_g].sum(0)))
                overall["fallbacks"] += 1
            else:
                gsel = route_fn(miss_k[cal_g], nref_k[cal_g], alpha, delta=0.05,
                                rng=np.random.default_rng(80_000 + si))
            lam_clip[gm] = gsel
        ev_miss = miss[ev, :, :][np.arange(len(ev))[:, None],
                                 np.arange(K)[None, :], lam_clip[ev][:, None]]
        ev_fps = fps[ev, :, :][np.arange(len(ev))[:, None],
                               np.arange(K)[None, :], lam_clip[ev][:, None]]
        tot = nref[ev].sum()
        share = ev_miss.sum() / max(tot, 1)
        overall["miss"].append(float(share))
        if ev_miss.sum() <= alpha * tot:
            overall["n_ok"] += 1
        overall["fph"].append(float(ev_fps.sum() / durations_h[ev].sum()))
        for g in labels:
            gm_ev = groups_vec[ev] == g
            n_ev = nref[ev][gm_ev].sum()
            if n_ev == 0:
                continue
            m_ev = ev_miss[gm_ev].sum()
            per_group[g]["miss"].append(float(m_ev / n_ev))
            if m_ev <= alpha * n_ev:
                per_group[g]["n_ok"] += 1
    res = {
        "scheme": scheme_name, "route": route,
        "overall_n_ok": overall["n_ok"],
        "overall_mean_miss": round(float(np.mean(overall["miss"])), 4),
        "fp_per_h_mean": round(float(np.mean(overall["fph"])), 1),
        "fallback_events": overall["fallbacks"],
        "groups": {g: {"n_ok": per_group[g]["n_ok"],
                       "n_splits_scored": len(per_group[g]["miss"]),
                       "mean_miss": round(float(np.mean(per_group[g]["miss"])), 4)
                       if per_group[g]["miss"] else None}
                   for g in labels},
    }
    print(tag, scheme_name, json.dumps(res["groups"]), flush=True)
    return res


def main():
    out = {}
    for tag, variant, matching, alpha in (("i20", "csebbmax", "intersect50", 0.2),
                                          ("c60", "csebb", "collar", 0.6)):
        sfx = "" if matching == "collar" else f"__{matching}"
        clip_ids, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        meta = load_meta("test")
        dur = load_durations("test")
        durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
        schemes = assignments(clip_ids, meta)
        out[tag] = {}
        for name, vec in schemes.items():
            out[tag][name] = run_scheme(tag, variant, matching, alpha,
                                        "ltt_bonf_clipmean", name, vec,
                                        miss, fps, nref, durations_h)
    with open(os.path.join(RESULTS, "group_conditional.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/group_conditional.json")


if __name__ == "__main__":
    main()
