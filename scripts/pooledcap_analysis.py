"""Companion analysis for the pooled-cap route: over the 100 protocol
splits, record the cap K chosen by half A, the Clopper-Pearson tail bound
q_up, the observed tail frequencies and tail EVENT share on the evaluation
half, and the realized capped vs full pooled miss at the selected threshold.
Replicates the route internals with the protocol rngs so numbers match the
run_experiment gate rows exactly."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations
from sed_crc.stats import load_tensors
from sed_crc.routes import ltt_split_pooledcap, pooledcap_tail_bound, _split
from sed_crc.evalx import make_splits

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

CONFIGS = [
    ("i20", "csebbmax", "intersect50", 0.2),
    ("i45", "csebb",    "intersect70", 0.45),
    ("c60", "csebb",    "collar",      0.6),
]
DELTA = 0.05
TAIL_FRAC = 0.2


def main():
    out = {}
    for tag, variant, matching, alpha in CONFIGS:
        sfx = "" if matching == "collar" else f"__{matching}"
        clip_ids, miss, fps, nref, classes, grid = load_tensors(
            os.path.join(CACHE, f"stats_test_{variant}{sfx}.npz"))
        dur = load_durations("test")
        durations_h = np.array([dur[c] for c in clip_ids]) / 3600.0
        C = miss.shape[0]
        miss_k, fps_k, nref_k = miss.sum(1), fps.sum(1), nref.sum(1)

        rec = {k: [] for k in ("K", "q_up", "tail_freq_ev", "tail_event_share_ev",
                               "miss_full_ev", "miss_capped_ev", "fph", "n_ok")}
        n_ok, n_fallback = 0, 0
        p_min_all = []
        for si, (cal, ev) in enumerate(make_splits(C, 100, 0.5)):
            rng = np.random.default_rng(10_000 + si)
            g = ltt_split_pooledcap(miss_k[cal], nref_k[cal], alpha,
                                    delta=DELTA, rng=rng,
                                    delta_tail_frac=TAIL_FRAC)
            # replicate A/B split and cap with the SAME rng sequence
            rng2 = np.random.default_rng(10_000 + si)
            A, Bh = _split(rng2, len(cal))
            K = max(int(nref_k[cal][A].max()), 1)
            q_up = pooledcap_tail_bound(nref_k[cal][Bh], K, DELTA * TAIL_FRAC)
            # power diagnostic: minimal achievable p over the whole grid
            nB_, mB_ = nref_k[cal][Bh], miss_k[cal][Bh]
            inK_ = nB_ <= K
            nK_ = int(inK_.sum())
            Xall = mB_[inK_] - alpha * nB_[inK_][:, None]      # [nK, G]
            xbar = Xall.mean(0)
            pgrid = np.where(xbar < 0,
                             np.exp(-2.0 * nK_ * xbar * xbar / (K * K)), 1.0)
            p_min = float(pgrid.min())
            p_min_all.append(p_min)
            if p_min > DELTA * (1.0 - TAIL_FRAC):
                n_fallback += 1
            evn, evm = nref_k[ev], miss_k[ev, g]
            inK = evn <= K
            rec["K"].append(K)
            rec["q_up"].append(q_up)
            rec["tail_freq_ev"].append(float((~inK).mean()))
            rec["tail_event_share_ev"].append(
                float(evn[~inK].sum() / max(evn.sum(), 1)))
            rec["miss_full_ev"].append(float(evm.sum() / max(evn.sum(), 1)))
            rec["miss_capped_ev"].append(
                float(evm[inK].sum() / max(evn[inK].sum(), 1)))
            rec["fph"].append(float(fps_k[ev, g].sum() / durations_h[ev].sum()))
            if evm.sum() <= alpha * evn.sum():
                n_ok += 1
        out[tag] = {
            "config": dict(variant=variant, matching=matching, alpha=alpha,
                           delta=DELTA, delta_tail_frac=TAIL_FRAC),
            "K_mean": round(float(np.mean(rec["K"])), 1),
            "K_min": int(np.min(rec["K"])), "K_max": int(np.max(rec["K"])),
            "q_up_mean": round(float(np.mean(rec["q_up"])), 4),
            "tail_freq_ev_mean": round(float(np.mean(rec["tail_freq_ev"])), 4),
            "tail_event_share_ev_mean":
                round(float(np.mean(rec["tail_event_share_ev"])), 4),
            "miss_full_ev_mean": round(float(np.mean(rec["miss_full_ev"])), 4),
            "miss_capped_ev_mean":
                round(float(np.mean(rec["miss_capped_ev"])), 4),
            "fp_per_h_mean": round(float(np.mean(rec["fph"])), 1),
            "n_ok_pooled_gate": n_ok,
            "fallback_frac": round(n_fallback / 100.0, 3),
            "p_min_mean": round(float(np.mean(p_min_all)), 4),
            "p_min_min": round(float(np.min(p_min_all)), 4),
        }
        print(tag, out[tag], flush=True)

    with open(os.path.join(RESULTS, "pooledcap_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/pooledcap_analysis.json")


if __name__ == "__main__":
    main()
