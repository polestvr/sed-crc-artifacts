"""Reproduction gate: PSDS1-M on RealDESED test must hit 0.731 +- 0.01 with
cSEBBs post-processing tuned on validation (paper Table 2; secondary checks:
median filter ~0.693, raw ~0.662).

Also caches cSEBB confidence score curves for validation+test (variant inputs
for build_stats.py) and the tuned predictor parameters.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/sed-crc-work/RealDESED"))

from sed_crc.gt import CACHE, load_durations, load_gt, load_scores
from sebbs import csebbs
from utils.evaluation import compute_psds_metrics  # official repo eval helper


def medfilt(scores, classes, window=9):
    out = {}
    for cid, df in scores.items():
        d = df.copy()
        d[classes] = d[classes].rolling(window=window, center=True).median().bfill().ffill()
        out[cid] = d
    return out


def psds1_m(scores, gt, durations):
    (p1, per_cls), _ = compute_psds_metrics(scores, gt, durations)
    return p1, float(np.mean(list(per_cls.values()))), per_cls


def main():
    val_scores = load_scores("validation")
    val_gt = load_gt("validation")
    val_dur = load_durations("validation")
    test_scores = load_scores("test")
    test_gt = load_gt("test")
    test_dur = load_durations("test")
    # metadata.csv carries one extra row per split vs the official dataset
    # class (val 1000->999, test 1008->1007); restrict GT to extracted clips
    val_gt = {k: val_gt[k] for k in val_scores}
    test_gt = {k: test_gt[k] for k in test_scores}
    classes = [c for c in next(iter(val_scores.values())).columns if c not in ("onset", "offset")]

    print("tuning cSEBBs on validation (classwise, PSDS1 selection)...", flush=True)
    predictor, best_vals = csebbs.tune(
        val_scores, val_gt, val_dur,
        selection_fn=csebbs.select_best_psds,
        dtc_threshold=0.7, gtc_threshold=0.7, cttc_threshold=None,
        alpha_ct=0.0, unit_of_time="hour", max_efpr=100.0, classwise=True,
    )
    params = {
        "step_filter_length": predictor.step_filter_length,
        "merge_threshold_abs": {k: (None if np.isinf(v) else v) for k, v in predictor.merge_threshold_abs.items()},
        "merge_threshold_rel": {k: (None if np.isinf(v) else v) for k, v in predictor.merge_threshold_rel.items()},
        "val_best_psds1": best_vals,
    }
    with open(os.path.join(CACHE, "csebb_params.json"), "w") as f:
        json.dump(params, f, indent=1, default=float)
    print("tuned params saved.", flush=True)

    for split, scores in (("validation", val_scores), ("test", test_scores)):
        sebb_scores = predictor.predict(scores, return_sed_scores=True)
        with open(os.path.join(CACHE, f"{split}_csebb_scores.pkl"), "wb") as f:
            pickle.dump(sebb_scores, f, protocol=4)
    print("cSEBB score curves cached.", flush=True)

    with open(os.path.join(CACHE, "test_csebb_scores.pkl"), "rb") as f:
        test_csebb = pickle.load(f)

    results = {}
    for name, sc in (
        ("raw", test_scores),
        ("medfilt", medfilt(test_scores, classes)),
        ("csebb", test_csebb),
    ):
        overall, macro, per_cls = psds1_m(sc, test_gt, test_dur)
        results[name] = {"psds1": overall, "psds1_macro": macro,
                         "per_class": {k: float(v) for k, v in per_cls.items()}}
        print(f"{name:8s} PSDS1={overall:.4f} PSDS1-M={macro:.4f}", flush=True)

    results["gate"] = {
        "target": 0.731, "tolerance": 0.01,
        "achieved": results["csebb"]["psds1_macro"],
        "pass": bool(abs(results["csebb"]["psds1_macro"] - 0.731) <= 0.01),
        "secondary_medfilt_expected": 0.693,
        "secondary_raw_expected": 0.662,
    }
    out = os.path.join(CACHE, "gate_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("GATE:", "PASS" if results["gate"]["pass"] else "FAIL",
          f"(csebb PSDS1-M {results['csebb']['psds1_macro']:.4f} vs 0.731 +- 0.01)")


if __name__ == "__main__":
    main()
