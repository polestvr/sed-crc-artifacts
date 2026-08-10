"""Re-score cSEBB boxes with frame-based event scores (dim-1 ablation):
same box extents, confidence replaced by (a) max frame score, (b) mean of
top-k frame scores within the box. Saves score-curve pkls for build_stats.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sed_crc.gt import CACHE, load_durations, load_scores
from sebbs.csebbs import CSEBBsPredictor
from sebbs.utils import sed_scores_from_sebbs


def rescore(boxes, raw_df, classes, k):
    """boxes: list of (onset, offset, class, conf); returns rescored list."""
    onsets = raw_df["onset"].to_numpy()
    out = []
    for on, off, cls, conf in boxes:
        mask = (onsets >= on - 1e-9) & (onsets < off - 1e-9)
        vals = raw_df.loc[mask, cls].to_numpy()
        if len(vals) == 0:
            new = conf
        elif k == 1:
            new = float(np.max(vals))
        else:
            new = float(np.mean(np.sort(vals)[-k:]))
        out.append((on, off, cls, new))
    return out


def main():
    params = json.load(open(os.path.join(CACHE, "csebb_params.json")))
    inf = float("inf")
    predictor = CSEBBsPredictor(
        step_filter_length=params["step_filter_length"],
        merge_threshold_abs={c: (inf if v is None else v)
                             for c, v in params["merge_threshold_abs"].items()},
        merge_threshold_rel={c: (inf if v is None else v)
                             for c, v in params["merge_threshold_rel"].items()},
    )
    for split in ("test", "validation"):
        raw = load_scores(split)
        durs = load_durations(split)
        classes = [c for c in next(iter(raw.values())).columns if c not in ("onset", "offset")]
        sebbs_pred = predictor.predict(raw, return_sed_scores=False)
        for name, k in (("csebbmax", 1), ("csebbtopk3", 3)):
            rescored = {cid: rescore(sebbs_pred[cid], raw[cid], classes, k)
                        for cid in raw}
            dfs = sed_scores_from_sebbs(rescored, sound_classes=classes,
                                        audio_duration=durs)
            with open(os.path.join(CACHE, f"{split}_{name}_scores.pkl"), "wb") as f:
                pickle.dump(dfs, f, protocol=4)
            print(split, name, "saved", flush=True)


if __name__ == "__main__":
    main()
