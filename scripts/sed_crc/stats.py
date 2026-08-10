"""Per-clip, per-class collar-based miss/FP counts on a fixed threshold grid.

Built once per (split, decoding variant) from sed_scores_eval's
collar_based.intermediate_statistics_deltas (the untouched eval machinery:
unique bipartite matching, collar rule max(0.2, 0.5*dur) on offsets).
Every experiment afterwards is pure numpy on the cached tensors.

Semantics: detection = score > threshold. stat(t) = sum of deltas at change
points cp > t (a change applies once the threshold falls below its cp).
Differentially validated against the library's own accumulation in build_stats.
"""
import numpy as np
from sed_scores_eval import collar_based, intersection_based

from .gt import COLLAR

GRID = np.linspace(0.0, 1.0, 1001)

# matching rule -> per-clip deltas function kwargs
MATCHINGS = {
    "collar": (collar_based.intermediate_statistics_deltas, COLLAR),
    "onset": (collar_based.intermediate_statistics_deltas,
              dict(onset_collar=0.2, offset_collar=1e6, offset_collar_rate=0.0)),
    "intersect70": (intersection_based.intermediate_statistics_deltas,
                    dict(dtc_threshold=0.7, gtc_threshold=0.7, cttc_threshold=None)),
    "intersect50": (intersection_based.intermediate_statistics_deltas,
                    dict(dtc_threshold=0.5, gtc_threshold=0.5, cttc_threshold=None)),
}


def clip_stats_tensors(scores, gt, classes, grid=GRID, num_jobs=8, matching="collar"):
    """Returns clip_ids (list), miss[c,k,g] int32, fps[c,k,g] int32, nref[c,k] int32."""
    fn, kw = MATCHINGS[matching]
    deltas = fn(scores, gt, num_jobs=num_jobs, **kw)
    clip_ids = sorted(scores.keys())
    C, K, G = len(clip_ids), len(classes), len(grid)
    miss = np.zeros((C, K, G), np.int32)
    fps = np.zeros((C, K, G), np.int32)
    nref = np.zeros((C, K), np.int32)
    for ci, cid in enumerate(clip_ids):
        n_by_class = {}
        for on, off, lbl in gt[cid]:
            n_by_class[lbl] = n_by_class.get(lbl, 0) + 1
        for ki, cls in enumerate(classes):
            nref[ci, ki] = n_by_class.get(cls, 0)
            cp, d = deltas[cid][cls]
            if len(cp) == 0:
                miss[ci, ki, :] = nref[ci, ki]
                continue
            order = np.argsort(cp, kind="stable")
            cp_s = np.asarray(cp)[order]
            dtp = np.asarray(d["tps"])[order]
            dfp = np.asarray(d["fps"])[order]
            # stat(t) = sum of deltas with cp > t  -> suffix sums
            suf_tp = np.concatenate([np.cumsum(dtp[::-1])[::-1], [0.0]])
            suf_fp = np.concatenate([np.cumsum(dfp[::-1])[::-1], [0.0]])
            idx = np.searchsorted(cp_s, grid, side="right")
            tps_g = suf_tp[idx]
            fps_g = suf_fp[idx]
            miss[ci, ki, :] = nref[ci, ki] - np.rint(tps_g).astype(np.int32)
            fps[ci, ki, :] = np.rint(fps_g).astype(np.int32)
    assert (miss >= 0).all(), "tps exceeded n_ref somewhere - semantics bug"
    return clip_ids, miss, fps, nref


def save_tensors(path, clip_ids, miss, fps, nref, classes, grid=GRID):
    np.savez_compressed(
        path, clip_ids=np.array(clip_ids), miss=miss, fps=fps, nref=nref,
        classes=np.array(classes), grid=grid,
    )


def load_tensors(path):
    z = np.load(path, allow_pickle=False)
    return (
        [str(x) for x in z["clip_ids"]], z["miss"], z["fps"], z["nref"],
        [str(x) for x in z["classes"]], z["grid"],
    )
