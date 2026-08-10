"""Ground-truth and cache I/O.

Labels are only ever consumed by sed_scores_eval functions downstream;
this module is mechanical plumbing (same role as RealDESED's dataset code).
"""
import json
import os
import pickle

import pandas as pd

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CACHE = os.environ.get("SED_CRC_CACHE", os.path.join(_REPO, "cache"))
DATA = os.environ.get("SED_CRC_DATA", os.path.expanduser("~/data/realdesed"))

# sed_eval / DESED collar convention (user-specified rule)
COLLAR = dict(onset_collar=0.2, offset_collar=0.2, offset_collar_rate=0.5)


def load_gt(split, data_root=DATA):
    """annotations.csv -> {clip_id: [(onset, offset, label), ...]} (end>start only,
    mirroring utils.evaluation.events_to_tuples), including clips with no events."""
    df = pd.read_csv(os.path.join(data_root, split, "annotations.csv"))
    meta = pd.read_csv(os.path.join(data_root, split, "metadata.csv"))
    gt = {os.path.splitext(fn)[0]: [] for fn in meta["filename"]}
    for fn, cls, on, off in zip(df["filename"], df["class"], df["onset"], df["offset"]):
        if float(off) > float(on):
            gt[os.path.splitext(fn)[0]].append((float(on), float(off), str(cls)))
    return gt


def load_scores(split, cache=CACHE):
    with open(os.path.join(cache, f"{split}_scores.pkl"), "rb") as f:
        return pickle.load(f)


def load_durations(split, cache=CACHE):
    with open(os.path.join(cache, f"{split}_durations.json")) as f:
        return json.load(f)


def load_meta(split, cache=CACHE):
    with open(os.path.join(cache, f"{split}_meta.json")) as f:
        return json.load(f)
