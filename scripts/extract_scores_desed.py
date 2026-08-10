"""Score DESED public eval audio with the frozen RealDESED baseline.

Produces sed_scores dataframes (RealDESED's 15 class columns) + durations for
the cross-corpus stress test. Ground truth mapping happens in the stress script.
"""
import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio

REPO = os.path.expanduser("~/sed-crc-work/RealDESED")
sys.path.insert(0, REPO)
os.chdir(REPO)

from dataset.dataset import RealDESED_CLASSES
from train import PLModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio_dir", default=os.path.expanduser(
        "~/data/desed_public_eval/dataset/audio/eval/public"))
    ap.add_argument("--out", default=os.path.expanduser("~/sed-crc-work/cache"))
    args = ap.parse_args()

    frame_hz = 25
    cfg = SimpleNamespace(
        chunk_size=10.0, hop_size=5.0, sample_rate=16000, inference_batch_size=64,
        sliding_window_stitching="average", triangular_filter_floor=0.3,
        median_window=9, mixup_p=0.5, mixup_alpha=0.2, freq_warp_p=0.5,
        train_annotation_aggregation="Weighted Soft",
        dataset_path="unused", output_dir="unused", experiment_name="extract",
    )
    model = PLModule(cfg, RealDESED_CLASSES, [], pretrained_checkpoint="ATST-F_strong_1",
                     frame_hz=frame_hz)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()

    from utils.evaluation import preds_to_score_df

    wavs = sorted(f for f in os.listdir(args.audio_dir) if f.endswith(".wav"))
    print(f"{len(wavs)} wav files")
    scores, durations = {}, {}
    with torch.no_grad():
        for i, fn in enumerate(wavs):
            wav, sr = torchaudio.load(os.path.join(args.audio_dir, fn))
            wav = wav.mean(0, keepdim=True)
            if sr != cfg.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
            duration = wav.shape[1] / cfg.sample_rate
            logits = model.sliding_window_inference(wav.cuda())
            n = logits.shape[1]
            ts = torch.tensor((np.arange(n) + 0.5) / frame_hz, dtype=torch.float32)
            probs = torch.sigmoid(logits)
            df = preds_to_score_df(probs, ts, RealDESED_CLASSES, frame_hz=frame_hz)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            cid = os.path.splitext(fn)[0]
            scores[cid] = df
            durations[cid] = float(duration)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(wavs)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "desed_scores.pkl"), "wb") as f:
        pickle.dump(scores, f, protocol=4)
    with open(os.path.join(args.out, "desed_durations.json"), "w") as f:
        json.dump(durations, f)
    print("saved", len(scores), "clips")


if __name__ == "__main__":
    main()
