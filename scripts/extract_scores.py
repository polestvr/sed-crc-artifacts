"""Extract frame-level SED score curves from the frozen RealDESED baseline.

Standalone replication of PLModule._shared_eval_step's inference path
(sliding-window, average stitching, triangular weights) that caches
sed_scores_eval-format score dataframes to disk, so that ALL downstream
experiments are inference-free numpy on cached curves.

Ground-truth labels are deliberately NOT touched here; they enter only
via sed_scores_eval at evaluation time.
"""
import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace

import torch

REPO = os.path.expanduser("~/sed-crc-work/RealDESED")
sys.path.insert(0, REPO)
os.chdir(REPO)  # RESOURCES_FOLDER is relative

from torch.utils.data import DataLoader
from dataset.collate import collate_fn
from dataset.dataset import RealDESEDDataset, RealDESED_CLASSES
from train import PLModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/sed-crc-work/cache"))
    ap.add_argument("--dataset_path", default=os.path.expanduser("~/data/realdesed"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    frame_hz = 25
    cfg = SimpleNamespace(
        chunk_size=10.0, hop_size=5.0, sample_rate=16000, inference_batch_size=64,
        sliding_window_stitching="average", triangular_filter_floor=0.3,
        median_window=9, mixup_p=0.5, mixup_alpha=0.2, freq_warp_p=0.5,
        train_annotation_aggregation="Weighted Soft",
        dataset_path=args.dataset_path, output_dir="unused", experiment_name="extract",
    )

    ds = RealDESEDDataset(
        root=args.dataset_path, split=args.split, sample_rate=cfg.sample_rate,
        frame_hz=frame_hz, classes=RealDESED_CLASSES,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    model = PLModule(cfg, RealDESED_CLASSES, [], pretrained_checkpoint="ATST-F_strong_1",
                     frame_hz=frame_hz)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()

    from utils.evaluation import preds_to_score_df

    scores, durations, meta = {}, {}, {}
    n_done = 0
    with torch.no_grad():
        for batch in dl:
            audios = batch["audio"]
            timestamps = batch["timestamps"]
            filenames = batch["filename"]
            durs = batch["duration"]
            metadata_batch = batch["metadata"]

            for i in range(audios.shape[0]):
                ts = timestamps[i]
                valid_len = (ts >= 0).sum().item()
                ts = ts[:valid_len]

                frame_duration = 1.0 / frame_hz
                audio_len = int((ts[-1].item() + 0.5 * frame_duration) * cfg.sample_rate)
                audio = audios[i, :, :audio_len].cuda()

                logits = model.sliding_window_inference(audio)
                min_len = min(logits.shape[1], valid_len)
                logits = logits[:, :min_len]
                ts_aligned = ts[:min_len]

                probs = torch.sigmoid(logits)
                score_df = preds_to_score_df(probs, ts_aligned, RealDESED_CLASSES, frame_hz=frame_hz)
                score_df = score_df.sort_index()
                score_df = score_df[~score_df.index.duplicated(keep="first")]

                file_id = filenames[i].replace(".wav", "")
                scores[file_id] = score_df.copy()
                durations[file_id] = durs[i].item() if isinstance(durs, torch.Tensor) else durs[i]
                md = metadata_batch[i]
                meta[file_id] = {
                    "device_placement": md.get("device_placement"),
                    "recording_device": md.get("recording_device"),
                    "recording_environment": md.get("recording_environment"),
                }
                n_done += 1
            print(f"{n_done}/{len(ds)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.split}_scores.pkl"), "wb") as f:
        pickle.dump(scores, f, protocol=4)
    with open(os.path.join(args.out, f"{args.split}_durations.json"), "w") as f:
        json.dump(durations, f)
    with open(os.path.join(args.out, f"{args.split}_meta.json"), "w") as f:
        json.dump(meta, f, default=str)
    print("saved", args.split, len(scores), "clips ->", args.out)


if __name__ == "__main__":
    main()
