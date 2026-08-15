"""
Dataset preparation for AeroSense.

Source dataset: TII UAV Propeller Anomaly Audio Dataset (ICASSP 2022)
https://github.com/tiiuae/UAV-Propeller-Anomaly-Audio-Dataset
Downloaded via `git clone` into data/raw/{class}/... (see README for exact steps).

This script:
  1. Walks data/raw/<class_name>/... and finds every .wav file.
  2. Treats each .wav file as one "recording" (one indoor flight mission).
  3. Cuts each recording into fixed-length segments (common/config.py params).
  4. Splits recordings (NOT segments) into train/val/test — this is the
     leakage-prevention step: every segment from a given recording ends up
     in exactly one split.
  5. Writes a manifest CSV (segment-level rows: file path + sample offset +
     label + split) to data/processed/manifest.csv. Spectrograms are
     computed on-the-fly in training/dataset.py, not pre-rendered to disk,
     to avoid multi-GB intermediate storage.
  6. Computes log-mel mean/std from the TRAIN split only and writes
     models/feature_stats.json (used to standardize inputs at both train
     and inference time).

Usage:
    python -m training.prepare_dataset
"""
import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import AUDIO_CONFIG, CLASS_NAMES, DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, RANDOM_SEED
from common.audio_features import load_audio, segment_waveform, normalize_waveform, generate_mel_spectrogram


def find_recordings(raw_dir: str) -> list[dict]:
    """Return one entry per .wav file with its class label and a unique recording_id."""
    recordings = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] class dir missing: {class_dir}")
            continue
        for root, _dirs, files in os.walk(class_dir):
            for fname in files:
                if fname.lower().endswith(".wav"):
                    fpath = os.path.join(root, fname)
                    rec_id = os.path.relpath(fpath, raw_dir)
                    recordings.append({"path": fpath, "label": class_name, "recording_id": rec_id})
    return recordings


def group_split(recordings: list[dict], train_ratio=0.7, val_ratio=0.15, seed=RANDOM_SEED):
    """
    Split at the RECORDING level, stratified by class, so that no segment
    from a given recording ever appears in more than one split
    (prevents data leakage from overlapping/adjacent segments of the same
    flight mission).
    """
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}
    for r in recordings:
        by_class[r["label"]].append(r)

    splits = {"train": [], "val": [], "test": []}
    for cls, recs in by_class.items():
        recs = recs[:]
        rng.shuffle(recs)
        n = len(recs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_val = min(n_val, n - n_train - 1) if n - n_train - 1 >= 0 else 0
        train_recs = recs[:n_train]
        val_recs = recs[n_train:n_train + n_val]
        test_recs = recs[n_train + n_val:]
        if not test_recs:  # guarantee every split has something for tiny classes
            test_recs = [val_recs.pop()] if val_recs else [train_recs.pop()]
        splits["train"].extend(train_recs)
        splits["val"].extend(val_recs)
        splits["test"].extend(test_recs)
    return splits


def build_manifest(splits: dict) -> list[dict]:
    rows = []
    seg_counter = 0
    for split_name, recs in splits.items():
        for rec in recs:
            try:
                audio = load_audio(rec["path"])
            except Exception as e:
                print(f"[SKIP] could not decode {rec['path']}: {e}")
                continue
            seg_len = AUDIO_CONFIG.segment_samples
            hop = AUDIO_CONFIG.hop_samples
            n_segments = 0
            start = 0
            while start < len(audio):
                end = start + seg_len
                if end <= len(audio) or (len(audio) - start) >= seg_len * 0.5:
                    rows.append({
                        "segment_id": seg_counter,
                        "path": rec["path"],
                        "recording_id": rec["recording_id"],
                        "start_sample": start,
                        "label": rec["label"],
                        "split": split_name,
                    })
                    seg_counter += 1
                    n_segments += 1
                if end > len(audio):
                    break
                start += hop
            print(f"[{split_name}] {rec['recording_id']}: {n_segments} segments ({len(audio)/AUDIO_CONFIG.sample_rate:.1f}s)")
    return rows


def compute_train_stats(rows: list[dict]) -> tuple[float, float]:
    """Compute log-mel mean/std over a sample of TRAIN segments only."""
    train_rows = [r for r in rows if r["split"] == "train"]
    sample = train_rows if len(train_rows) <= 400 else random.Random(RANDOM_SEED).sample(train_rows, 400)

    sums, sq_sums, count = 0.0, 0.0, 0
    audio_cache = {}
    for r in sample:
        if r["path"] not in audio_cache:
            audio_cache[r["path"]] = load_audio(r["path"])
            if len(audio_cache) > 8:  # keep memory bounded
                audio_cache.pop(next(iter(audio_cache)))
        audio = audio_cache[r["path"]]
        seg = audio[r["start_sample"]: r["start_sample"] + AUDIO_CONFIG.segment_samples]
        if len(seg) < AUDIO_CONFIG.segment_samples:
            pad = np.zeros(AUDIO_CONFIG.segment_samples, dtype=np.float32)
            pad[: len(seg)] = seg
            seg = pad
        seg = normalize_waveform(seg)
        mel = generate_mel_spectrogram(seg)
        sums += mel.sum()
        sq_sums += (mel ** 2).sum()
        count += mel.size

    mean = sums / count
    var = sq_sums / count - mean ** 2
    std = float(np.sqrt(max(var, 1e-8)))
    return float(mean), std


def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    recordings = find_recordings(DATA_RAW_DIR)
    print(f"Found {len(recordings)} recordings across {len(CLASS_NAMES)} classes")
    for c in CLASS_NAMES:
        n = sum(1 for r in recordings if r["label"] == c)
        print(f"  {c}: {n} recordings")

    if not recordings:
        print("ERROR: no recordings found. Did you download the dataset into data/raw/?")
        sys.exit(1)

    splits = group_split(recordings)
    for s, recs in splits.items():
        print(f"split={s}: {len(recs)} recordings")

    rows = build_manifest(splits)
    manifest_path = os.path.join(DATA_PROCESSED_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "path", "recording_id", "start_sample", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest: {manifest_path} ({len(rows)} segments)")

    for s in ["train", "val", "test"]:
        n = sum(1 for r in rows if r["split"] == s)
        print(f"  segments[{s}] = {n}")

    mean, std = compute_train_stats(rows)
    stats_path = os.path.join(MODEL_DIR, "feature_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "log_mel_mean": mean,
            "log_mel_std": std,
            "computed_from": "train split only, sample of up to 400 segments",
            "audio_config": {
                "sample_rate": AUDIO_CONFIG.sample_rate,
                "n_mels": AUDIO_CONFIG.n_mels,
                "n_fft": AUDIO_CONFIG.n_fft,
                "hop_length": AUDIO_CONFIG.hop_length,
                "segment_duration": AUDIO_CONFIG.segment_duration,
            },
        }, f, indent=2)
    print(f"Wrote feature stats: {stats_path} (mean={mean:.3f}, std={std:.3f})")


if __name__ == "__main__":
    main()
