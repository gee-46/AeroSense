"""
Train the AeroSense convolutional autoencoder for anomaly detection.

Trained ONLY on segments labeled `normal` from the train split (an
autoencoder for anomaly detection must never see anomalous examples during
training — that's the whole point: it should reconstruct normal acoustic
patterns well and reconstruct unfamiliar (faulty) patterns poorly).

The anomaly threshold itself is NOT chosen here. It is fit afterwards by
evaluation/evaluate.py from the distribution of reconstruction errors on
the validation set (normal + faulty segments) and written to
models/registry.json.

Usage:
    python -m training.train_autoencoder --epochs 15
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import MODEL_DIR, DATA_PROCESSED_DIR, RANDOM_SEED
from training.dataset import AeroSenseSegmentDataset
from backend.app.ml.models.autoencoder import ConvAutoencoder


class NormalOnlyDataset(AeroSenseSegmentDataset):
    """Same manifest loader, filtered to label == 'normal'."""

    def __init__(self, manifest_path, split, augment=False):
        super().__init__(manifest_path, split, augment=augment)
        self.rows = [r for r in self.rows if r["label"] == "normal"]

    def __getitem__(self, idx):
        x, _label = super().__getitem__(idx)
        return x  # autoencoder doesn't need the class label


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget-seconds", type=float, default=0)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    manifest_path = os.path.join(DATA_PROCESSED_DIR, "manifest.csv")
    train_ds = NormalOnlyDataset(manifest_path, "train", augment=True)
    val_ds = NormalOnlyDataset(manifest_path, "val", augment=False)
    print(f"train normal segments={len(train_ds)}  val normal segments={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConvAutoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "aerosense_autoencoder_v1.pt")
    last_ckpt_path = os.path.join(MODEL_DIR, "aerosense_autoencoder_last.pt")
    state_path = os.path.join(MODEL_DIR, "autoencoder_trainer_state.json")
    log_path = os.path.join(MODEL_DIR, "autoencoder_training_log.json")

    start_epoch = 1
    if args.resume and os.path.exists(state_path) and os.path.exists(last_ckpt_path):
        state = json.load(open(state_path))
        last = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        best_val_loss = state["best_val_loss"]
        best_epoch = state["best_epoch"]
        epochs_no_improve = state["epochs_no_improve"]
        start_epoch = state["last_epoch"] + 1
        history = json.load(open(log_path))["history"] if os.path.exists(log_path) else []
        print(f"Resumed from epoch {state['last_epoch']}: best_val_loss={best_val_loss:.5f} "
              f"(epoch {best_epoch}). Continuing from epoch {start_epoch}.")

    run_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.time_budget_seconds and (time.time() - run_start) > args.time_budget_seconds:
            print(f"Time budget ({args.time_budget_seconds}s) reached before epoch {epoch}; stopping this chunk.")
            break
        model.train()
        t0 = time.time()
        running_loss, n_seen = 0.0, 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        train_loss = running_loss / max(n_seen, 1)

        model.eval()
        val_running, val_n = 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon = model(x)
                loss = criterion(recon, x)
                val_running += loss.item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_running / max(val_n, 1)
        scheduler.step(val_loss)
        dt = time.time() - t0

        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": dt})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  -> new best (val_loss={best_val_loss:.5f}), saved to {ckpt_path}")
        else:
            epochs_no_improve += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, last_ckpt_path)
        with open(state_path, "w") as f:
            json.dump({
                "last_epoch": epoch, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
            }, f)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            with open(log_path, "w") as f:
                json.dump({
                    "history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                    "n_params": n_params, "device": str(device),
                    "done": True, "stop_reason": "early_stopping",
                }, f, indent=2)
            break
    else:
        finished_all_epochs = history and history[-1]["epoch"] >= args.epochs
        with open(log_path, "w") as f:
            json.dump({
                "history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                "n_params": n_params, "device": str(device),
                "done": bool(finished_all_epochs),
                "stop_reason": "max_epochs" if finished_all_epochs else "time_budget",
            }, f, indent=2)

    print(f"Best checkpoint so far: epoch {best_epoch}, val_loss={best_val_loss:.5f} -> {ckpt_path}")
    print(f"Trainer state saved to {state_path} (use --resume to continue)")


if __name__ == "__main__":
    main()
