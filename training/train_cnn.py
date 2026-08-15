"""
Train the AeroSense CNN classifier on the real manifest produced by
prepare_dataset.py.

Usage:
    python -m training.train_cnn --epochs 15 --batch-size 32

Produces:
    models/aerosense_cnn_v1.pt        (best checkpoint by val macro-F1)
    models/cnn_training_log.json      (per-epoch metrics, for the README)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import CLASS_NAMES, MODEL_DIR, DATA_PROCESSED_DIR, RANDOM_SEED
from training.dataset import AeroSenseSegmentDataset, CLASS_TO_IDX
from backend.app.ml.models.cnn import AeroSenseCNN


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "preds": all_preds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0, help="debug: cap batches/epoch, 0=all")
    parser.add_argument("--resume", action="store_true",
                         help="resume from models/cnn_trainer_state.json + last checkpoint "
                              "(needed because this environment does not keep background "
                              "processes alive between tool calls, so training happens in "
                              "resumable synchronous chunks)")
    parser.add_argument("--time-budget-seconds", type=float, default=0,
                         help="stop after roughly this many wall-clock seconds (0 = no limit, run --epochs)")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    manifest_path = os.path.join(DATA_PROCESSED_DIR, "manifest.csv")
    train_ds = AeroSenseSegmentDataset(manifest_path, "train", augment=True)
    val_ds = AeroSenseSegmentDataset(manifest_path, "val", augment=False)
    print(f"train segments={len(train_ds)}  val segments={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_labels = [CLASS_TO_IDX[r["label"]] for r in train_ds.rows]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=train_labels
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("class distribution (train):", {c: train_labels.count(i) for i, c in enumerate(CLASS_NAMES)})
    print("class weights:", class_weights.round(3).tolist())

    model = AeroSenseCNN(num_classes=len(CLASS_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "aerosense_cnn_v1.pt")
    last_ckpt_path = os.path.join(MODEL_DIR, "aerosense_cnn_last.pt")
    state_path = os.path.join(MODEL_DIR, "cnn_trainer_state.json")
    log_path = os.path.join(MODEL_DIR, "cnn_training_log.json")

    start_epoch = 1
    if args.resume and os.path.exists(state_path) and os.path.exists(last_ckpt_path):
        state = json.load(open(state_path))
        last = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        best_f1 = state["best_f1"]
        best_epoch = state["best_epoch"]
        epochs_no_improve = state["epochs_no_improve"]
        start_epoch = state["last_epoch"] + 1
        history = json.load(open(log_path))["history"] if os.path.exists(log_path) else []
        print(f"Resumed from epoch {state['last_epoch']}: best_f1={best_f1:.4f} (epoch {best_epoch}), "
              f"epochs_no_improve={epochs_no_improve}. Continuing from epoch {start_epoch}.")

    run_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.time_budget_seconds and (time.time() - run_start) > args.time_budget_seconds:
            print(f"Time budget ({args.time_budget_seconds}s) reached before epoch {epoch}; stopping this chunk.")
            break
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for i, (x, y) in enumerate(train_loader):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["macro_f1"])
        dt = time.time() - t0

        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  ({dt:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "seconds": dt,
        })

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": len(CLASS_NAMES),
                "class_names": CLASS_NAMES,
                "val_macro_f1": best_f1,
                "val_accuracy": val_metrics["accuracy"],
                "epoch": epoch,
            }, ckpt_path)
            print(f"  -> new best (val_macro_f1={best_f1:.4f}), saved to {ckpt_path}")
        else:
            epochs_no_improve += 1

        # Always persist "last" state (weights + optimizer + counters) so the
        # NEXT synchronous invocation of this script can resume seamlessly.
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, last_ckpt_path)
        with open(state_path, "w") as f:
            json.dump({
                "last_epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
            }, f)
        with open(log_path, "w") as f:
            json.dump({
                "history": history, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                "n_params": n_params, "device": str(device), "class_names": CLASS_NAMES,
                "done": False,
            }, f, indent=2)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            with open(log_path, "w") as f:
                json.dump({
                    "history": history, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                    "n_params": n_params, "device": str(device), "class_names": CLASS_NAMES,
                    "done": True, "stop_reason": "early_stopping",
                }, f, indent=2)
            break
    else:
        # loop completed without hitting patience-based early stop:
        # mark done only if we actually reached the final requested epoch
        finished_all_epochs = history and history[-1]["epoch"] >= args.epochs
        with open(log_path, "w") as f:
            json.dump({
                "history": history, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                "n_params": n_params, "device": str(device), "class_names": CLASS_NAMES,
                "done": bool(finished_all_epochs),
                "stop_reason": "max_epochs" if finished_all_epochs else "time_budget",
            }, f, indent=2)

    print(f"Best checkpoint so far: epoch {best_epoch}, val_macro_f1={best_f1:.4f} -> {ckpt_path}")
    print(f"Trainer state saved to {state_path} (use --resume to continue)")


if __name__ == "__main__":
    main()
