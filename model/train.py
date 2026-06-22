"""
train.py
--------
Member 3 (ML Engineer) — model/ folder

Trains the TransitCNN on the curated labelled dataset.
Saves the best checkpoint to  model/cnn_best.h5

Usage
-----
  python model/train.py                              # defaults
  python model/train.py --data  data/processed/train
  python model/train.py --epochs 60 --batch 32

Output files
------------
  model/cnn_best.h5           best validation-accuracy checkpoint
  results/training_history.npy  loss + accuracy curves (for M4 plots)
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from model      import build_transit_cnn, CLASS_NAMES
from dataloader import load_training_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Defaults  (override via CLI args)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    data_root   = "data/processed/train",
    model_out   = "model/cnn_best.h5",
    history_out = "results/training_history.npy",
    epochs      = 50,
    batch_size  = 64,
    lr          = 3e-4,
    val_split   = 0.15,
    patience    = 10,           # early stopping patience
    seed        = 42,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> keras.callbacks.History:
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    train_ds, val_ds, info = load_training_dataset(
        data_root  = cfg["data_root"],
        batch_size = cfg["batch_size"],
        val_split  = cfg["val_split"],
        augment    = True,
        seed       = cfg["seed"],
    )
    class_weights = info["class_weights"]

    # ── 2. Model ──────────────────────────────────────────────────────────
    print("\n[2/4] Building model …")
    model = build_transit_cnn()
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary(line_length=80)

    # ── 3. Callbacks ──────────────────────────────────────────────────────
    Path(cfg["model_out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["history_out"]).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save best model by val_accuracy
        keras.callbacks.ModelCheckpoint(
            filepath          = cfg["model_out"],
            monitor           = "val_accuracy",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),
        # Stop early if val_accuracy stagnates
        keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = cfg["patience"],
            restore_best_weights = True,
            verbose              = 1,
        ),
        # Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 5,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        # Training progress logging
        keras.callbacks.CSVLogger("results/training_log.csv"),
    ]

    # ── 4. Train ──────────────────────────────────────────────────────────
    print(f"\n[3/4] Training for up to {cfg['epochs']} epochs …")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["epochs"],
        callbacks       = callbacks,
        class_weight    = class_weights,
        verbose         = 1,
    )

    # ── 5. Save history for M4 visualisation ─────────────────────────────
    print(f"\n[4/4] Saving training history → {cfg['history_out']}")
    np.save(cfg["history_out"], history.history)

    # ── 6. Final evaluation report ────────────────────────────────────────
    _print_final_report(model, val_ds, history)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_final_report(model, val_ds, history):
    """Print per-class accuracy on the validation set."""
    print("\n" + "="*55)
    print("  Final Validation Report")
    print("="*55)

    best_val_acc = max(history.history["val_accuracy"])
    best_epoch   = np.argmax(history.history["val_accuracy"]) + 1
    print(f"  Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")

    # Per-class confusion
    all_preds, all_labels = [], []
    for X_batch, y_batch in val_ds:
        probs = model.predict(X_batch, verbose=0)
        all_preds.extend(np.argmax(probs, axis=1))
        all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n  Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if mask.sum() == 0:
            print(f"    {name:22s}: no samples")
            continue
        acc = (all_preds[mask] == i).mean()
        print(f"    {name:22s}: {acc:.3f}  ({mask.sum()} samples)")

    print("="*55 + "\n")


def evaluate_on_test(model_path: str, test_ds: tf.data.Dataset):
    """
    Utility for post-hoc evaluation on a held-out test set.
    Called by M4's validation.py if needed.
    """
    model = keras.models.load_model(model_path)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.4f}")
    return loss, acc


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> dict:
    p = argparse.ArgumentParser(description="Train TransitCNN")
    p.add_argument("--data",    default=DEFAULTS["data_root"],   help="Path to labelled training data root")
    p.add_argument("--out",     default=DEFAULTS["model_out"],   help="Output path for best .h5 model")
    p.add_argument("--epochs",  type=int,   default=DEFAULTS["epochs"],     help="Max training epochs")
    p.add_argument("--batch",   type=int,   default=DEFAULTS["batch_size"], help="Batch size")
    p.add_argument("--lr",      type=float, default=DEFAULTS["lr"],         help="Initial learning rate")
    p.add_argument("--patience",type=int,   default=DEFAULTS["patience"],   help="Early stopping patience")
    p.add_argument("--seed",    type=int,   default=DEFAULTS["seed"],       help="Random seed")
    args = p.parse_args()
    return dict(
        data_root   = args.data,
        model_out   = args.out,
        history_out = DEFAULTS["history_out"],
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
        val_split   = DEFAULTS["val_split"],
        patience    = args.patience,
        seed        = args.seed,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    print("Training config:")
    for k, v in cfg.items():
        print(f"  {k:15s}: {v}")
    train(cfg)
