"""
dataloader.py
-------------
Member 3 (ML Engineer) — model/ folder

Loads M2's folded_candidates.npy and the curated training dataset,
returns tf.data.Dataset objects ready for the CNN.

INTERFACES.md contract:
  - folded array size : exactly (200,) float32
  - values            : between 0.8 and 1.1 approximately
  - CNN input shape   : (batch_size, 200, 1)  float32
  - class label order : 0=planet  1=eclipsing_binary  2=blend  3=other

Expected training data layout  (curated dataset from problem statement):
  data/
    processed/
      train/
        planet/            *.npy  — (200,) arrays
        eclipsing_binary/  *.npy
        blend/             *.npy
        other/             *.npy
      val/
        <same structure>

For inference  (run_pipeline.py):
  results/folded_candidates.npy  — dict with keys:
    "tic_ids"  : (N,)      int
    "fluxes"   : (N, 200)  float32
    "periods"  : (N,)      float32   (from BLS)
    "phases"   : (N, 200)  float32   (optional, for param estimation)
"""

from __future__ import annotations

import os
import glob
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, List

from model import CLASS_NAMES, INPUT_LENGTH   # shared constants

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _augment(flux: tf.Tensor) -> tf.Tensor:
    """
    Light augmentation applied only during training.

    Operations  (all transit-safe — they don't change class identity):
      • Random phase roll  — shifts the transit position slightly
      • Gaussian noise jitter  — simulates different noise levels
      • Random flux scaling  — small multiplicative factor
    """
    # Phase roll  (circular shift by random amount)
    shift = tf.random.uniform((), minval=0, maxval=INPUT_LENGTH, dtype=tf.int32)
    flux  = tf.roll(flux, shift=shift, axis=0)

    # Additive Gaussian noise
    flux = flux + tf.random.normal(tf.shape(flux), mean=0.0, stddev=5e-4)

    # Multiplicative scaling  (±1 % — keeps values in 0.8–1.1 range)
    scale = tf.random.uniform((), minval=0.99, maxval=1.01)
    flux  = flux * scale

    return flux


def _preprocess(flux: np.ndarray) -> np.ndarray:
    """
    Ensure a single folded array is correctly shaped and normalised.
    Input  : (200,)  or  (200, 1)
    Output : (200, 1)  float32,  median-normalised
    """
    flux = np.asarray(flux, dtype=np.float32).flatten()

    # Robust normalisation : divide by median of out-of-transit region
    # (middle 40 % of phase is most likely OOT for any period)
    n = len(flux)
    oot_slice = slice(n // 3, 2 * n // 3)
    median = np.median(flux[oot_slice])
    if median > 0:
        flux = flux / median
    flux = np.clip(flux, 0.5, 1.5)            # safety clip

    # Pad or truncate to exactly INPUT_LENGTH
    if len(flux) < INPUT_LENGTH:
        flux = np.pad(flux, (0, INPUT_LENGTH - len(flux)), constant_values=1.0)
    else:
        flux = flux[:INPUT_LENGTH]

    return flux.reshape(INPUT_LENGTH, 1)       # (200, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training dataset  (from curated labelled directories)
# ─────────────────────────────────────────────────────────────────────────────

def load_training_dataset(
    data_root  : str,
    batch_size : int   = 64,
    val_split  : float = 0.15,
    augment    : bool  = True,
    seed       : int   = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Load labelled .npy files from data_root and return train + val datasets.

    Parameters
    ----------
    data_root  : Path to directory containing class subdirectories.
                 e.g. "data/processed/train/"
                 Structure:
                   train/planet/*.npy
                   train/eclipsing_binary/*.npy
                   train/blend/*.npy
                   train/other/*.npy
    batch_size : Mini-batch size.
    val_split  : Fraction of data reserved for validation.
    augment    : Apply augmentation to training set.
    seed       : Random seed for reproducibility.

    Returns
    -------
    train_ds  : tf.data.Dataset
    val_ds    : tf.data.Dataset
    info      : dict with class counts and label mapping
    """
    all_fluxes, all_labels = [], []
    class_counts = {}

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(data_root) / class_name
        if not class_dir.exists():
            print(f"  [WARN] Missing class directory: {class_dir}")
            class_counts[class_name] = 0
            continue

        files = sorted(glob.glob(str(class_dir / "*.npy")))
        class_counts[class_name] = len(files)

        for fp in files:
            try:
                raw = np.load(fp, allow_pickle=True)
                # Handle dict format from M2's fold.py
                if isinstance(raw, dict):
                    flux = raw.get("flux", raw.get("fluxes"))
                else:
                    flux = np.asarray(raw)
                all_fluxes.append(_preprocess(flux))
                all_labels.append(label_idx)
            except Exception as e:
                print(f"  [WARN] Skipping {fp}: {e}")

    if len(all_fluxes) == 0:
        raise ValueError(f"No .npy files found under {data_root}. Check your path.")

    X = np.stack(all_fluxes).astype(np.float32)   # (N, 200, 1)
    y = np.array(all_labels, dtype=np.int32)

    # Shuffle
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train / val split
    split = int(len(X) * (1 - val_split))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # Class weights  (handle imbalanced datasets)
    class_weights = _compute_class_weights(y_tr)

    # Build tf.data pipelines
    AUTOTUNE = tf.data.AUTOTUNE

    def make_ds(X_arr, y_arr, training=False):
        ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
        if training:
            ds = ds.shuffle(buffer_size=2048, seed=seed)
            if augment:
                ds = ds.map(
                    lambda x, y: (_augment(x), y),
                    num_parallel_calls=AUTOTUNE,
                )
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(X_tr, y_tr, training=True)
    val_ds   = make_ds(X_val, y_val, training=False)

    info = {
        "class_counts"  : class_counts,
        "class_weights" : class_weights,
        "n_train"       : len(X_tr),
        "n_val"         : len(X_val),
        "label_map"     : {i: n for i, n in enumerate(CLASS_NAMES)},
    }

    print(f"Dataset loaded: {info['n_train']} train / {info['n_val']} val")
    for cn, cc in class_counts.items():
        print(f"  {cn:20s}: {cc} samples")

    return train_ds, val_ds, info


def _compute_class_weights(labels: np.ndarray) -> dict:
    """Inverse-frequency class weights to handle class imbalance."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    counts = np.where(counts == 0, 1, counts)          # avoid div by zero
    total  = counts.sum()
    weights = total / (len(CLASS_NAMES) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Inference dataset  (M2's folded_candidates.npy)
# ─────────────────────────────────────────────────────────────────────────────

def load_inference_batch(
    npy_path   : str,
    batch_size : int = 128,
) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load M2's folded_candidates.npy for prediction.

    Parameters
    ----------
    npy_path   : Path to results/folded_candidates.npy
    batch_size : Batch size for prediction.

    Returns
    -------
    ds       : tf.data.Dataset  of (200, 1) float32 tensors
    tic_ids  : np.ndarray  shape (N,)
    periods  : np.ndarray  shape (N,)  in days
    phases   : np.ndarray  shape (N, 200)  or None
    """
    data = np.load(npy_path, allow_pickle=True).item()

    tic_ids = np.asarray(data["tic_ids"])
    fluxes  = np.asarray(data["fluxes"],  dtype=np.float32)   # (N, 200)
    periods = np.asarray(data["periods"], dtype=np.float32)   # (N,)
    phases  = data.get("phases")                               # optional

    # Preprocess each light curve
    X = np.stack([_preprocess(f) for f in fluxes])            # (N, 200, 1)

    ds = (
        tf.data.Dataset.from_tensor_slices(X)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Inference batch loaded: {len(tic_ids)} candidates from {npy_path}")
    return ds, tic_ids, periods, phases


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    print("=== Synthetic dataloader test ===\n")

    # Build a tiny fake dataset on disk
    with tempfile.TemporaryDirectory() as tmpdir:
        for cls in CLASS_NAMES:
            cls_dir = Path(tmpdir) / cls
            cls_dir.mkdir()
            for i in range(30):
                flux = np.ones(200, dtype=np.float32)
                flux[90:110] -= np.random.uniform(0.005, 0.02)   # fake dip
                np.save(cls_dir / f"lc_{i:03d}.npy", flux)

        train_ds, val_ds, info = load_training_dataset(
            tmpdir, batch_size=16, val_split=0.2
        )

        for X_batch, y_batch in train_ds.take(1):
            print(f"Batch shape : {X_batch.shape}   (expected (16, 200, 1))")
            print(f"Labels      : {y_batch.numpy()[:8]}")

        print(f"\nClass weights: {info['class_weights']}")
