"""
model.py
--------
Member 3 (ML Engineer) — model/ folder

CNN architecture for classifying phase-folded light curves.

Input  : (batch_size, 200, 1)  float32   ← INTERFACES.md spec
Output : (batch_size, 4)       float32   softmax probabilities

Class label order  (NEVER change — INTERFACES.md):
    0 = planet
    1 = eclipsing_binary
    2 = blend
    3 = other

Saved weights → model/cnn_best.h5
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ── Label map  (shared contract — DO NOT reorder) ─────────────────────────────
CLASS_NAMES  = ["planet", "eclipsing_binary", "blend", "other"]
NUM_CLASSES  = len(CLASS_NAMES)
INPUT_LENGTH = 200     # folded array size  (INTERFACES.md)


# ─────────────────────────────────────────────────────────────────────────────
# Building block : Residual 1-D Conv block
# ─────────────────────────────────────────────────────────────────────────────

def _res_block(x, filters: int, kernel: int, dilation: int = 1):
    """
    Residual block with two dilated 1-D convolutions + skip connection.
    Dilation lets the network see multi-scale transit shapes without
    losing temporal resolution.
    """
    shortcut = x

    x = layers.Conv1D(
        filters, kernel,
        padding       = "same",
        dilation_rate = dilation,
        kernel_regularizer = regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(
        filters, kernel,
        padding       = "same",
        dilation_rate = dilation,
        kernel_regularizer = regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)

    # Match channel dimension if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Main model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_transit_cnn(
    input_length : int   = INPUT_LENGTH,
    num_classes  : int   = NUM_CLASSES,
    dropout_rate : float = 0.4,
) -> keras.Model:
    """
    Build and return the transit classification CNN.

    Architecture overview
    ---------------------
    Input (200, 1)
      -> Stem Conv  (captures broad transit shape)
      -> Res block x3 with increasing dilation  (multi-scale features)
      -> Global Average Pool  (removes sequence length dependency)
      -> Dense 128 -> Dropout -> Dense 64 -> Dropout
      -> Softmax output (4 classes)

    Why 1-D CNN instead of LSTM?
      Phase-folded light curves are fixed-length, spatially regular
      signals — CNNs are faster to train and less prone to vanishing
      gradients on 200-point sequences than RNNs.

    Parameters
    ----------
    input_length : Length of folded array  (default 200).
    num_classes  : Number of output classes  (default 4).
    dropout_rate : Dropout probability in the dense head.

    Returns
    -------
    keras.Model  (uncompiled)
    """
    inp = keras.Input(shape=(input_length, 1), name="folded_flux")

    # Stem : broad feature extraction
    x = layers.Conv1D(32, 7, padding="same", name="stem_conv")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)                   # 200 -> 100

    # Stage 1 : local transit shape  (dilation=1)
    x = _res_block(x, filters=64,  kernel=5, dilation=1)
    x = layers.MaxPooling1D(2)(x)                   # 100 -> 50

    # Stage 2 : mid-scale features  (dilation=2)
    x = _res_block(x, filters=128, kernel=5, dilation=2)
    x = layers.MaxPooling1D(2)(x)                   # 50  -> 25

    # Stage 3 : long-range context  (dilation=4)
    x = _res_block(x, filters=128, kernel=3, dilation=4)

    # Global pooling : collapses temporal dimension
    x = layers.GlobalAveragePooling1D()(x)

    # Dense classification head
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_128",
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_64",
    )(x)
    x = layers.Dropout(dropout_rate / 2)(x)

    out = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs=inp, outputs=out, name="TransitCNN")
    return model


def load_model(path: str = "model/cnn_best.h5") -> keras.Model:
    """Load a previously saved model from disk."""
    return keras.models.load_model(path)


# Quick sanity check
if __name__ == "__main__":
    model = build_transit_cnn()
    model.summary()

    import numpy as np
    dummy = np.random.rand(8, 200, 1).astype("float32")
    probs = model.predict(dummy, verbose=0)
    print(f"\nOutput shape : {probs.shape}   (expected (8, 4))")
    print(f"Row sums     : {probs.sum(axis=1).round(4)}   (should all be 1.0)")
    print(f"Class names  : {CLASS_NAMES}")
