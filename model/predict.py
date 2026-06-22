"""
predict.py
----------
Member 3 (ML Engineer) — model/ folder

Loads trained CNN (model/cnn_best.h5) and runs it on M2's candidates.
Also runs parameter estimation on planet candidates.

Reads  : results/folded_candidates.npy     (from M2)
Reads  : model/cnn_best.h5                 (from train.py)
Writes : results/predictions.csv           (for M4)

predictions.csv columns (INTERFACES.md):
  TIC_ID, planet_prob, binary_prob, blend_prob, other_prob,
  predicted_class, period_days, depth_ppm, duration_hours, snr

Usage (in Jupyter):
  exec(open('model/predict.py', encoding='utf-8').read())
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras

import sys
sys.path.insert(0, '.')

from model.model import CLASS_NAMES, INPUT_LENGTH
from model.parameter_estimation import estimate_transit_parameters

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_PATH      = Path("model/cnn_best.h5")
CANDIDATES_PATH = Path("results/folded_candidates.npy")
OUTPUT_PATH     = Path("results/predictions.csv")

# ── Preprocessing (same as dataloader) ───────────────────────────────
def preprocess_flux(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float32).flatten()
    n    = len(flux)

    # Normalise by out-of-transit median
    oot    = slice(n // 3, 2 * n // 3)
    median = np.median(flux[oot])
    if median > 0:
        flux = flux / median
    flux = np.clip(flux, 0.5, 1.5)

    # Pad or truncate to exactly 200
    if len(flux) < INPUT_LENGTH:
        flux = np.pad(flux, (0, INPUT_LENGTH - len(flux)), constant_values=1.0)
    else:
        flux = flux[:INPUT_LENGTH]

    return flux.reshape(INPUT_LENGTH, 1)   # (200, 1)


def run_predictions():

    # ── 1. Load model ─────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"[ERROR] No trained model found at {MODEL_PATH}")
        print("  → Run train.py first!")
        return None

    print(f"Loading model from {MODEL_PATH} ...")
    model = keras.models.load_model(str(MODEL_PATH))
    print("Model loaded.")

    # ── 2. Load M2's candidates ───────────────────────────────────────
    if not CANDIDATES_PATH.exists():
        print(f"[ERROR] No candidates file at {CANDIDATES_PATH}")
        print("  → M2 needs to run fold.py first!")
        return None

    print(f"Loading candidates from {CANDIDATES_PATH} ...")
    data    = np.load(str(CANDIDATES_PATH), allow_pickle=True).item()
    tic_ids = np.asarray(data["tic_ids"])
    fluxes  = np.asarray(data["fluxes"],  dtype=np.float32)   # (N, 200)
    periods = np.asarray(data["periods"], dtype=np.float32)   # (N,)
    phases  = data.get("phases", None)                         # (N, 200) optional
    print(f"Found {len(tic_ids)} candidates.")

    # ── 3. Preprocess ─────────────────────────────────────────────────
    print("Preprocessing light curves ...")
    X = np.stack([preprocess_flux(f) for f in fluxes])        # (N, 200, 1)

    # ── 4. Predict ────────────────────────────────────────────────────
    print("Running CNN predictions ...")
    probs = model.predict(X, batch_size=128, verbose=1)        # (N, 4)

    planet_prob  = probs[:, 0]
    binary_prob  = probs[:, 1]
    blend_prob   = probs[:, 2]
    other_prob   = probs[:, 3]
    predicted_idx = np.argmax(probs, axis=1)
    predicted_cls = [CLASS_NAMES[i] for i in predicted_idx]

    # ── 5. Parameter estimation for planet candidates ─────────────────
    print("Running parameter estimation on planet candidates ...")

    depth_ppm_list     = []
    duration_hours_list = []
    snr_list           = []

    for i in range(len(tic_ids)):
        # Only run full estimation on likely planets (prob > 0.3)
        if planet_prob[i] > 0.3 and phases is not None:
            try:
                phase = np.asarray(phases[i], dtype=np.float32)
                flux  = np.asarray(fluxes[i], dtype=np.float32)

                # Centre phase around 0
                phase = phase - np.median(phase)

                params = estimate_transit_parameters(
                    phase       = phase,
                    flux        = flux,
                    flux_err    = None,
                    period_days = float(periods[i]),
                )
                depth_ppm_list.append(round(params.depth_ppm, 1))
                duration_hours_list.append(round(params.duration_hours, 3))
                snr_list.append(round(params.snr, 2))

            except Exception as e:
                depth_ppm_list.append(None)
                duration_hours_list.append(None)
                snr_list.append(None)
        else:
            depth_ppm_list.append(None)
            duration_hours_list.append(None)
            snr_list.append(None)

    # ── 6. Build results dataframe ────────────────────────────────────
    results = pd.DataFrame({
        "TIC_ID"         : tic_ids,
        "planet_prob"    : np.round(planet_prob,  4),
        "binary_prob"    : np.round(binary_prob,  4),
        "blend_prob"     : np.round(blend_prob,   4),
        "other_prob"     : np.round(other_prob,   4),
        "predicted_class": predicted_cls,
        "period_days"    : np.round(periods, 6),
        "depth_ppm"      : depth_ppm_list,
        "duration_hours" : duration_hours_list,
        "snr"            : snr_list,
    })

    # Sort — planet candidates first, then by confidence
    results = results.sort_values(
        ["predicted_class", "planet_prob"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # ── 7. Save ───────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(str(OUTPUT_PATH), index=False)
    print(f"\n✓ Saved predictions → {OUTPUT_PATH}")

    # ── 8. Print summary ──────────────────────────────────────────────
    print("\n══════  Prediction Summary  ══════")
    print(results["predicted_class"].value_counts().to_string())
    print(f"\nTop planet candidates:")
    planets = results[results["predicted_class"] == "planet"].head(10)
    if len(planets) > 0:
        print(planets[["TIC_ID","planet_prob","period_days",
                        "depth_ppm","duration_hours","snr"]].to_string(index=False))
    else:
        print("  No planet candidates found.")
    print("══════════════════════════════════")

    return results


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_predictions()
