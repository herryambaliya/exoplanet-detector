# INTERFACES.md

This file is the contract every team member codes against. If you need to
change any format described here, message the team first -- changing this
silently breaks other people's code.

## 1. Processed light curve files (Member 1 produces, Members 2 & 3 consume)

Path pattern: `data/processed/TIC_{tic_id}.npy`

Load with:
```python
import numpy as np
data = np.load(path, allow_pickle=True).item()
time = data["time"]   # np.float32 array, shape (N,), units = BTJD days
flux = data["flux"]   # np.float32 array, shape (N,), normalized so median = 1.0
```

Guarantees:
- No NaNs remain in either array.
- flux has had outliers removed (sigma=5) and slow drift removed (flatten),
  with real transit signals protected via a two-pass flatten -- see the
  docstring in `preprocessing/preprocess.py::clean()` for why this matters.
- N varies per star (some have more data gaps than others). Do not assume a
  fixed N for this array -- that only applies after folding (see below).

## 2. Candidate list (Member 2 produces, Members 3 & 4 consume)

Path: `results/candidates.csv`

Columns: `TIC_ID, period_days, power, depth, duration_days, t0`

One row per star that passed the BLS power threshold and is worth
classifying.

## 3. Folded candidate arrays (Member 2 produces, Member 3 consumes)

Path: `results/folded_candidates.npy`

Shape: `(N_candidates, 200)`, dtype float32. This is a FIXED size of 200
points per candidate, regardless of how many raw data points the original
star had. Built by phase-folding on the detected period and resampling to
200 evenly spaced points.

Path: `results/candidate_ids.txt`

One TIC ID per line, in the exact same row order as folded_candidates.npy.
Used to map model predictions back to the correct star.

## 4. Trained model (Member 3 produces, Member 3 & run_pipeline.py consume)

Path: `model/cnn_best.h5`

Input shape expected: `(batch_size, 200, 1)` float32
Output shape: `(batch_size, 4)` float32, softmax probabilities

Class index order (NEVER changes):
- 0 = planet
- 1 = eclipsing_binary
- 2 = blend
- 3 = other

## 5. Predictions (Member 3 produces, Members 2 & 4 consume)

Path: `results/predictions.csv`

Columns: `TIC_ID, planet_prob, binary_prob, blend_prob, other_prob,
predicted_class`

Member 2's `params.py` later appends: `period_days, depth_pct,
duration_hrs, snr` to this same file, for rows where predicted_class ==
"planet".

## Why these contracts matter

If any of these formats change without updating this file and notifying
the team, code that depends on it will fail silently or produce wrong
numbers rather than a clear error -- which is much harder to debug under
hackathon time pressure. Treat this file as the single source of truth.
