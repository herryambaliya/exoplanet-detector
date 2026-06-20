"""
fold.py
Member 2 - Signal Processing

Purpose:
    Takes a candidate star (one that passed detect.py's BLS threshold)
    and "folds" its light curve on the detected period -- meaning all
    the individual transit dips, which occur at different times
    throughout the observation, get stacked on top of each other into
    one single, clean, averaged dip shape. This compressed shape is
    then resampled to a FIXED size of 200 points, since M3's CNN
    expects every input to be exactly the same length.

    Why folding works: if a planet has a 4-day orbital period and you
    observed for 27 days, you saw roughly 6-7 individual transits. Each
    one alone is noisy and hard to see clearly. By "folding" -- taking
    the time modulo the period, so every transit lands at the same
    phase position -- and overlaying all of them, the real signal
    reinforces itself while random noise averages out. This is what
    makes a faint, noisy dip become visually obvious.

Reads from:
    data/processed/TIC_{id}.npy -- M1's output (full light curve)
    results/candidates.csv -- M2's own detect.py output (period, t0)

Output:
    results/folded_candidates.npy -- shape (N_candidates, 200), float32
    results/candidate_ids.txt -- TIC IDs in the same row order

Usage:
    python fold.py
    or
    from detection.fold import phase_fold, resample_to_200, fold_all_candidates
"""

import os
import csv
import numpy as np

PROCESSED_DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CANDIDATES_PATH = os.path.join(RESULTS_DIR, "candidates.csv")
FOLDED_PATH = os.path.join(RESULTS_DIR, "folded_candidates.npy")
CANDIDATE_IDS_PATH = os.path.join(RESULTS_DIR, "candidate_ids.txt")

# Fixed output size -- agreed in INTERFACES.md. M3's CNN input layer is
# built expecting exactly this many points. Do not change without
# updating INTERFACES.md and notifying M3.
FOLDED_LENGTH = 200


def phase_fold(time, flux, period, t0):
    """
    Folds a light curve on a given period, centering the transit at
    phase 0.

    Mechanically: for each time value, compute how far it is from the
    nearest predicted transit center (t0, t0+period, t0+2*period, ...),
    expressed as a fraction of the period, ranging from -0.5 to +0.5.
    A point exactly at a transit center gets phase 0. A point exactly
    halfway between two transits gets phase -0.5 or +0.5 (same thing,
    wraps around).

    Args:
        time (np.ndarray): time values in days (BTJD)
        flux (np.ndarray): normalized flux values
        period (float): orbital period in days, from detect.py's BLS result
        t0 (float): time of a transit center, from detect.py's BLS result

    Returns:
        tuple: (phase, flux_sorted)
            phase (np.ndarray): values in [-0.5, 0.5], sorted ascending
            flux_sorted (np.ndarray): flux values reordered to match
                the sorted phase array
    """
    # Center on t0, wrap into [-0.5, 0.5) of the period
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5

    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def resample_to_200(phase, flux, length=FOLDED_LENGTH):
    """
    Resamples a phase-folded light curve to exactly `length` evenly
    spaced points, regardless of how many raw points it started with.

    Why this is necessary: different stars have different amounts of
    data (some have more gaps than others), and BLS-detected periods
    vary, so the number of points landing in any given phase-folded
    curve differs star to star. M3's CNN needs every input to be
    exactly the same fixed length. We use linear interpolation
    (np.interp) to resample onto a uniform phase grid.

    Args:
        phase (np.ndarray): sorted phase values in [-0.5, 0.5], any length
        flux (np.ndarray): corresponding flux values, same length as phase
        length (int): desired output length, default 200 per
            INTERFACES.md

    Returns:
        np.ndarray: shape (length,), dtype float32, evenly spaced
            across phase [-0.5, 0.5]
    """
    if len(phase) < 2:
        raise ValueError(
            f"Cannot resample a light curve with fewer than 2 points "
            f"(got {len(phase)}). This candidate's data may be too sparse."
        )

    # Build a uniform target grid across the full phase range
    target_phase = np.linspace(-0.5, 0.5, length)

    # np.interp requires the input x-values (phase) to be strictly
    # increasing -- duplicate phase values (possible with dense data)
    # are handled by np.interp automatically taking the first match,
    # which is fine for this purpose since we're downsampling, not
    # needing exact uniqueness
    resampled_flux = np.interp(target_phase, phase, flux)

    return resampled_flux.astype(np.float32)


def load_processed_star(filepath):
    """Loads one of M1's processed .npy files. Same format as detect.py uses."""
    data = np.load(filepath, allow_pickle=True).item()
    return data["time"], data["flux"]


def load_candidates():
    """
    Reads results/candidates.csv (M2's own detect.py output).

    Returns:
        list of dict, each with keys: TIC_ID, period_days, power, depth,
            duration_days, t0 (all read as strings except TIC_ID stays
            a string, the rest get cast to float)
    """
    if not os.path.exists(CANDIDATES_PATH):
        return []

    candidates = []
    with open(CANDIDATES_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append({
                "TIC_ID": row["TIC_ID"],
                "period": float(row["period_days"]),
                "power": float(row["power"]),
                "depth": float(row["depth"]),
                "duration": float(row["duration_days"]),
                "t0": float(row["t0"]),
            })
    return candidates


def fold_one_candidate(tic_id, period, t0):
    """
    Full pipeline for a single candidate: load its processed light
    curve, fold it on the detected period, resample to 200 points.

    Args:
        tic_id (str): TIC ID, used to find the processed .npy file
        period (float): orbital period in days
        t0 (float): transit center time

    Returns:
        np.ndarray: shape (200,), float32
    """
    filepath = os.path.join(PROCESSED_DATA_DIR, f"TIC_{tic_id}.npy")
    time, flux = load_processed_star(filepath)
    phase, flux_sorted = phase_fold(time, flux, period, t0)
    folded_array = resample_to_200(phase, flux_sorted)
    return folded_array


def fold_all_candidates(verbose=True):
    """
    Reads every candidate from results/candidates.csv (M2's detect.py
    output), folds and resamples each one, and saves the results in
    the format M3's CNN expects.

    Returns:
        dict with keys: 'total_candidates' (int), 'successfully_folded' (int)
    """
    candidates = load_candidates()

    if not candidates:
        if verbose:
            print("No candidates found in results/candidates.csv. "
                  "Run detect.py's run_bls_all() first.")
        return {"total_candidates": 0, "successfully_folded": 0}

    if verbose:
        print(f"Folding {len(candidates)} candidates...")

    folded_arrays = []
    successful_ids = []

    for i, c in enumerate(candidates):
        try:
            folded = fold_one_candidate(c["TIC_ID"], c["period"], c["t0"])
            folded_arrays.append(folded)
            successful_ids.append(c["TIC_ID"])
            if verbose:
                print(f"[{i+1}/{len(candidates)}] TIC {c['TIC_ID']}: folded successfully")
        except Exception as e:
            if verbose:
                print(f"[{i+1}/{len(candidates)}] TIC {c['TIC_ID']}: FAILED - {e}")
            continue

    if not folded_arrays:
        if verbose:
            print("No candidates were successfully folded.")
        return {"total_candidates": len(candidates), "successfully_folded": 0}

    # Stack into one array: shape (N_candidates, 200)
    folded_matrix = np.stack(folded_arrays).astype(np.float32)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(FOLDED_PATH, folded_matrix)

    with open(CANDIDATE_IDS_PATH, "w") as f:
        for tic_id in successful_ids:
            f.write(f"{tic_id}\n")

    if verbose:
        print(f"\nDone. {len(successful_ids)}/{len(candidates)} candidates folded successfully.")
        print(f"Saved folded array {folded_matrix.shape} to {FOLDED_PATH}")
        print(f"Saved candidate IDs to {CANDIDATE_IDS_PATH}")

    return {"total_candidates": len(candidates), "successfully_folded": len(successful_ids)}


if __name__ == "__main__":
    fold_all_candidates()
