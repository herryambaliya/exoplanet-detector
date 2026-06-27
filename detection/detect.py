"""
detect.py
Member 2 - Signal Processing

Purpose:
    Runs the BLS (Box Least Squares) algorithm on every cleaned light
    curve to find stars that have a repeating brightness dip. Outputs
    a list of "candidates" -- stars worth investigating further. This
    is the filter that reduces dozens/hundreds of stars down to a
    handful of candidates worth classifying.

    BLS is NOT something we implement from scratch -- it's a standard,
    already-built algorithm in astropy. Our job is to call it correctly,
    tune the sensitivity threshold, and package the results in the
    format M3's CNN needs downstream.

Reads from:
    data/processed/TIC_{id}.npy  -- M1's output, see INTERFACES.md

Output:
    results/candidates.csv with columns:
        TIC_ID, period_days, power, depth, duration_days, t0

Usage:
    python detect.py
    or
    from detection.detect import run_bls, is_candidate, run_bls_all
"""

import os
import csv
import numpy as np
from astropy.timeseries import BoxLeastSquares

PROCESSED_DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CANDIDATES_PATH = os.path.join(RESULTS_DIR, "candidates.csv")

# BLS power threshold -- a starting guess. Stars with a BLS power score
# above this are flagged as worth investigating. This number is NOT
# universal -- it depends on how noisy your specific dataset is. If you
# get far too many or far too few candidates after running on your real
# data, adjust this and re-run. See run_bls_all() docstring for guidance
# on how to tune it.
DEFAULT_POWER_THRESHOLD = 0.0005


def run_bls(time, flux, min_period=0.3, max_period=27.0):
    """
    Runs the BLS algorithm on one star's light curve to search for a
    repeating periodic dip.

    Args:
        time (np.ndarray): time values in days (BTJD), from M1's output
        flux (np.ndarray): normalized flux values, from M1's output
        min_period (float): shortest orbital period to search for, in days.
        max_period (float): longest orbital period to search for, in days.

    Returns:
        dict with keys:
            period (float): best-fit orbital period in days
            power (float): BLS power score
            depth (float): fractional flux drop during transit
            duration (float): transit duration in days
            t0 (float): time of the first transit center (BTJD)
            alt_periods (list): alternative periods to try
    """
    duration_grid = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    duration_grid = [d for d in duration_grid if d < min_period * 0.5]

    model = BoxLeastSquares(time, flux)
    periodogram = model.autopower(
        duration_grid,
        minimum_period=min_period,
        maximum_period=max_period,
    )

    # Take top 3 periods instead of just best
    top3_idx = np.argsort(periodogram.power)[::-1][:3]
    best_idx = top3_idx[0]

    best_p = float(periodogram.period[best_idx])

    # Alternative periods including harmonics
    alt_periods = [
        best_p * 2,
        best_p / 2,
        float(periodogram.period[top3_idx[1]]) if len(top3_idx) > 1 else best_p,
        float(periodogram.period[top3_idx[2]]) if len(top3_idx) > 2 else best_p,
    ]

    return {
        "period"     : best_p,
        "power"      : float(periodogram.power[best_idx]),
        "depth"      : float(periodogram.depth[best_idx]),
        "duration"   : float(periodogram.duration[best_idx]),
        "t0"         : float(periodogram.transit_time[best_idx]),
        "alt_periods": alt_periods,
    }
    
def is_candidate(bls_result, power_threshold=DEFAULT_POWER_THRESHOLD):
    """
    Decides whether a star's BLS result is strong enough to flag as a
    candidate worth classifying.

    Args:
        bls_result (dict): output of run_bls()
        power_threshold (float): minimum BLS power score to qualify.
            See module docstring -- this needs tuning per dataset.

    Returns:
        bool: True if this star should be flagged as a candidate
    """
    return bls_result["power"] >= power_threshold


def load_processed_star(filepath):
    """
    Loads one of M1's processed .npy files in the exact format
    documented in INTERFACES.md.

    Args:
        filepath (str): path to a TIC_{id}.npy file

    Returns:
        tuple: (time array, flux array), both float32
    """
    data = np.load(filepath, allow_pickle=True).item()
    return data["time"], data["flux"]


def extract_tic_id(filepath):
    """Pulls the TIC ID out of a filename like TIC_12345.npy"""
    base = os.path.basename(filepath)
    return base.replace("TIC_", "").replace(".npy", "")


def list_processed_files():
    """Returns a sorted list of all processed .npy file paths from M1."""
    if not os.path.exists(PROCESSED_DATA_DIR):
        return []
    files = [
        os.path.join(PROCESSED_DATA_DIR, f)
        for f in os.listdir(PROCESSED_DATA_DIR)
        if f.endswith(".npy")
    ]
    return sorted(files)


def run_bls_all(power_threshold=DEFAULT_POWER_THRESHOLD, verbose=True):
    """
    Loops over every processed star from M1, runs BLS on each, and
    writes any that pass the power threshold to results/candidates.csv.

    HOW TO TUNE power_threshold:
    Run this once with the default value and check how many candidates
    you get out of your total star count.
      - If you get almost ALL stars flagged as candidates (e.g. 20 out
        of 23), the threshold is too low -- BLS power scores from pure
        noise can still look non-zero, so a too-low threshold doesn't
        actually filter anything. Raise it.
      - If you get ZERO candidates, the threshold is too high, or your
        dataset genuinely has no detectable signals (possible with a
        small, unlabeled sample -- not every star has a transiting
        planet). Try lowering it first before assuming there's nothing
        to find.
      - A reasonable approach: run once, print out the full distribution
        of power scores across all stars (not just pass/fail), and set
        the threshold somewhere that separates a small top group from
        the rest -- this project's small dataset (23 stars) means you
        should expect to manually sanity-check whatever count comes out
        rather than trusting the default blindly.

    Returns:
        dict with keys: 'total' (int), 'candidates' (int)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    files = list_processed_files()

    if not files:
        print("No processed files found in data/processed/. "
              "Make sure M1's preprocess.py has been run first.")
        return {"total": 0, "candidates": 0}

    if verbose:
        print(f"Found {len(files)} processed stars. Running BLS on each...")

    candidates = []
    all_results = []  # keep every result, not just candidates, for tuning/debugging

    for i, filepath in enumerate(files):
        tic_id = extract_tic_id(filepath)
        try:
            time, flux = load_processed_star(filepath)
            bls_result = run_bls(time, flux)
            all_results.append((tic_id, bls_result))

            if is_candidate(bls_result, power_threshold):
                candidates.append((tic_id, bls_result))
                if verbose:
                    print(f"[{i+1}/{len(files)}] TIC {tic_id}: CANDIDATE "
                          f"(power={bls_result['power']:.5f}, "
                          f"period={bls_result['period']:.3f}d)")
            else:
                if verbose:
                    print(f"[{i+1}/{len(files)}] TIC {tic_id}: not significant "
                          f"(power={bls_result['power']:.5f})")

        except Exception as e:
            if verbose:
                print(f"[{i+1}/{len(files)}] TIC {tic_id}: FAILED - {e}")
            continue

    # Save candidates to the shared results file every other module reads
    with open(CANDIDATES_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TIC_ID", "period_days", "power", "depth", "duration_days", "t0", "alt_periods"])
        for tic_id, r in candidates:
            writer.writerow([
                tic_id, r["period"], r["power"],
                r["depth"], r["duration"], r["t0"],
                str(r.get("alt_periods", []))
            ])

    if verbose:
        print(f"\nDone. {len(candidates)} candidates out of {len(files)} stars.")
        print(f"Saved to {CANDIDATES_PATH}")

        # Print the power score distribution to help with threshold tuning
        if all_results:
            powers = sorted([r["power"] for _, r in all_results], reverse=True)
            print(f"\nPower score distribution (for threshold tuning):")
            print(f"  Highest: {powers[0]:.5f}")
            print(f"  Median:  {powers[len(powers)//2]:.5f}")
            print(f"  Lowest:  {powers[-1]:.5f}")
            print(f"  Current threshold: {power_threshold}")

    return {"total": len(files), "candidates": len(candidates)}


if __name__ == "__main__":
    run_bls_all()
