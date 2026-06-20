"""
params.py
Member 2 - Signal Processing

Purpose:
    After M3's CNN classifies a candidate as "planet", this module
    extracts the final, reportable numbers ISRO's problem statement
    explicitly asks for: orbital period, transit depth, transit
    duration, and a confidence/significance measure (SNR).

    Most of the raw values (period, depth, duration) already come from
    detect.py's BLS output -- this module's job is mainly to: (1) pull
    those values for the right candidates, (2) convert units into the
    human-readable form the report needs (e.g. duration in hours, not
    days), and (3) compute SNR, which BLS does not provide directly.

Reads from:
    results/candidates.csv -- M2's detect.py output (period, depth,
        duration, t0 for ALL candidates)
    results/predictions.csv -- M3's predict.py output (which
        candidates were actually classified as "planet", with what
        confidence)
    data/processed/TIC_{id}.npy -- M1's output (needed to compute SNR,
        which requires the actual noise level in the light curve)

Output:
    Updates results/predictions.csv in place, adding columns:
        period_days, depth_pct, duration_hrs, snr

Usage:
    python params.py
    or
    from detection.params import estimate_params, estimate_all_planets
"""

import os
import csv
import numpy as np

PROCESSED_DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CANDIDATES_PATH = os.path.join(RESULTS_DIR, "candidates.csv")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.csv")


def load_candidates_lookup():
    """
    Loads results/candidates.csv into a dict keyed by TIC_ID for fast
    lookup, since params.py needs to cross-reference BLS results
    (from detect.py) against classification results (from M3).

    Returns:
        dict: {TIC_ID: {period, power, depth, duration, t0}}
    """
    lookup = {}
    if not os.path.exists(CANDIDATES_PATH):
        return lookup

    with open(CANDIDATES_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["TIC_ID"]] = {
                "period": float(row["period_days"]),
                "power": float(row["power"]),
                "depth": float(row["depth"]),
                "duration": float(row["duration_days"]),
                "t0": float(row["t0"]),
            }
    return lookup


def load_processed_star(filepath):
    """Loads one of M1's processed .npy files. Same format used throughout."""
    data = np.load(filepath, allow_pickle=True).item()
    return data["time"], data["flux"]


def compute_snr(tic_id, period, t0, duration, depth):
    """
    Computes a signal-to-noise ratio for a transit detection: how many
    times larger the transit depth is compared to the typical
    out-of-transit scatter (noise) in the light curve.

    A higher SNR means a more statistically convincing detection --
    this is the "confidence level" ISRO's problem statement asks for
    alongside the transit parameters.

    Method: load the star's full cleaned light curve, identify which
    points fall OUTSIDE any transit window (using the known period,
    t0, duration), measure the standard deviation of flux among just
    those out-of-transit points (this is the noise level), then
    SNR = depth / noise_std.

    Args:
        tic_id (str): TIC ID, used to load the processed light curve
        period (float): orbital period in days
        t0 (float): transit center time
        duration (float): transit duration in days
        depth (float): transit depth as a fraction (e.g. 0.01 = 1%)

    Returns:
        float: SNR value. Higher is more significant. As a rough
            guide, SNR > 7 is often used as a minimum threshold for a
            credible detection in real transit-search literature,
            though this varies by survey and should be treated as a
            general guideline, not a strict cutoff.
    """
    filepath = os.path.join(PROCESSED_DATA_DIR, f"TIC_{tic_id}.npy")
    time, flux = load_processed_star(filepath)

    # Distance from the nearest transit center, wrapped by period
    dist_from_transit = np.abs(((time - t0 + period / 2) % period) - period / 2)

    # Use a slightly wider window than the transit duration itself to
    # safely exclude any partial-transit points from the noise
    # estimate -- using exactly the transit duration risks including
    # ingress/egress points that are partially in-transit, which would
    # underestimate the true out-of-transit noise level
    out_of_transit_mask = dist_from_transit > (duration * 0.75)

    if out_of_transit_mask.sum() < 10:
        # Not enough out-of-transit points to get a reliable noise
        # estimate -- this can happen for very long-period candidates
        # where most of the light curve is technically "near" a
        # transit prediction due to limited baseline
        raise ValueError(
            f"Only {out_of_transit_mask.sum()} out-of-transit points "
            f"available for TIC {tic_id} -- too few for a reliable SNR estimate."
        )

    noise_std = np.std(flux[out_of_transit_mask])

    if noise_std == 0:
        raise ValueError(f"Zero noise standard deviation for TIC {tic_id} -- "
                          f"cannot compute a meaningful SNR.")

    snr = depth / noise_std
    return float(snr)


def estimate_params(tic_id, candidates_lookup):
    """
    Computes the final, reportable parameter set for one candidate.

    Args:
        tic_id (str): TIC ID of the candidate
        candidates_lookup (dict): output of load_candidates_lookup()

    Returns:
        dict with keys: period_days, depth_pct, duration_hrs, snr
    """
    if tic_id not in candidates_lookup:
        raise KeyError(
            f"TIC {tic_id} not found in candidates.csv. This candidate "
            f"may not have come from detect.py's BLS search -- params.py "
            f"can only compute parameters for BLS-detected candidates."
        )

    c = candidates_lookup[tic_id]

    period_days = c["period"]
    depth_pct = c["depth"] * 100  # convert fraction to percentage
    duration_hrs = c["duration"] * 24  # convert days to hours

    snr = compute_snr(tic_id, c["period"], c["t0"], c["duration"], c["depth"])

    return {
        "period_days": round(period_days, 4),
        "depth_pct": round(depth_pct, 4),
        "duration_hrs": round(duration_hrs, 3),
        "snr": round(snr, 2),
    }


def load_predictions():
    """
    Reads results/predictions.csv (M3's output). Returns the rows as a
    list of dicts, preserving all existing columns so they can be
    written back out with the new parameter columns appended.

    Returns:
        list of dict, or empty list if the file doesn't exist yet
    """
    if not os.path.exists(PREDICTIONS_PATH):
        return []

    with open(PREDICTIONS_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def estimate_all_planets(verbose=True):
    """
    Reads M3's predictions.csv, finds every row classified as "planet",
    computes the final parameter set for each, and writes the enriched
    results back to predictions.csv with the new columns added.

    Rows NOT classified as "planet" are kept in the output file
    unchanged, just without parameter values (since period/depth/etc.
    are only meaningful for confirmed-as-planet candidates).

    Returns:
        dict with keys: 'total_predictions' (int), 'planets_found' (int),
            'params_computed' (int)
    """
    predictions = load_predictions()

    if not predictions:
        if verbose:
            print("No predictions found in results/predictions.csv. "
                  "Run M3's predict.py first.")
        return {"total_predictions": 0, "planets_found": 0, "params_computed": 0}

    candidates_lookup = load_candidates_lookup()

    if not candidates_lookup:
        if verbose:
            print("No candidates found in results/candidates.csv. "
                  "Run detect.py's run_bls_all() first.")
        return {"total_predictions": len(predictions), "planets_found": 0, "params_computed": 0}

    planet_rows = [p for p in predictions if p.get("predicted_class") == "planet"]

    if verbose:
        print(f"Found {len(planet_rows)} candidates classified as 'planet' "
              f"out of {len(predictions)} total predictions.")

    params_computed = 0
    enriched_predictions = []

    for row in predictions:
        new_row = dict(row)  # copy, so we don't mutate the original

        if row.get("predicted_class") == "planet":
            tic_id = row["TIC_ID"]
            try:
                params = estimate_params(tic_id, candidates_lookup)
                new_row.update(params)
                params_computed += 1
                if verbose:
                    print(f"TIC {tic_id}: period={params['period_days']}d, "
                          f"depth={params['depth_pct']}%, "
                          f"duration={params['duration_hrs']}hr, "
                          f"SNR={params['snr']}")
            except (KeyError, ValueError) as e:
                if verbose:
                    print(f"TIC {tic_id}: could not compute params - {e}")
                new_row.update({
                    "period_days": "", "depth_pct": "",
                    "duration_hrs": "", "snr": "",
                })
        else:
            # Non-planet rows get empty parameter columns, keeping the
            # CSV structure consistent across all rows
            new_row.update({
                "period_days": "", "depth_pct": "",
                "duration_hrs": "", "snr": "",
            })

        enriched_predictions.append(new_row)

    # Write back out with the new columns included
    if enriched_predictions:
        fieldnames = list(enriched_predictions[0].keys())
        with open(PREDICTIONS_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched_predictions)

    if verbose:
        print(f"\nDone. Computed parameters for {params_computed}/{len(planet_rows)} "
              f"planet candidates.")
        print(f"Updated {PREDICTIONS_PATH}")

    return {
        "total_predictions": len(predictions),
        "planets_found": len(planet_rows),
        "params_computed": params_computed,
    }


if __name__ == "__main__":
    estimate_all_planets()
