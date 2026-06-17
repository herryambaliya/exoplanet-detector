"""
preprocess.py
Member 1 - Data Engineer

Purpose:
    Cleans a raw light curve: removes missing values, removes noise
    spikes, removes slow brightness drift, and normalizes it.
    Outputs a clean numpy array that every other module reads.

    THIS IS THE MOST CRITICAL FILE IN THE PROJECT.
    Member 3 imports clean() directly and applies it to the labelled
    training dataset too -- so whatever this function does must be
    IDENTICAL for both TESS data and training data, or the trained
    model will perform badly on real data.

Output format (the contract everyone else codes against):
    data/processed/TIC_{id}.npy
    Loaded with: np.load(path, allow_pickle=True).item()
    Returns: {"time": float32 array, "flux": float32 array}

Usage:
    python preprocess.py
    or
    from preprocessing.preprocess import clean, load_fits, to_array, run_all
"""

import os
import numpy as np
import lightkurve as lk
from preprocessing.downloader import list_downloaded

PROCESSED_DATA_DIR = "data/processed"
FAILED_LOG = "data/processed/failed_preprocessing.txt"


def load_fits(filepath):
    """
    Opens a single FITS file and returns a lightkurve LightCurve object.

    Args:
        filepath (str): path to a .fits file produced by downloader.py

    Returns:
        lightkurve.LightCurve object
    """
    lc = lk.read(filepath)
    return lc


def clean(lc):
    """
    Cleans a LightCurve object. Runs these steps in order:

    1. remove_nans()      - drop rows where flux is missing
    2. remove_outliers()  - drop extreme spikes (cosmic rays, glitches)
    3. flatten() x2       - remove slow brightness drift (Savitzky-Golay
                             filter), done TWICE -- see note below
    4. normalize()        - rescale so the median flux = 1.0

    IMPORTANT - why flatten() runs twice (two-pass flattening):

    lightkurve's flatten() internally sigma-clips outliers (default
    niters=3, sigma=3) WHILE fitting the trend. A real transit dip of
    ~1% looks exactly like an outlier to that sigma-clipper, so a naive
    single flatten() call partially erases real transit signals --
    verified experimentally: a true 1.00% transit depth came out as
    0.27% after a single flatten() call. That is a serious bug for this
    project, since the whole pipeline depends on transit depth being
    accurate.

    The fix lightkurve provides is the `mask` parameter: pass a boolean
    array marking which points are in-transit, and flatten() will skip
    sigma-clipping/fitting on those points and interpolate over them
    instead, leaving the real flux there untouched by the trend fit.

    But we don't know where the transits are yet on a first pass -- that
    is M2's BLS job downstream. So we do a standard two-pass approach:
      Pass 1: flatten with no mask, just to get a rough light curve
              clean enough to run a quick internal BLS on.
      Pass 2: use that rough BLS result to build a transit mask, then
              re-flatten the ORIGINAL outlier-removed light curve with
              that mask applied. This is the version we keep and save.

    This was verified against synthetic data with a known 1.00% depth,
    3.5 day period, 3 hour duration transit: single-pass flatten
    recovered 0.27% depth (wrong), two-pass flatten recovered 0.999%
    depth (correct).

    Args:
        lc (lightkurve.LightCurve): a raw light curve

    Returns:
        lightkurve.LightCurve: the cleaned light curve, with real
            transit signals preserved
    """
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma=5)

    # Pass 1: rough flatten, just to find approximate transit timing
    rough_flat = lc.flatten(window_length=401)

    transit_mask = _find_rough_transit_mask(lc, rough_flat)

    # Pass 2: re-flatten the original (pre-flatten) light curve, this
    # time protecting the suspected transit points from the trend fit
    lc = lc.flatten(window_length=401, mask=transit_mask)
    lc = lc.normalize()
    return lc


def _find_rough_transit_mask(lc_original, lc_rough_flat):
    """
    Internal helper. Runs a fast, low-precision BLS search on a roughly
    flattened light curve just to estimate WHERE the transits might be,
    so clean() can protect those points during the real flatten pass.

    This is intentionally quick and approximate -- the real, careful BLS
    search with full period range and proper thresholding is M2's job in
    detection/detect.py. This is only a preprocessing aid.

    Args:
        lc_original (lightkurve.LightCurve): outlier-removed, not yet flattened
        lc_rough_flat (lightkurve.LightCurve): same data after one flatten() pass

    Returns:
        np.ndarray of bool, same length as lc_original.time, True where
        a point is suspected to be in-transit
    """
    from astropy.timeseries import BoxLeastSquares

    time = lc_rough_flat.time.value
    flux = lc_rough_flat.flux.value

    try:
        model = BoxLeastSquares(time, flux)
        # Quick coarse search -- not the final answer, just a hint.
        # autopower's first argument is the duration *grid* to test as
        # fractions of trial periods. 0.05 was too narrow and made BLS
        # underestimate true transit duration (verified: detected 1.2hr
        # when the true duration was 3hr). A wider duration grid lets it
        # actually find the real duration instead of clipping to the
        # shortest option tested.
        durations_to_test = np.linspace(0.01, 0.2, 10)  # as fraction of period
        periodogram = model.autopower(
            durations_to_test, minimum_period=0.5, maximum_period=15
        )
        best_period = periodogram.period[np.argmax(periodogram.power)]
        best_t0 = periodogram.transit_time[np.argmax(periodogram.power)]
        best_duration = periodogram.duration[np.argmax(periodogram.power)]

        # Build a mask: True for any point near a predicted transit
        # center, repeated every period. We deliberately widen this by
        # 1.5x the detected duration as a safety margin -- the quick
        # coarse BLS search above is imprecise about exact duration, and
        # it is much safer to protect slightly too much (a few extra
        # out-of-transit points skipped from the trend fit, harmless)
        # than too little (part of the real dip still gets clipped as
        # an "outlier" and the recovered depth comes out wrong).
        orig_time = lc_original.time.value
        safety_margin = 1.5
        phase = ((orig_time - best_t0 + 0.5 * best_period) % best_period) - 0.5 * best_period
        mask = np.abs(phase) < (best_duration * safety_margin / 2)
        return mask

    except Exception:
        # If the quick BLS search fails for any reason (too few points,
        # no signal, etc.), fall back to no masking -- better to flatten
        # without protection than to crash the whole pipeline on one star
        return np.zeros(len(lc_original.time), dtype=bool)


def to_array(lc):
    """
    Converts a cleaned LightCurve object into two plain numpy arrays.

    Args:
        lc (lightkurve.LightCurve): a cleaned light curve

    Returns:
        tuple: (time_array, flux_array), both np.float32
    """
    time_array = np.asarray(lc.time.value, dtype=np.float32)
    flux_array = np.asarray(lc.flux.value, dtype=np.float32)
    return time_array, flux_array


def save_processed(tic_id, time_array, flux_array):
    """
    Saves the cleaned arrays to disk in the standard format that
    every other module reads.

    Args:
        tic_id (str): the TESS Input Catalog ID for this star
        time_array (np.ndarray): float32 array of time values (BTJD)
        flux_array (np.ndarray): float32 array of normalized flux values
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, f"TIC_{tic_id}.npy")
    data = {"time": time_array, "flux": flux_array}
    np.save(out_path, data, allow_pickle=True)
    return out_path


def extract_tic_id(filepath):
    """Pulls the TIC ID back out of a filename like TIC_12345.fits"""
    base = os.path.basename(filepath)
    tic_id = base.replace("TIC_", "").replace(".fits", "")
    return tic_id


def process_one(filepath):
    """
    Runs the full chain on a single FITS file: load -> clean -> save.

    Args:
        filepath (str): path to a raw .fits file

    Returns:
        str or None: path to the saved .npy file, or None if it failed
    """
    tic_id = extract_tic_id(filepath)
    lc = load_fits(filepath)
    lc_clean = clean(lc)
    time_array, flux_array = to_array(lc_clean)

    # Sanity check -- a star with almost no data points left after
    # cleaning is not usable. Skip it rather than save junk.
    if len(flux_array) < 100:
        raise ValueError(
            f"Only {len(flux_array)} points remain after cleaning -- too few to use"
        )

    out_path = save_processed(tic_id, time_array, flux_array)
    return out_path


def run_all(sector=1):
    """
    Loops through every downloaded FITS file, cleans it, and saves the
    result. Skips files that have already been processed, so this is
    safe to re-run.

    Returns:
        dict with keys: 'success' (int), 'failed' (int), 'skipped' (int)
    """
    files = list_downloaded(sector=sector)
    if not files:
        print("No downloaded files found. Run downloader.py first.")
        return {"success": 0, "failed": 0, "skipped": 0}

    print(f"Found {len(files)} raw files. Starting preprocessing...")

    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_ids = []

    for i, filepath in enumerate(files):
        tic_id = extract_tic_id(filepath)
        expected_output = os.path.join(PROCESSED_DATA_DIR, f"TIC_{tic_id}.npy")

        if os.path.exists(expected_output):
            skipped_count += 1
            continue

        try:
            process_one(filepath)
            success_count += 1
            print(f"[{i+1}/{len(files)}] Processed TIC {tic_id}")
        except Exception as e:
            failed_count += 1
            failed_ids.append(tic_id)
            print(f"[{i+1}/{len(files)}] FAILED TIC {tic_id}: {e}")
            continue

    if failed_ids:
        with open(FAILED_LOG, "a") as f:
            f.write(f"Sector {sector} failed TIC IDs: {','.join(failed_ids)}\n")

    print(f"\nDone. Success: {success_count}, Failed: {failed_count}, "
          f"Skipped (already done): {skipped_count}")
    return {"success": success_count, "failed": failed_count, "skipped": skipped_count}


if __name__ == "__main__":
    run_all(sector=1)
