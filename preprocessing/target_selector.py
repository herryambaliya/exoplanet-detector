"""
target_selector.py
Member 1 - Data Engineer

Purpose:
    Takes a raw TIC (TESS Input Catalog) bulk CSV file -- like the
    tic_dec*.csv.gz files distributed by MAST -- and filters it down to
    a manageable, useful list of target stars to download light curves
    for. The raw TIC catalog files have ~200 columns of stellar metadata
    (mass, temperature, magnitudes, Gaia IDs, etc.) but NO time-series
    brightness data -- they are a catalog of WHICH stars exist, not
    light curves. This script is the bridge: filter catalog -> pass
    resulting TIC IDs to downloader.py -> get actual light curves.

Why filtering matters:
    A single declination-band TIC file like tic_dec90_00S__88_00S.csv
    contains ~200,000 stars. Downloading light curves for all of them is
    unnecessary, slow, and most are too faint for TESS to detect a clean
    transit on anyway. Filtering by Tmag (TESS magnitude -- LOWER number
    = brighter star) and object type up front means downloader.py only
    fetches stars actually worth analyzing.

TIC bulk CSV columns referenced here (per MAST TIC-8 documentation,
files are headerless, 0-indexed in the comments below to match common
TIC field docs, 1-indexed in the actual awk/pandas column positions):
    column 1  (index 0)   = ID        (TIC ID, used by lightkurve)
    column 12 (index 11)  = objType   ("STAR" or other classifications)
    column 14 (index 13)  = ra        (degrees)
    column 15 (index 14)  = dec       (degrees)
    column 61 (index 60)  = Tmag      (TESS magnitude, brightness)

Output:
    A clean CSV with just: TIC_ID, ra, dec, Tmag
    This is what gets handed to downloader.py.

Usage:
    python target_selector.py
    or
    from preprocessing.target_selector import filter_catalog
    filter_catalog("tic_dec90_00S__88_00S.csv.gz", max_tmag=12, num_targets=300)
"""

import gzip
import csv
import os

# Column positions (1-indexed, matching the raw file structure)
COL_ID = 0
COL_OBJTYPE = 11
COL_RA = 13
COL_DEC = 14
COL_TMAG = 60

OUTPUT_DIR = "data/targets"


def _open_catalog(filepath):
    """Opens a TIC catalog file whether it's gzipped or plain CSV."""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt")
    return open(filepath, "r")


def filter_catalog(filepath, max_tmag=12.0, num_targets=300, object_type="STAR"):
    """
    Reads a raw TIC bulk catalog file and filters it down to a usable
    target list.

    Args:
        filepath (str): path to the tic_dec*.csv or .csv.gz file
        max_tmag (float): only keep stars brighter than this TESS
            magnitude. LOWER Tmag = BRIGHTER star. TESS can reliably
            detect transits on stars roughly Tmag < 12-13; fainter
            stars have too much photon noise for a clean signal.
            Default 12.0 is a reasonable starting cutoff.
        num_targets (int): stop once this many qualifying stars are found.
            Set to None to keep all matches (could be a lot).
        object_type (str): only keep rows where objType matches this
            exactly. "STAR" excludes galaxies and other extended objects
            that occasionally appear in the TIC.

    Returns:
        list of dict, each with keys: TIC_ID, ra, dec, Tmag
    """
    selected = []
    total_seen = 0
    total_skipped_faint = 0
    total_skipped_type = 0
    total_skipped_malformed = 0

    with _open_catalog(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            total_seen += 1

            if num_targets is not None and len(selected) >= num_targets:
                break

            try:
                obj_type = row[COL_OBJTYPE]
                tmag_str = row[COL_TMAG]

                if obj_type != object_type:
                    total_skipped_type += 1
                    continue

                if tmag_str == "" or tmag_str is None:
                    total_skipped_malformed += 1
                    continue

                tmag = float(tmag_str)
                if tmag > max_tmag:
                    total_skipped_faint += 1
                    continue

                selected.append({
                    "TIC_ID": row[COL_ID],
                    "ra": float(row[COL_RA]),
                    "dec": float(row[COL_DEC]),
                    "Tmag": tmag,
                })

            except (IndexError, ValueError):
                total_skipped_malformed += 1
                continue

    print(f"Scanned {total_seen} rows from catalog.")
    print(f"  Skipped (wrong object type): {total_skipped_type}")
    print(f"  Skipped (too faint, Tmag > {max_tmag}): {total_skipped_faint}")
    print(f"  Skipped (malformed row): {total_skipped_malformed}")
    print(f"  Selected: {len(selected)} target stars")

    if len(selected) == 0:
        print(f"\nWARNING: No stars found with Tmag < {max_tmag} in this file. "
              f"This declination band may only contain faint stars -- try "
              f"raising max_tmag, or use a different tic_dec*.csv.gz file "
              f"covering a different part of the sky.")

    return selected


def save_target_list(targets, out_filename="target_list.csv"):
    """
    Saves the filtered target list to a clean CSV that downloader.py
    (or any team member) can read easily.

    Args:
        targets (list of dict): output from filter_catalog()
        out_filename (str): filename to save under data/targets/

    Returns:
        str: path to the saved file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_filename)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["TIC_ID", "ra", "dec", "Tmag"])
        writer.writeheader()
        writer.writerows(targets)

    print(f"Saved {len(targets)} targets to {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python target_selector.py <path_to_tic_dec_file.csv.gz>")
        sys.exit(1)

    catalog_path = sys.argv[1]
    targets = filter_catalog(catalog_path, max_tmag=12.0, num_targets=300)
    if targets:
        save_target_list(targets)
