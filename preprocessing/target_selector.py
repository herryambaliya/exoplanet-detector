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


def get_real_sector_targets(sector, dec_min=None, dec_max=None, num_targets=700):
    """
    Queries MAST directly for the ACTUAL stars observed at 2-minute
    cadence in a given sector -- rather than guessing from a static
    catalog filtered by brightness alone.

    Why this exists, and why it's a better approach than
    filter_catalog(): TESS's real 2-min target selection is driven by
    a specific scientific ranking (the Candidate Target List, CTL) that
    favors small, cool, nearby dwarf stars -- NOT simply "any bright
    star." Filtering a general TIC catalog by Tmag alone (what
    filter_catalog() does) can select many stars that pass the
    brightness cut but were never actually chosen for 2-min cadence,
    because they're giants, too far away, or otherwise not prioritized.
    This was confirmed in practice: two different catalog files, picked
    for two different declination bands, both gave low real hit rates
    (~6-13%) when their Tmag-filtered targets were checked against
    actual sector observations.

    This function flips the approach: instead of filtering a catalog
    and hoping it overlaps with what TESS observed, it asks MAST
    directly "what did you actually observe at 2-min cadence in this
    sector," which guarantees every returned TIC ID has real data
    available -- a 100% hit rate by construction, rather than a
    filtered guess.

    Args:
        sector (int): which TESS sector to query
        dec_min, dec_max (float or None): optional declination range
            in degrees to restrict results to (e.g. -90, -88). If
            None, returns targets across the whole sector.
        num_targets (int): cap on how many results to return

    Returns:
        list of dict, each with keys: TIC_ID, ra, dec
    """
    from astroquery.mast import Observations

    print(f"Querying MAST for real 2-minute cadence observations in sector {sector}...")

    obs = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=sector,
    )

    if len(obs) == 0:
        print(f"No observations found for sector {sector}. Check the sector "
              f"number is valid and has been observed/released yet.")
        return []

    print(f"MAST returned {len(obs)} total timeseries products for this sector.")

    results = []
    seen_tic_ids = set()

    for row in obs:
        try:
            target_name = str(row["target_name"])
            dec = float(row["s_dec"])
            ra = float(row["s_ra"])

            if dec_min is not None and dec < dec_min:
                continue
            if dec_max is not None and dec > dec_max:
                continue

            if target_name in seen_tic_ids:
                continue

            seen_tic_ids.add(target_name)
            results.append({"TIC_ID": target_name, "ra": ra, "dec": dec})

            if len(results) >= num_targets:
                break

        except (KeyError, ValueError, TypeError):
            continue

    print(f"Selected {len(results)} confirmed real targets "
          f"{'(within declination filter)' if dec_min is not None else ''}.")

    if len(results) == 0:
        print("WARNING: No targets matched. If using a declination filter, "
              "try widening dec_min/dec_max, or remove the filter to see "
              "the full sector's real targets.")

    return results


def get_known_planet_anchors():
    """
    Returns a small list of TIC IDs for CONFIRMED exoplanets, verified
    against published literature, that fall within or near this
    project's declination band and were observed in sectors 12/13.

    Why this exists: searching unlabeled stars at random has a real but
    low chance of containing a detectable planet (~0.5-1% of stars
    typically host one detectable via transit). Mixing in a small
    number of KNOWN, CONFIRMED planet hosts guarantees at least one
    true positive to validate the pipeline against, regardless of what
    the random search turns up.

    Verified entries:
        TIC 383390264 (HD 110082 / TOI-1098): confirmed sub-Neptune,
        period 10.1827 days, radius 3.2 Earth radii. Declination
        -88:07:15.72 -- inside this project's catalog band. Observed
        in Sectors 12 and 13 (Tofflemire et al. 2021, arXiv:2102.06066).
        This is the anchor star referenced throughout this project's
        development for pipeline validation.

    Returns:
        list of dict, same shape as filter_catalog() output, so it can
        be directly combined with a random target list
    """
    return [
        {"TIC_ID": "383390264", "ra": 192.5918, "dec": -88.1211, "Tmag": None,
         "note": "HD 110082 / TOI-1098, confirmed planet, period=10.1827d"},
    ]


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
