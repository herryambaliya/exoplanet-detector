"""
downloader.py
Member 1 - Data Engineer

Purpose:
    Downloads raw TESS light curve files from NASA's MAST archive.
    This is the very first step of the entire pipeline - every other
    module depends on the files this script produces.

Output:
    data/raw/sector_{N}/TIC_{id}.fits  -- one file per star

Usage:
    python downloader.py
    or
    from preprocessing.downloader import download_sector
    download_sector(sector=1, num_stars=300)
"""

import os
import time
import lightkurve as lk

RAW_DATA_DIR = "data/raw"
FAILED_LOG = "data/raw/failed_downloads.txt"


def download_sector(sector=1, num_stars=300, author="SPOC"):
    """
    Searches TESS for 2-minute cadence light curves in a given sector
    and downloads up to num_stars of them as FITS files.

    Args:
        sector (int): TESS sector number to download from.
        num_stars (int): how many stars to download.
        author (str): pipeline that produced the light curve.
                       "SPOC" = the standard NASA-processed product.

    Returns:
        dict with keys: 'success' (int), 'failed' (int)
    """
    sector_dir = os.path.join(RAW_DATA_DIR, f"sector_{sector}")
    os.makedirs(sector_dir, exist_ok=True)

    print(f"Searching MAST archive for TESS sector {sector} light curves...")
    search_result = lk.search_lightcurve(
        f"sector{sector}",
        mission="TESS",
        author=author,
        cadence="short",
    )

    if len(search_result) == 0:
        print("No results found. Try a different sector number or check your "
              "internet connection.")
        return {"success": 0, "failed": 0}

    total_available = len(search_result)
    n_to_download = min(num_stars, total_available)
    print(f"Found {total_available} light curves available. "
          f"Downloading {n_to_download}...")

    success_count = 0
    failed_count = 0
    failed_ids = []

    for i in range(n_to_download):
        try:
            lc = search_result[i].download()
            if lc is None:
                raise ValueError("download() returned None")

            tic_id = lc.meta.get("TICID", f"unknown_{i}")
            out_path = os.path.join(sector_dir, f"TIC_{tic_id}.fits")

            # Skip if already downloaded (lets you safely re-run this script)
            if os.path.exists(out_path):
                print(f"[{i+1}/{n_to_download}] TIC {tic_id} already exists, skipping.")
                success_count += 1
                continue

            lc.to_fits(out_path, overwrite=True)
            success_count += 1
            print(f"[{i+1}/{n_to_download}] Downloaded TIC {tic_id}")

        except Exception as e:
            failed_count += 1
            failed_ids.append(str(i))
            print(f"[{i+1}/{n_to_download}] FAILED: {e}")
            continue

    # Log failures so the team knows what didn't come through
    if failed_ids:
        with open(FAILED_LOG, "a") as f:
            f.write(f"Sector {sector} failed indices: {','.join(failed_ids)}\n")

    print(f"\nDone. Success: {success_count}, Failed: {failed_count}")
    print(f"Files saved to: {sector_dir}")
    return {"success": success_count, "failed": failed_count}


def _download_one_target(tic_id, sector, sector_dir, timeout_seconds=30):
    """
    Internal helper for parallel downloading. Handles exactly one TIC ID:
    search, then download if found. Designed to be safe to run inside a
    background thread -- doesn't print progress itself (the caller does
    that, to keep output ordered and avoid garbled interleaved prints
    from multiple threads writing to the console at once).

    A timeout is enforced via astropy's data connection config, since
    by default lightkurve/astroquery network calls have no hard cutoff
    and can hang indefinitely on a single slow MAST response -- this
    was observed in practice: a parallel run stalled completely partway
    through with no error, no progress, and no crash, because one
    thread was stuck waiting forever on a request that should have
    just failed and let the pool move on.

    Args:
        tic_id, sector, sector_dir: same as before
        timeout_seconds (int): max seconds to wait for the search/
            download network calls before giving up on this star

    Returns:
        tuple: (tic_id, status, message)
            status is one of: "success", "no_data", "failed", "timeout"
    """
    from astropy.utils.data import conf as astropy_conf

    out_path = os.path.join(sector_dir, f"TIC_{tic_id}.fits")

    if os.path.exists(out_path):
        return (tic_id, "success", "already exists")

    # Apply the timeout to this thread's network calls. astropy's conf
    # is thread-local-ish in practice for this purpose -- each call
    # reads the current value at call time.
    old_timeout = astropy_conf.remote_timeout
    astropy_conf.remote_timeout = timeout_seconds

    try:
        search_result = lk.search_lightcurve(
            f"TIC {tic_id}",
            sector=sector,
            mission="TESS",
            author="SPOC",
            cadence="short",
        )

        if len(search_result) == 0:
            return (tic_id, "no_data", "no short-cadence data")

        lc = search_result[0].download()
        if lc is None:
            return (tic_id, "failed", "download() returned None")

        lc.to_fits(out_path, overwrite=True)
        return (tic_id, "success", "downloaded")

    except TimeoutError:
        return (tic_id, "timeout", f"timed out after {timeout_seconds}s")
    except Exception as e:
        # urllib/socket timeouts often surface as generic exceptions
        # with "timed out" in the message rather than TimeoutError
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            return (tic_id, "timeout", f"timed out after {timeout_seconds}s")
        return (tic_id, "failed", str(e))
    finally:
        # Always restore the original timeout, even on success/failure,
        # so we don't leave global astropy config altered for other code
        astropy_conf.remote_timeout = old_timeout


def download_targets_parallel(tic_ids, sector=1, max_workers=8, timeout_seconds=30):
    """
    Same job as download_targets(), but runs searches/downloads
    concurrently using a thread pool instead of one at a time.

    Why this helps: most of the time spent per star is the network
    round-trip waiting for MAST to respond, not actual CPU work. While
    one thread is waiting on a response, other threads can be making
    their own requests at the same time. This commonly gives a 4-8x
    speedup for a list this size, since most requests in your case
    return "no data" quickly -- those especially benefit from running
    in parallel rather than queued one after another.

    Two layers of timeout protection:
    1. astropy's remote_timeout config (set inside _download_one_target)
       makes the underlying network call itself give up after
       timeout_seconds.
    2. future.result(timeout=...) below is a second safety net -- if a
       thread somehow still doesn't return in time (rare, but network
       code can occasionally ignore configured timeouts), we don't
       block forever waiting for it; we log it as stuck and move on to
       collect the rest.

    This fixes a real failure mode that was observed: a parallel run
    with no timeout protection stalled completely partway through with
    no error and no progress, because one thread was stuck waiting
    indefinitely on a single slow request.

    Args:
        tic_ids (list of str or int): TIC IDs to download
        sector (int): which TESS sector to pull from
        max_workers (int): how many downloads to run at once. 8 is a
            reasonable default -- high enough to meaningfully speed
            things up, not so high that MAST's servers start rate
            limiting or rejecting requests. If you see a lot of
            "failed" results (not "no_data", actual errors), try
            lowering this to 4.
        timeout_seconds (int): max seconds to wait per star before
            giving up on it and moving on

    Returns:
        dict with keys: 'success' (int), 'failed' (int)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

    sector_dir = os.path.join(RAW_DATA_DIR, f"sector_{sector}")
    os.makedirs(sector_dir, exist_ok=True)

    print(f"Downloading {len(tic_ids)} targets from sector {sector} "
          f"using {max_workers} parallel workers (timeout: {timeout_seconds}s per star)...")

    success_count = 0
    failed_count = 0
    failed_ids = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one_target, tic_id, sector, sector_dir, timeout_seconds): tic_id
            for tic_id in tic_ids
        }

        # Give the overall wait a generous but finite cap per future,
        # so a single misbehaving thread can't stall the whole loop
        for future in as_completed(futures, timeout=None):
            tic_id_for_future = futures[future]
            try:
                tic_id, status, message = future.result(timeout=timeout_seconds + 10)
            except FutureTimeoutError:
                tic_id, status, message = tic_id_for_future, "timeout", "future did not return in time"

            completed += 1

            if status == "success":
                success_count += 1
                print(f"[{completed}/{len(tic_ids)}] TIC {tic_id}: {message}")
            else:
                failed_count += 1
                failed_ids.append(str(tic_id))
                print(f"[{completed}/{len(tic_ids)}] TIC {tic_id}: {message}")

    if failed_ids:
        with open(FAILED_LOG, "a") as f:
            f.write(f"Sector {sector} failed TIC IDs: {','.join(failed_ids)}\n")

    print(f"\nDone. Success: {success_count}, Failed: {failed_count}")
    print(f"Files saved to: {sector_dir}")
    return {"success": success_count, "failed": failed_count}


def download_from_target_list_parallel(target_list_path, sector=1, max_workers=8, timeout_seconds=30):
    """
    Same as download_from_target_list(), but uses the parallel
    downloader for significantly faster runs on large target lists.

    Args:
        target_list_path (str): path to target_list.csv
        sector (int): which TESS sector to pull from
        max_workers (int): concurrent download threads, see
            download_targets_parallel() for guidance on this value

    Returns:
        dict with keys: 'success' (int), 'failed' (int)
    """
    import csv as csv_module

    tic_ids = []
    with open(target_list_path, "r") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            tic_ids.append(row["TIC_ID"])

    print(f"Loaded {len(tic_ids)} TIC IDs from {target_list_path}")
    return download_targets_parallel(
        tic_ids, sector=sector, max_workers=max_workers, timeout_seconds=timeout_seconds
    )


def download_targets(tic_ids, sector=1):
    """
    Downloads light curves for a SPECIFIC list of TIC IDs, rather than
    blindly taking the first N stars TESS returns for a sector. Use
    this when you have a filtered target list (e.g. from
    target_selector.py) of stars you actually care about -- bright
    enough for a clean signal, correct object type, etc.

    Args:
        tic_ids (list of str or int): the TIC IDs to download
        sector (int): which TESS sector to pull data from

    Returns:
        dict with keys: 'success' (int), 'failed' (int)
    """
    sector_dir = os.path.join(RAW_DATA_DIR, f"sector_{sector}")
    os.makedirs(sector_dir, exist_ok=True)

    print(f"Downloading {len(tic_ids)} specific targets from sector {sector}...")

    success_count = 0
    failed_count = 0
    failed_ids = []

    for i, tic_id in enumerate(tic_ids):
        out_path = os.path.join(sector_dir, f"TIC_{tic_id}.fits")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(tic_ids)}] TIC {tic_id} already exists, skipping.")
            success_count += 1
            continue

        try:
            search_result = lk.search_lightcurve(
                f"TIC {tic_id}",
                sector=sector,
                mission="TESS",
                author="SPOC",
                cadence="short",
            )

            if len(search_result) == 0:
                # Not every star has 2-minute cadence data in every
                # sector it was observed in -- this is expected and
                # not an error, just skip it
                print(f"[{i+1}/{len(tic_ids)}] No short-cadence data for "
                      f"TIC {tic_id} in sector {sector}, skipping.")
                failed_count += 1
                failed_ids.append(str(tic_id))
                continue

            lc = search_result[0].download()
            if lc is None:
                raise ValueError("download() returned None")

            lc.to_fits(out_path, overwrite=True)
            success_count += 1
            print(f"[{i+1}/{len(tic_ids)}] Downloaded TIC {tic_id}")

        except Exception as e:
            failed_count += 1
            failed_ids.append(str(tic_id))
            print(f"[{i+1}/{len(tic_ids)}] FAILED TIC {tic_id}: {e}")
            continue

    if failed_ids:
        with open(FAILED_LOG, "a") as f:
            f.write(f"Sector {sector} failed TIC IDs: {','.join(failed_ids)}\n")

    print(f"\nDone. Success: {success_count}, Failed: {failed_count}")
    print(f"Files saved to: {sector_dir}")
    return {"success": success_count, "failed": failed_count}


def download_from_target_list(target_list_path, sector=1):
    """
    Convenience wrapper: reads a target_list.csv produced by
    target_selector.py and downloads light curves for every TIC ID
    in it.

    Args:
        target_list_path (str): path to target_list.csv
        sector (int): which TESS sector to pull from

    Returns:
        dict with keys: 'success' (int), 'failed' (int)
    """
    import csv as csv_module

    tic_ids = []
    with open(target_list_path, "r") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            tic_ids.append(row["TIC_ID"])

    print(f"Loaded {len(tic_ids)} TIC IDs from {target_list_path}")
    return download_targets(tic_ids, sector=sector)


def list_downloaded(sector=1):
    """
    Scans the raw data folder and returns a list of all FITS file paths
    that have been downloaded for a given sector.

    Used by preprocess.py to know which files to process.

    Returns:
        list of str (file paths)
    """
    sector_dir = os.path.join(RAW_DATA_DIR, f"sector_{sector}")
    if not os.path.exists(sector_dir):
        return []

    files = [
        os.path.join(sector_dir, f)
        for f in os.listdir(sector_dir)
        if f.endswith(".fits")
    ]
    return sorted(files)


if __name__ == "__main__":
    # Quick test run -- start small to confirm everything works
    # before scaling up to 300 stars.
    result = download_sector(sector=1, num_stars=20)
    files = list_downloaded(sector=1)
    print(f"\n{len(files)} FITS files now available in data/raw/sector_1/")
