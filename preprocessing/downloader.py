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
import warnings
import lightkurve as lk
from astropy.utils.exceptions import AstropyWarning

# Suppress cosmetic AstropyWarning/UnitsWarning messages about TESS's
# non-standard flux units ('e-/s', 'pixels') in FITS headers. These are
# harmless -- they don't affect the actual data values, only astropy's
# strictness about unit string formatting -- but they print very
# verbosely, especially with many parallel download threads, which was
# observed to contribute to Jupyter notebook stdout instability during
# heavy concurrent downloads.
warnings.filterwarnings("ignore", category=AstropyWarning)

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


def _download_worker_process(tic_id, sector, sector_dir, result_queue):
    """
    Runs inside a SEPARATE PROCESS (not a thread). This is the key fix
    for a real bug found in practice: Python threads cannot be force-
    killed once started, so a single hung network request inside a
    thread permanently occupies that thread forever, and
    ThreadPoolExecutor/ProcessPoolExecutor's shutdown also waits for
    all workers by default -- confirmed experimentally that even
    shutdown(wait=False) and cancel_futures=True did not allow the
    Python process itself to exit while a stuck worker thread was
    still alive, since non-daemon threads block process exit
    regardless of what the main thread does.

    multiprocessing.Process is different: the OS-level process CAN be
    forcefully killed with .terminate(), which actually works even if
    the code inside is stuck in a network call. This was verified
    directly: a process stuck in an artificial 9999-second sleep was
    successfully terminated within ~3 seconds using this approach,
    whereas every thread-based approach hung indefinitely.

    Args:
        tic_id, sector, sector_dir: same meaning as elsewhere
        result_queue (multiprocessing.Queue): used to send the result
            back to the parent process, since Process doesn't return
            a value directly the way a function call does
    """
    import warnings
    from astropy.utils.exceptions import AstropyWarning
    warnings.filterwarnings("ignore", category=AstropyWarning)
    import lightkurve as lk

    out_path = os.path.join(sector_dir, f"TIC_{tic_id}.fits")

    if os.path.exists(out_path):
        result_queue.put((tic_id, "success", "already exists"))
        return

    try:
        search_result = lk.search_lightcurve(
            f"TIC {tic_id}",
            sector=sector,
            mission="TESS",
            author="SPOC",
            cadence="short",
        )

        if len(search_result) == 0:
            result_queue.put((tic_id, "no_data", "no short-cadence data"))
            return

        lc = search_result[0].download()
        if lc is None:
            result_queue.put((tic_id, "failed", "download() returned None"))
            return

        lc.to_fits(out_path, overwrite=True)
        result_queue.put((tic_id, "success", "downloaded"))

    except Exception as e:
        result_queue.put((tic_id, "failed", str(e)))


def download_targets_parallel(tic_ids, sector=1, max_workers=8, timeout_seconds=30):
    """
    Downloads light curves for multiple TIC IDs using SEPARATE
    PROCESSES (not threads), running up to max_workers at a time, with
    a real, enforceable timeout per star.

    WHY PROCESSES INSTEAD OF THREADS (found through direct testing,
    not just theory): an earlier thread-based version of this function
    used ThreadPoolExecutor with future.result(timeout=...). This
    looked correct, but a real run on 65 stars slowed progressively
    and then stalled completely. Root cause, confirmed experimentally:
    Python threads cannot be force-killed once started. A single star
    whose network request hangs (ignoring astropy's configured
    timeout, which doesn't reach every code path inside lightkurve's
    .download()) leaves that thread permanently stuck. Worse,
    ThreadPoolExecutor's shutdown (including via the `with` statement)
    blocks waiting for ALL worker threads by default, including stuck
    ones -- verified directly that even shutdown(wait=False) does not
    let the underlying Python process exit while a non-daemon worker
    thread is still alive.

    multiprocessing.Process is different: calling .terminate() on a
    stuck process actually kills it at the OS level, regardless of
    what code it's stuck running. This was verified directly: an
    artificially stuck worker (sleeping 9999 seconds) was successfully
    terminated within seconds using this approach.

    Args:
        tic_ids (list of str or int): TIC IDs to download
        sector (int): which TESS sector to pull from
        max_workers (int): how many downloads to run at once
        timeout_seconds (int): max seconds to wait per star before
            forcefully terminating that star's worker process and
            moving on

    Returns:
        dict with keys: 'success' (int), 'failed' (int)
    """
    import multiprocessing as mp

    sector_dir = os.path.join(RAW_DATA_DIR, f"sector_{sector}")
    os.makedirs(sector_dir, exist_ok=True)

    print(f"Downloading {len(tic_ids)} targets from sector {sector} "
          f"using up to {max_workers} parallel processes "
          f"(hard timeout: {timeout_seconds}s per star)...")

    success_count = 0
    failed_count = 0
    failed_ids = []
    completed = 0

    pending = list(tic_ids)
    running = {}  # tic_id -> (Process, Queue, start_time)

    while pending or running:
        # Start new workers up to max_workers
        while pending and len(running) < max_workers:
            tic_id = pending.pop(0)
            q = mp.Queue()
            p = mp.Process(
                target=_download_worker_process,
                args=(tic_id, sector, sector_dir, q),
            )
            p.start()
            running[tic_id] = (p, q, time.time())

        # Check all running workers for completion or timeout
        finished_tic_ids = []
        for tic_id, (p, q, start_time) in running.items():
            elapsed = time.time() - start_time

            if not q.empty():
                result = q.get()
                _, status, message = result
                completed += 1

                if status == "success":
                    success_count += 1
                else:
                    failed_count += 1
                    failed_ids.append(str(tic_id))

                try:
                    print(f"[{completed}/{len(tic_ids)}] TIC {tic_id}: {message}")
                except (ValueError, OSError):
                    pass

                p.join(timeout=2)
                finished_tic_ids.append(tic_id)

            elif elapsed > timeout_seconds:
                # Hard kill -- this is the part that actually works,
                # unlike thread-based timeouts
                p.terminate()
                p.join(timeout=2)
                completed += 1
                failed_count += 1
                failed_ids.append(str(tic_id))

                try:
                    print(f"[{completed}/{len(tic_ids)}] TIC {tic_id}: "
                          f"timed out after {timeout_seconds}s, force-killed")
                except (ValueError, OSError):
                    pass

                finished_tic_ids.append(tic_id)

        for tic_id in finished_tic_ids:
            del running[tic_id]

        if running:
            time.sleep(0.2)  # avoid busy-waiting the CPU at 100%

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
