import os
import csv
import numpy as np
from astropy.timeseries import BoxLeastSquares
from concurrent.futures import ProcessPoolExecutor, as_completed

PROCESSED_DIR   = "data/processed"
RESULTS_DIR     = "results"
CANDIDATES_PATH = os.path.join(RESULTS_DIR, "candidates.csv")
POWER_THRESHOLD = 0.005

def run_bls_single(filepath):
    """Process one star — runs in separate process."""
    try:
        tic_id = os.path.basename(filepath).replace('TIC_','').replace('.npy','')
        data   = np.load(filepath, allow_pickle=True).item()
        time   = np.asarray(data['time'], dtype=np.float64)
        flux   = np.asarray(data['flux'], dtype=np.float64)

        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]
        if len(time) < 200:
            return None

        min_period    = 0.5
        max_period    = 27.0
        duration_grid = [0.01, 0.02, 0.05, 0.1]

        model       = BoxLeastSquares(time, flux)
        periodogram = model.autopower(
            duration_grid,
            minimum_period=min_period,
            maximum_period=max_period,
        )

        top3     = np.argsort(periodogram.power)[::-1][:3]
        best_idx = top3[0]
        best_p   = float(periodogram.period[best_idx])
        power    = float(periodogram.power[best_idx])

        return {
            'TIC_ID'      : tic_id,
            'period_days' : best_p,
            'power'       : power,
            'depth'       : float(periodogram.depth[best_idx]),
            'duration_days': float(periodogram.duration[best_idx]),
            't0'          : float(periodogram.transit_time[best_idx]),
            'alt_periods' : [
                best_p*2, best_p/2,
                float(periodogram.period[top3[1]]) if len(top3)>1 else best_p,
                float(periodogram.period[top3[2]]) if len(top3)>2 else best_p,
            ]
        }
    except:
        return None

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = sorted([
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith('.npy') and
        not os.path.isdir(os.path.join(PROCESSED_DIR, f))
    ])
    print(f"Total stars: {len(files)}")

    candidates = []
    all_results = []
    done = 0

    # Use 4 parallel workers
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_bls_single, f): f for f in files}

        for future in as_completed(futures):
            done += 1
            result = future.result()

            if result:
                all_results.append(result)
                if result['power'] >= POWER_THRESHOLD:
                    candidates.append(result)

            if done % 100 == 0:
                print(f"  [{done}/{len(files)}] "
                      f"candidates so far: {len(candidates)}")

    # Save
    with open(CANDIDATES_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TIC_ID','period_days','power','depth',
                         'duration_days','t0','alt_periods'])
        for c in sorted(candidates, key=lambda x: x['power'], reverse=True):
            writer.writerow([
                c['TIC_ID'], c['period_days'], c['power'],
                c['depth'], c['duration_days'], c['t0'],
                str(c['alt_periods'])
            ])

    powers = sorted([r['power'] for r in all_results], reverse=True)
    print(f"\n✓ Done!")
    print(f"  Stars processed : {len(all_results)}")
    print(f"  Candidates found: {len(candidates)}")
    print(f"\nPower distribution:")
    print(f"  Top 5  : {[round(p,5) for p in powers[:5]]}")
    print(f"  Median : {powers[len(powers)//2]:.5f}")
    print(f"  Above {POWER_THRESHOLD}: {len(candidates)}")
    print(f"\n✓ Saved → {CANDIDATES_PATH}")