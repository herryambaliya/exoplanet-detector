import numpy as np
import lightkurve as lk
import joblib
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from astropy.timeseries import BoxLeastSquares
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

clf    = joblib.load("model/rf_4class.pkl")
scaler = joblib.load("model/scaler_4class.pkl")
CLASS_NAMES = ["planet", "eclipsing_binary", "blend", "other"]

def predict_star(tic_id, sector=None, plot=True):
    print(f"\n======  Predicting TIC {tic_id}  ======")

    # Step 1: Download
    print("Step 1: Downloading...")
    try:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector)
        if len(search) == 0:
            print("  No data found")
            return None
        lc = search[0].download()
        print(f"  Downloaded sector {search.table['sequence_number'][0]}")
    except Exception as e:
        print(f"  Failed: {e}")
        return None

    # Step 2: Clean
    print("Step 2: Cleaning...")
    lc   = lc.remove_nans().remove_outliers(sigma=5)
    lc   = lc.flatten(window_length=401).normalize()
    time = np.asarray(lc.time.value, dtype=np.float64)
    flux = np.asarray(lc.flux.value, dtype=np.float64)
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    print(f"  {len(time)} points")

    # Step 3: BLS
    print("Step 3: BLS period finding...")
    model  = BoxLeastSquares(time, flux/np.median(flux))
    pg     = model.autopower(
        duration=[0.05,0.1,0.2],
        minimum_period=0.5,
        maximum_period=15.0
    )
    period = float(pg.period[np.argmax(pg.power)])
    print(f"  Period: {period:.4f} days")

    # Step 4: Phase fold
    print("Step 4: Phase folding...")
    flux_norm = flux / np.median(flux)
    phase     = (time % period) / period
    idx       = np.argsort(phase)
    grid      = np.linspace(0, 1, 200)
    flux_200  = np.interp(grid, phase[idx], flux_norm[idx])
    min_i     = np.argmin(flux_200)
    flux_200  = np.roll(flux_200, 100-min_i).astype(np.float32)

    # Step 5: Extract features
    print("Step 5: Classifying...")
    f = flux_200.copy()
    m = np.median(f)
    if m > 0: f = f/m
    f = np.clip(f, 0.5, 1.5)

    features = []
    for hw in [5,10,15,20,30]:
        center = f[100-hw:100+hw]
        oot    = np.concatenate([f[:100-hw], f[100+hw:]])
        features += [
            float(np.median(oot)-np.min(center)),
            float(np.mean(oot)-np.mean(center)),
            float((np.mean(oot)-np.mean(center))/(np.std(oot)+1e-9)),
        ]
    left  = f[50:100]
    right = f[100:150][::-1]
    c20   = f[90:110]
    edge  = np.concatenate([f[:15], f[185:]])
    thresh = float(np.median(f)-2*np.std(f))
    features += [
        float(np.mean(np.abs(left-right))),
        float(c20[0]-np.min(c20)),
        float(np.std(c20)),
        float(np.mean(f[90:110])-np.min(edge)),
        float(skew(f)), float(kurtosis(f)),
        float(np.std(f)),
        float(np.sqrt(np.mean((f-1.0)**2))),
        float(np.percentile(f,1)),
        float(np.percentile(f,5)),
        float(np.percentile(f,95)),
        float(np.percentile(f,99)),
        float(np.min(f)), float(np.max(f)),
        float(np.max(f)-np.min(f)),
        float(np.sum(f[85:115]<thresh)),
        float(np.sum(f<thresh)),
        float(np.mean(np.diff(f[90:100]))),
        float(np.mean(np.diff(f[100:110]))),
    ]

    X     = np.array([features], dtype=np.float32)
    X_s   = scaler.transform(X)
    probs = clf.predict_proba(X_s)[0]
    pred  = CLASS_NAMES[clf.predict(X_s)[0]]
    conf  = float(np.max(probs))

    # Step 6: Parameter estimation if planet
    params = {}
    if pred == "planet":
        print("Step 6: Estimating parameters...")
        try:
            phase_arr = np.linspace(-0.5, 0.5, 200)
            depth0    = float(max(np.median(f)-np.min(f[85:115]), 1e-5))

            def trapezoid(x, depth, dur, ing, cen):
                ing = np.clip(ing, 1e-4, 0.499)
                t12 = ing*dur; t14 = dur/2
                p   = x-cen; fl = np.ones_like(p)
                fl[np.abs(p)<=(t14-t12)] = 1-depth
                mask = (np.abs(p)>(t14-t12))&(np.abs(p)<=t14)
                fl[mask] = 1-depth*(t14-np.abs(p[mask]))/(t12+1e-9)
                return fl

            popt, pcov = curve_fit(
                trapezoid, phase_arr, f,
                p0=[depth0,0.1,0.15,0.0],
                bounds=([1e-6,0.01,0.02,-0.4],[0.5,0.45,0.49,0.4]),
                maxfev=5000
            )
            perr = np.sqrt(np.diag(pcov))
            depth, dur, ing, cen = popt
            oot_  = f[np.abs(phase_arr-cen)>dur]
            in_   = f[np.abs(phase_arr-cen)<=dur/2]
            rms   = float(np.std(oot_)) if len(oot_)>2 else 1e-4
            snr   = float((1-np.mean(in_))/(rms/np.sqrt(max(len(in_),1))))
            params = {
                "depth_ppm"      : round(float(depth*1e6),1),
                "depth_ppm_err"  : round(float(perr[0]*1e6),1),
                "duration_hours" : round(float(dur*period*24),3),
                "ingress_fraction": round(float(ing),3),
                "transit_snr"    : round(snr,2),
            }
        except Exception as e:
            print(f"  Param estimation failed: {e}")

    # Plot
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14,4))

        axes[0].plot(time, flux_norm, ".", ms=1, color="steelblue", alpha=0.5)
        axes[0].set_xlabel("Time (BTJD)")
        axes[0].set_ylabel("Normalised Flux")
        axes[0].set_title(f"TIC {tic_id} — Raw Light Curve")

        color = {"planet":"gold","eclipsing_binary":"tomato",
                 "blend":"violet","other":"steelblue"}.get(pred,"gray")
        axes[1].plot(np.linspace(-0.5,0.5,200), flux_200, ".", ms=3, color=color)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Phase")
        axes[1].set_ylabel("Normalised Flux")
        axes[1].set_title(
            f"TIC {tic_id} — Folded\n"
            f"Class: {pred} | Conf: {conf:.1%} | Period: {period:.4f}d"
        )
        plt.tight_layout()
        Path("results").mkdir(exist_ok=True)
        out = f"results/quick_TIC{tic_id}.png"
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"  Plot saved: {out}")

    # Print result
    print(f"""
+----------------------------------------+
  TIC {tic_id}
  Class      : {pred.upper()}
  Confidence : {conf:.1%}
  Period     : {period:.6f} days ({period*24:.2f} hours)
  Probs:
    planet           : {probs[0]:.1%}
    eclipsing_binary : {probs[1]:.1%}
    blend            : {probs[2]:.1%}
    other            : {probs[3]:.1%}""")

    if params:
        print(f"""  Transit Parameters:
    depth    : {params.get("depth_ppm")} +/- {params.get("depth_ppm_err")} ppm
    duration : {params.get("duration_hours")} hours
    SNR      : {params.get("transit_snr")}""")

    print("+----------------------------------------+")

    return {
        "tic_id"    : tic_id,
        "class"     : pred,
        "confidence": conf,
        "period"    : period,
        "probs"     : dict(zip(CLASS_NAMES, probs)),
        "params"    : params,
        "flux_200"  : flux_200,
    }
