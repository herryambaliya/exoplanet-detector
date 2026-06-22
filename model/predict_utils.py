"""
predict_utils.py
----------------
Shared prediction utility for the exoplanet detection pipeline.
Anyone on the team can use this with one import.

Usage:
    from model.predict_utils import classify_candidates
    results = classify_candidates('results/folded_candidates.npy')
"""

import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_DIR     = Path(__file__).parent
MODEL_4CLASS  = MODEL_DIR / 'rf_4class.pkl'
SCALER_4CLASS = MODEL_DIR / 'scaler_4class.pkl'
MODEL_3CLASS  = MODEL_DIR / 'rf_3class.pkl'
SCALER_3CLASS = MODEL_DIR / 'scaler_3class.pkl'

CLASS_NAMES_4 = ['planet', 'eclipsing_binary', 'blend', 'other']
CLASS_NAMES_3 = ['planet', 'eclipsing_binary', 'background']

def _load_models():
    clf    = joblib.load(MODEL_4CLASS)
    scaler = joblib.load(SCALER_4CLASS)
    return clf, scaler

def _extract_features(flux):
    flux     = np.asarray(flux, dtype=np.float32).flatten()
    features = []
    for hw in [5, 10, 15, 20, 30]:
        center    = flux[100-hw:100+hw]
        oot       = np.concatenate([flux[:100-hw], flux[100+hw:]])
        features += [
            float(np.median(oot) - np.min(center)),
            float(np.mean(oot) - np.mean(center)),
            float((np.mean(oot)-np.mean(center))/(np.std(oot)+1e-9)),
        ]
    left  = flux[50:100]
    right = flux[100:150][::-1]
    c20   = flux[90:110]
    edge  = np.concatenate([flux[:15], flux[185:]])
    thresh = float(np.median(flux) - 2*np.std(flux))
    features += [
        float(np.mean(np.abs(left-right))),
        float(c20[0]-np.min(c20)),
        float(np.std(c20)),
        float(np.mean(flux[90:110])-np.min(edge)),
        float(skew(flux)), float(kurtosis(flux)),
        float(np.std(flux)),
        float(np.sqrt(np.mean((flux-1.0)**2))),
        float(np.percentile(flux,1)),
        float(np.percentile(flux,5)),
        float(np.percentile(flux,95)),
        float(np.percentile(flux,99)),
        float(np.min(flux)), float(np.max(flux)),
        float(np.max(flux)-np.min(flux)),
        float(np.sum(flux[85:115]<thresh)),
        float(np.sum(flux<thresh)),
        float(np.mean(np.diff(flux[90:100]))),
        float(np.mean(np.diff(flux[100:110]))),
    ]
    return features

def _estimate_transit_params(flux):
    """Fit trapezoidal model to planet candidate."""
    try:
        phase  = np.linspace(-0.5, 0.5, 200)
        f      = flux.copy()
        median = np.median(f)
        if median > 0: f = f / median

        depth0  = float(max(np.median(f) - np.min(f[85:115]), 1e-5))
        center0 = float(phase[np.argmin(f)])

        def trapezoid(x, depth, duration, ing_frac, center):
            ing_frac = np.clip(ing_frac, 1e-4, 0.499)
            t12  = ing_frac * duration
            t14  = duration / 2.0
            p    = x - center
            fl   = np.ones_like(p)
            fl[np.abs(p) <= (t14-t12)] = 1.0 - depth
            mask = (np.abs(p) > (t14-t12)) & (np.abs(p) <= t14)
            fl[mask] = 1.0 - depth*(t14-np.abs(p[mask]))/(t12+1e-9)
            return fl

        popt, pcov = curve_fit(
            trapezoid, phase, f,
            p0=[depth0, 0.1, 0.15, center0],
            bounds=([1e-6,0.01,0.02,-0.4],[0.5,0.45,0.49,0.4]),
            maxfev=5000
        )
        perr  = np.sqrt(np.diag(pcov))
        depth, duration, ing, center = popt

        oot    = f[np.abs(phase-center) > duration]
        in_tr  = f[np.abs(phase-center) <= duration/2]
        oot_rms = float(np.std(oot)) if len(oot)>2 else 1e-4
        snr    = float((1-np.mean(in_tr))/(oot_rms/np.sqrt(max(len(in_tr),1))))

        return {
            'depth_ppm'        : round(float(depth*1e6), 1),
            'depth_ppm_err'    : round(float(perr[0]*1e6), 1),
            'duration_fraction': round(float(duration), 4),
            'duration_frac_err': round(float(perr[1]), 4),
            'ingress_fraction' : round(float(ing), 3),
            'transit_snr'      : round(snr, 2),
        }
    except:
        return {
            'depth_ppm': None, 'depth_ppm_err': None,
            'duration_fraction': None, 'duration_frac_err': None,
            'ingress_fraction': None, 'transit_snr': None,
        }

def classify_candidates(
    npy_path='results/folded_candidates.npy',
    save_csv=True,
    out_path='results/predictions.csv',
    estimate_params=True,
):
    """
    Main function — classify all candidates and save predictions.

    Parameters
    ----------
    npy_path      : path to folded_candidates.npy from M2
    save_csv      : save results to CSV
    out_path      : where to save CSV
    estimate_params: run transit parameter estimation on planets

    Returns
    -------
    pandas DataFrame with all results
    """
    # Load model
    clf, scaler = _load_models()

    # Load candidates
    raw = np.load(npy_path, allow_pickle=True)
    if raw.ndim == 0:
        data    = raw.item()
        fluxes  = np.asarray(data['fluxes'],  dtype=np.float32)
        tic_ids = list(data.get('tic_ids',
                  [f'candidate_{i+1:03d}' for i in range(len(fluxes))]))
        periods = np.asarray(data.get('periods',
                  np.zeros(len(fluxes))), dtype=np.float32)
    else:
        fluxes  = raw.astype(np.float32)
        tic_ids = [f'candidate_{i+1:03d}' for i in range(len(fluxes))]
        periods = np.zeros(len(fluxes), dtype=np.float32)

    # Extract features
    X_feat = []
    for flux in fluxes:
        f = flux.copy()
        m = np.median(f)
        if m > 0: f = f / m
        f = np.clip(f, 0.5, 1.5)
        if len(f) < 200:
            f = np.pad(f,(0,200-len(f)),constant_values=1.0)
        X_feat.append(_extract_features(f[:200]))

    X_feat   = np.array(X_feat, dtype=np.float32)
    X_scaled = scaler.transform(X_feat)
    probs    = clf.predict_proba(X_scaled)
    preds    = clf.predict(X_scaled)
    classes  = [CLASS_NAMES_4[p] for p in preds]

    # Parameter estimation for planets
    params_list = []
    for i, cls in enumerate(classes):
        if cls == 'planet' and estimate_params:
            params_list.append(_estimate_transit_params(fluxes[i].copy()))
        else:
            params_list.append({
                'depth_ppm': None, 'depth_ppm_err': None,
                'duration_fraction': None, 'duration_frac_err': None,
                'ingress_fraction': None, 'transit_snr': None,
            })

    # Build DataFrame
    results = pd.DataFrame({
        'TIC_ID'           : tic_ids,
        'predicted_class'  : classes,
        'planet_prob'      : probs[:,0].round(4),
        'binary_prob'      : probs[:,1].round(4),
        'blend_prob'       : probs[:,2].round(4),
        'other_prob'       : probs[:,3].round(4),
        'confidence'       : np.max(probs,axis=1).round(4),
        'period_days'      : periods.round(6),
        'depth_ppm'        : [p['depth_ppm']         for p in params_list],
        'depth_ppm_err'    : [p['depth_ppm_err']      for p in params_list],
        'duration_fraction': [p['duration_fraction']  for p in params_list],
        'duration_frac_err': [p['duration_frac_err']  for p in params_list],
        'ingress_fraction' : [p['ingress_fraction']   for p in params_list],
        'transit_snr'      : [p['transit_snr']        for p in params_list],
    })

    results = results.sort_values(
        ['predicted_class','confidence'],
        ascending=[True,False]
    ).reset_index(drop=True)

    if save_csv:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

    # Print summary
    print("\n══════  Classification Results  ══════")
    print(results['predicted_class'].value_counts().to_string())
    planets = results[results['predicted_class']=='planet']
    print(f"\n🪐 Planet candidates: {len(planets)}")
    if len(planets) > 0:
        print(planets[[
            'TIC_ID','planet_prob','confidence',
            'period_days','depth_ppm','transit_snr'
        ]].to_string(index=False))
    print("══════════════════════════════════════")

    return results
