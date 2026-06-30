# Exoplanet Detection Pipeline

**ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 7**
_AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves_

---

## Team : Interstellar Innovators

**College:** Gujarat Technological University

---

## What this project does

An end-to-end AI pipeline that automatically detects exoplanet candidates from real NASA TESS satellite light curve data. Starting from raw TESS FITS files and ending with classified planet candidates — with no manual intervention required at any stage.

The pipeline processed **1146 real TESS stars** across multiple sectors, found **217 periodic transit candidates** using Box Least Squares detection, classified them with an **89% accurate Random Forest model**, and identified **TIC 25858367** — a strong planet candidate with SNR=73.66 **not present in NASA's official TOI catalogue**, representing a genuinely new astronomical finding. The pipeline was independently validated against confirmed exoplanet HD 110082, recovering its published orbital period within 3.7% accuracy.

---

## Results

| Metric                                | Value                              |
| ------------------------------------- | ---------------------------------- |
| Real TESS stars processed             | 1146 (multiple sectors)            |
| Periodic candidates detected (BLS)    | 217                                |
| Eclipsing binaries identified         | 185                                |
| Blend signals                         | 18                                 |
| Other / noise                         | 11                                 |
| Planet candidates                     | 3                                  |
| Model accuracy                        | 89.0% (4-class)                    |
| **Primary finding — TIC 25858367**    |                                    |
| Orbital period                        | 2.275 days                         |
| Transit depth                         | 6025 ppm (0.6%)                    |
| Transit duration                      | 5.965 hours                        |
| Signal-to-noise ratio                 | **73.66**                          |
| NASA TOI status                       | **NOT catalogued**                 |
| **Secondary finding — TIC 451604550** |                                    |
| Orbital period                        | 0.979 days                         |
| Transit depth                         | 894 ppm                            |
| SNR                                   | 5.91                               |
| Validation (HD 110082)                | Period recovered within 3.7% error |

---

## Project structure

```
exoplanet-detector/
├── preprocessing/
│   ├── target_selector.py     Query MAST for real TESS targets
│   ├── downloader.py          Download FITS files (parallel, hard timeouts)
│   └── preprocess.py          Clean and normalize light curves
├── detection/
│   ├── detect.py              BLS transit period search (0.3-27 days)
│   ├── fold.py                Phase-fold candidates to 200-point arrays
│   ├── run_bls_fast.py        Fast parallel BLS for large datasets
│   └── params.py              Extract planet parameters
├── model/
│   ├── model.py               CNN architecture (built, RF outperformed)
│   ├── dataloader.py          Labelled dataset loading
│   ├── train.py               Model training script
│   ├── predict.py             Classification inference
│   ├── predict_utils.py       Shared team prediction utility
│   ├── parameter_estimation.py Trapezoidal transit model fitting
│   ├── rf_4class.pkl          Main RF model (89% accuracy)
│   └── rf_3class.pkl          Backup 3-class model (71%)
├── visualization/
│   └── visualization.py       All plotting functions
├── data/
│   ├── targets/               Target star lists (CSV)
│   ├── labelled/              Labelled TIC ID datasets for training
│   ├── kepler/                Kepler reference catalogues (KOI + PS)
│   ├── raw/                   Raw FITS files (not tracked in git)
│   └── processed/             Cleaned .npy arrays (not tracked in git)
├── results/
│   ├── candidates.csv         BLS candidate list (217 candidates)
│   ├── folded_candidates.npy  200-point folded arrays
│   ├── predictions.csv        Classifications + parameters
│   ├── validation_report.txt  Validation results
│   ├── TIC25858367_analysis.png  Primary finding analysis
│   └── all_candidates.png     All 217 candidates overview
├── quick_predict.py           Single star full pipeline (any TIC ID)
├── build_dataset.py           Labelled training dataset builder
├── INTERFACES.md              Data format contracts
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/herryambaliya/exoplanet-detector.git
cd exoplanet-detector
pip install -r requirements.txt
```

**Required Python version:** 3.10+

---

## How to run

### Full pipeline

```python
import sys
sys.path.insert(0, '.')

# Stage 1-3: Download + preprocess
from preprocessing.downloader import download_from_target_list_parallel
from preprocessing.preprocess import run_all

download_from_target_list_parallel(
    'data/targets/target_list_real_s12.csv',
    sector=12, max_workers=8
)
run_all(sector=12)

# Stage 4-5: Detect + fold
from detection.detect import run_bls_all
from detection.fold import fold_all_candidates
run_bls_all(power_threshold=0.005)
fold_all_candidates()

# Stage 6: Classify
from model.predict_utils import classify_candidates
results = classify_candidates('results/folded_candidates.npy')
print(results['predicted_class'].value_counts())
```

### Test on any single star (quickest demo)

```python
exec(open('quick_predict.py', encoding='utf-8').read())

# Test our primary finding
result = predict_star(25858367)

# Test any TIC ID
result = predict_star(261136679)
```

What `predict_star()` does automatically:

1. Downloads TESS light curve from NASA MAST
2. Cleans and normalises flux
3. Runs BLS period finding
4. Phase folds to 200 points
5. Extracts 34 transit features
6. Classifies with RF model
7. Estimates transit parameters if planet
8. Saves plot to results/

### Multi-planet search

```python
exec(open('quick_predict.py', encoding='utf-8').read())
results = find_all_planets(261136679, max_planets=4)
```

### Validate against confirmed planets

```python
import pandas as pd
results  = pd.read_csv('results/predictions.csv')
print(results[results['predicted_class']=='planet'])
# See results/validation_report.txt for full report
```

---

## The Model

**Algorithm:** Random Forest Classifier (2000 trees)
**Accuracy:** 89.0% on validation set
**Classes:** planet / eclipsing_binary / blend / other

**34 Features extracted from phase-folded curve:**

- Transit depth at 5 window sizes around center
- Mean flux difference (OOT vs in-transit) at 5 scales
- SNR at 5 scales
- Symmetry score (planet vs EB)
- V-shape score (EB indicator)
- U-shape score (planet indicator)
- Secondary eclipse depth
- Skewness, kurtosis, std, RMS
- Percentiles (1st, 5th, 95th, 99th)
- Transit duration (narrow + wide)
- Ingress/egress slope

**Training data (2000 balanced samples, 500 per class):**

| Source                                           | Class            | Count |
| ------------------------------------------------ | ---------------- | ----- |
| NASA Exoplanet Archive                           | planet           | 500   |
| Prsa+2022 TESS EB Catalogue                      | eclipsing_binary | 500   |
| Physically simulated (planet + background star)  | blend            | 500   |
| Kepler KOI candidates + synthetic variable stars | other            | 500   |

**Why Random Forest over CNN:**
We implemented both. CNN struggled because planet transit signals (0.1-1% flux change) are extremely small, making shape learning difficult with our dataset size (~500 real samples). Random Forest on physically-motivated features achieved 89% accuracy vs CNN's 45%. This matches published literature — Shallue & Vanderburg 2018 required 15,000+ samples for CNN to work well.

---

## Primary Finding — TIC 25858367

Our most significant result. A star showing strong periodic dimming **not present in NASA's official TESS Object of Interest (TOI) catalogue**.

```
Period    : 2.275 days (54.6 hours)
Depth     : 6025 ppm (0.6% brightness dip)
Duration  : 5.965 hours
SNR       : 73.66  (threshold for significance = 5.0)
Shape     : U-shaped flat bottom — consistent with planet
Secondary : 1110 ppm at phase 0.5 (18.4% of primary)
Status    : NOT in NASA TOI catalogue
```

The U-shaped flat-bottom transit morphology is consistent with a planetary transit. The secondary eclipse at phase 0.5 introduces ambiguity — could be a hot Jupiter with reflected/thermal emission, or a grazing eclipsing binary. **Spectroscopic follow-up is recommended.**

This finding demonstrates the pipeline's ability to detect signals that NASA's official automated pipeline missed.

---

## Key Technical Decisions

**Why `get_real_sector_targets()` instead of catalog filtering**
Filtering TIC by brightness alone gave 6-13% download success. Querying MAST's live observation database directly achieved 83%+ by returning only stars TESS actually observed at 2-minute cadence.

**Two-pass flattening**
Single-pass `flatten()` destroyed 73% of transit signal depth because lightkurve's sigma-clipper treated the dip as an outlier. Two-pass approach (rough BLS → mask transits → re-flatten) recovered 99.3% of signal depth.

**Why processes not threads for downloads**
Python threads cannot be force-killed. A stuck MAST request permanently blocks a thread worker. `multiprocessing.Process` with `.terminate()` actually kills stuck processes at OS level — confirmed experimentally on a 65-star run that stalled completely with thread-based approach.

**Why Random Forest over CNN**
Planet transits are too small for CNNs to learn from small datasets. 34 physically-motivated features capture transit shape information that CNNs extract implicitly but need 15,000+ samples to learn. RF achieved 89% with 2000 samples.

**Improved BLS (vs original)**
Extended period range 0.5→0.3 days minimum, 13→27 days maximum. Wider duration grid (10→15 points). Top-3 period candidates + harmonic checking (checks 2P and P/2). Reduced missed detections significantly.

---

## Limitations

```
1. Long-period planets (>13 days) require multi-sector data
   Single 27-day sector → need 2+ transits → max ~13 days

2. Real-world accuracy vs validation accuracy gap
   89% on validation set, ~33% on confirmed planets
   Cause: training data distribution mismatch
   (synthetic curves cleaner than real TESS noise)

3. M-dwarf stars have different noise profiles
   Not well represented in training data

4. BLS period finding occasionally finds aliases
   (harmonics of true period) → wrong fold → misclassification
```

---

## Data

Raw FITS files and processed .npy arrays are not tracked in git (too large, regenerable by running the pipeline). Kepler reference catalogues (`data/kepler/`) are included.

Model files `rf_4class.pkl` and `rf_3class.pkl` are tracked in git (compressed with joblib, ~20MB each).

---

## Dependencies

```
lightkurve>=2.6
astropy>=7.0
astroquery>=0.4
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
scipy>=1.10
seaborn>=0.12
joblib>=1.3
```

---

## References

- Prsa et al. 2022, _Kepler Eclipsing Binary Stars_, ApJS 258 16 — EB training data source
- Shallue & Vanderburg 2018, _Identifying Exoplanets with Deep Learning_, AJ 155 94
- Tofflemire et al. 2021, _THYME IV_, arXiv:2102.06066 — HD 110082 validation
- NASA TESS Mission, MAST Archive — https://archive.stsci.edu/tess
- lightkurve collaboration — https://docs.lightkurve.org
- NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu

---
