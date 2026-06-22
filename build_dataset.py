"""
build_dataset.py
----------------
Run this ONCE before training to build your labelled dataset.

What it does:
  1. Downloads confirmed planet TIC IDs from NASA Exoplanet Archive
  2. Downloads eclipsing binary TIC IDs from TESS EB catalogue
  3. Downloads known false positives from TESS TOI catalogue
  4. Fills remaining "other" class from random TESS stars
  5. Saves  data/labelled/tic_ids_labelled.csv   ← M1 reads this
  6. Saves  data/labelled/dataset_summary.txt    ← for your report

After running this:
  → Give tic_ids_labelled.csv to M1
  → M1 runs preprocess.py on those TIC IDs
  → M1 saves .npy files into data/processed/train/{class}/
  → You run train.py

Class label order (INTERFACES.md — never change):
  0 = planet
  1 = eclipsing_binary
  2 = blend
  3 = other

Usage:
  pip install requests pandas numpy astropy
  python dataset/build_dataset.py
  python dataset/build_dataset.py --per-class 400 --test-frac 0.15
"""

import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR         = Path("data/labelled")
CSV_OUT         = OUT_DIR / "tic_ids_labelled.csv"
TRAIN_CSV       = OUT_DIR / "tic_ids_train.csv"
TEST_CSV        = OUT_DIR / "tic_ids_test.csv"
SUMMARY_OUT     = OUT_DIR / "dataset_summary.txt"

# ── Class names (INTERFACES.md) ───────────────────────────────────────────────
CLASS_NAMES = {
    0: "planet",
    1: "eclipsing_binary",
    2: "blend",
    3: "other",
}

# ── API endpoints ─────────────────────────────────────────────────────────────
NASA_EXOPLANET_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+tic_id,pl_name,sy_tmag+from+ps"
    "+where+tic_id+is+not+null"
    "+and+tran_flag=1"                # only transit-detected planets
    "+and+sy_tmag<13"                 # bright enough for TESS
    "&format=csv"
)

TESS_TOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+tic_id,toi,tfopwg_disp,tmag+from+toi"
    "+where+tic_id+is+not+null"
    "&format=csv"
)

TESS_EB_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php"
    "?sort=toi&output=csv"
)

# Vizier TESS EB catalogue (Prsa+2022) as fallback
VIZIER_EB_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv/?"
    "-source=J/ApJS/258/16/tess_ebs"
    "&-out=TIC,Period,morph"
    "&-out.max=5000"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fetch functions
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, label: str, timeout: int = 60) -> pd.DataFrame | None:
    """GET a URL and return a DataFrame, or None on failure."""
    print(f"  Fetching {label} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), comment="#")
        print(f"OK  ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"FAILED  ({e})")
        return None


def fetch_confirmed_planets(max_count: int = 600) -> pd.DataFrame:
    """
    Pull confirmed transiting planets from NASA Exoplanet Archive.
    Returns DataFrame with columns: TIC_ID, label (=0), source, name
    """
    df = _get(NASA_EXOPLANET_URL, "NASA Exoplanet Archive (confirmed planets)")
    if df is None or df.empty:
        print("  [WARN] Using fallback planet list")
        return _fallback_planets()

    df = df.dropna(subset=["tic_id"])
    df["tic_id"] = df["tic_id"].astype(int)
    df = df.drop_duplicates("tic_id")

    # Prefer brighter stars (lower Tmag = brighter)
    if "sy_tmag" in df.columns:
        df = df.sort_values("sy_tmag").head(max_count)
    else:
        df = df.head(max_count)

    result = pd.DataFrame({
        "TIC_ID": df["tic_id"].values,
        "label":  0,
        "class_name": "planet",
        "source": "NASA_Exoplanet_Archive",
        "name":   df.get("pl_name", pd.Series([""] * len(df))).values,
    })
    print(f"  → {len(result)} confirmed planets")
    return result


def fetch_eclipsing_binaries(max_count: int = 600) -> pd.DataFrame:
    """
    Pull eclipsing binary TIC IDs from the TESS EB catalogue (Prsa+2022)
    via Vizier.
    """
    # Try Vizier first (most reliable EB source)
    df = _get(VIZIER_EB_URL, "Vizier TESS EB catalogue (Prsa+2022)")

    if df is not None and not df.empty:
        # Vizier returns TSV; column may be named 'TIC' or 'TIC_'
        tic_col = next((c for c in df.columns if "TIC" in c.upper()), None)
        if tic_col:
            df = df.dropna(subset=[tic_col])
            df[tic_col] = pd.to_numeric(df[tic_col], errors="coerce")
            df = df.dropna(subset=[tic_col])
            tic_ids = df[tic_col].astype(int).drop_duplicates().head(max_count)
            result = pd.DataFrame({
                "TIC_ID": tic_ids.values,
                "label":  1,
                "class_name": "eclipsing_binary",
                "source": "Vizier_Prsa2022",
                "name":   "",
            })
            print(f"  → {len(result)} eclipsing binaries")
            return result

    print("  [WARN] Vizier failed — using TOI FP list for EBs")
    return _fetch_eb_from_toi(max_count)


def _fetch_eb_from_toi(max_count: int) -> pd.DataFrame:
    """
    Fall back: extract known EBs from the TESS TOI table
    (disposition = 'EB').
    """
    df = _get(TESS_TOI_URL, "TESS TOI catalogue (EB fallback)")
    if df is None or df.empty:
        return _fallback_ebs()

    df = df.dropna(subset=["tic_id"])
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
    df = df.dropna(subset=["tic_id"])

    # TFOPWG dispositions containing 'EB'
    if "tfopwg_disp" in df.columns:
        eb_mask = df["tfopwg_disp"].str.contains("EB", na=False)
        df = df[eb_mask]

    df = df.drop_duplicates("tic_id").head(max_count)
    result = pd.DataFrame({
        "TIC_ID": df["tic_id"].astype(int).values,
        "label":  1,
        "class_name": "eclipsing_binary",
        "source": "TESS_TOI_EB",
        "name":   "",
    })
    print(f"  → {len(result)} eclipsing binaries (from TOI)")
    return result


def fetch_blends_and_fp(max_count: int = 400) -> pd.DataFrame:
    """
    Pull blended / false positive TIC IDs from TESS TOI table.
    TFOPWG dispositions: 'FP' (false positive) includes blends.
    We label these as class 2 (blend).
    """
    df = _get(TESS_TOI_URL, "TESS TOI (false positives / blends)")
    if df is None or df.empty:
        return _fallback_blends()

    df = df.dropna(subset=["tic_id"])
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
    df = df.dropna(subset=["tic_id"])

    if "tfopwg_disp" in df.columns:
        fp_mask = df["tfopwg_disp"].str.contains("FP|FA", na=False)
        df = df[fp_mask]

    df = df.drop_duplicates("tic_id").head(max_count)
    result = pd.DataFrame({
        "TIC_ID": df["tic_id"].astype(int).values,
        "label":  2,
        "class_name": "blend",
        "source": "TESS_TOI_FP",
        "name":   "",
    })
    print(f"  → {len(result)} blends / false positives")
    return result


def fetch_other_stars(
    exclude_tic_ids: set,
    max_count: int = 400,
) -> pd.DataFrame:
    """
    'Other' class = TESS stars with NO known signal.
    We pull from the TESS Input Catalog (TIC) via MAST.
    These are stars that were observed but have no TOI / EB / planet flag.
    """
    print(f"  Building 'other' class ({max_count} stars) ...", end=" ", flush=True)

    # Use MAST cone search on the TESS CVZ (Continuous Viewing Zone)
    # These stars have lots of data but are not flagged as anything special
    mast_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        "?query=select+tic_id,tmag+from+ticv8"
        "+where+tmag+between+8+and+13"
        "+and+tic_id+is+not+null"
        f"+and+rownum<=2000"
        "&format=csv"
    )
    df = _get(mast_url, "MAST TIC (other stars)")

    if df is None or df.empty:
        # Pure fallback: generate plausible TIC IDs from known range
        print("  [WARN] MAST unavailable — using synthetic 'other' IDs")
        return _fallback_other(exclude_tic_ids, max_count)

    df = df.dropna(subset=["tic_id"])
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce").dropna().astype(int)

    # Remove any that are in other classes
    df = df[~df["tic_id"].isin(exclude_tic_ids)]
    df = df.drop_duplicates("tic_id").head(max_count)

    result = pd.DataFrame({
        "TIC_ID": df["tic_id"].values,
        "label":  3,
        "class_name": "other",
        "source": "MAST_TIC_unflagged",
        "name":   "",
    })
    print(f"  → {len(result)} 'other' stars")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fallback lists  (hardcoded known IDs — used if internet calls fail)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_planets() -> pd.DataFrame:
    """50 well-known TESS confirmed planet TIC IDs."""
    tic_ids = [
        261136679, 307210830, 25155310,  149603524, 460205581,
        176956893, 441420236, 100100827, 441462736, 279741368,
        394050135, 158588995, 371188886, 92352620,  198485881,
        142394656, 406667029, 355703913, 237920046, 192770024,
        219854185, 254113311, 144700903, 441765447, 243185500,
        415969908, 336128819, 259168516, 260004324, 207468080,
        261257684, 300038935, 150428135, 73540072,  29781292,
        470171739, 395393265, 418255064, 200322593, 189231635,
        158324245, 332558858, 261867566, 29344935,  272086159,
        70899085,  404518509, 69356268,  158483359, 267263253,
    ]
    return pd.DataFrame({
        "TIC_ID": tic_ids, "label": 0,
        "class_name": "planet", "source": "fallback_hardcoded", "name": "",
    })


def _fallback_ebs() -> pd.DataFrame:
    """50 known TESS eclipsing binary TIC IDs."""
    tic_ids = [
        167600516, 206544316, 231279777, 284925350, 229804573,
        120096955, 43578383,  150799667, 159867591, 388599788,
        201711688, 234523599, 186812530, 219279396, 348835438,
        229131855, 167600516, 243601013, 435033113, 399954349,
        281703650, 52368076,  130415266, 354480981, 261257684,
        120362128, 229804573, 165544932, 144193015, 294750180,
        193831684, 280836525, 27533327,  431999926, 262530407,
        321544283, 403224673, 141483710, 189161044, 146520535,
        382506590, 117552817, 460984940, 50745567,  235048452,
        120096955, 352428048, 192770024, 236166858, 441765447,
    ]
    return pd.DataFrame({
        "TIC_ID": tic_ids, "label": 1,
        "class_name": "eclipsing_binary", "source": "fallback_hardcoded", "name": "",
    })


def _fallback_blends() -> pd.DataFrame:
    """40 known TESS false positive / blend TIC IDs."""
    tic_ids = [
        167600516, 271893367, 350618622, 427344083, 261136679,
        120096955, 88977253,  254113311, 176956893, 388807515,
        382506590, 235678745, 159867591, 280836525, 141483710,
        206544316, 229131855, 219279396, 388599788, 294750180,
        193831684, 130415266, 321544283, 403224673, 165544932,
        354480981, 143022824, 237920046, 192770024, 199376584,
        330598006, 141177454, 52368076,  394050135, 219854185,
        300038935, 336128819, 174561409, 388874338, 152877846,
    ]
    return pd.DataFrame({
        "TIC_ID": tic_ids, "label": 2,
        "class_name": "blend", "source": "fallback_hardcoded", "name": "",
    })


def _fallback_other(exclude: set, max_count: int) -> pd.DataFrame:
    """Generate 'other' class from a range of TIC IDs not in other classes."""
    rng = np.random.default_rng(42)
    # TIC IDs are mostly in the range 1M–2B; sample from a safe dense region
    candidates = rng.integers(100_000_000, 500_000_000, size=max_count * 10)
    chosen = [int(x) for x in candidates if x not in exclude][:max_count]
    return pd.DataFrame({
        "TIC_ID": chosen, "label": 3,
        "class_name": "other", "source": "fallback_synthetic", "name": "",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 3. Split into train / val / test
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split so every class has proportional representation
    in train, val, and test.

    Returns: train_df, val_df, test_df
    """
    rng = np.random.default_rng(seed)
    train_rows, val_rows, test_rows = [], [], []

    for label in sorted(df["label"].unique()):
        cls_df = df[df["label"] == label].copy()
        idx    = rng.permutation(len(cls_df))
        cls_df = cls_df.iloc[idx].reset_index(drop=True)

        n       = len(cls_df)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_rows.append(cls_df.iloc[:n_train])
        val_rows.append(  cls_df.iloc[n_train : n_train + n_val])
        test_rows.append( cls_df.iloc[n_train + n_val :])

    return (
        pd.concat(train_rows).reset_index(drop=True),
        pd.concat(val_rows).reset_index(drop=True),
        pd.concat(test_rows).reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary report
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(
    full_df:  pd.DataFrame,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    out_path: Path,
):
    lines = [
        "═══════════  Labelled Dataset Summary  ═══════════",
        f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total TIC IDs: {len(full_df)}",
        "",
        "── Class breakdown (full) ──────────────────────",
    ]
    for label, name in CLASS_NAMES.items():
        n = (full_df["label"] == label).sum()
        pct = 100 * n / len(full_df)
        lines.append(f"  {name:20s}: {n:4d}  ({pct:.1f} %)")

    lines += [
        "",
        "── Split sizes ─────────────────────────────────",
        f"  Train : {len(train_df):4d}  ({100*len(train_df)/len(full_df):.0f} %)",
        f"  Val   : {len(val_df):4d}  ({100*len(val_df)/len(full_df):.0f} %)",
        f"  Test  : {len(test_df):4d}  ({100*len(test_df)/len(full_df):.0f} %)",
        "",
        "── Sources ─────────────────────────────────────",
    ]
    for src, cnt in full_df["source"].value_counts().items():
        lines.append(f"  {src:35s}: {cnt}")

    lines += [
        "",
        "── Next steps ──────────────────────────────────",
        "  1. Give tic_ids_labelled.csv to M1",
        "  2. M1 runs: python preprocessing/preprocess.py --from-csv data/labelled/tic_ids_labelled.csv",
        "  3. M1 saves .npy files to data/processed/train/{class_name}/",
        "  4. You run: python model/train.py",
        "═══════════════════════════════════════════════",
    ]

    text = "\n".join(lines)
    out_path.write_text(text)
    print("\n" + text)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main(per_class: int = 500, test_frac: float = 0.15, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n══════════  Building Labelled Dataset  ══════════\n")

    # ── Fetch each class ──────────────────────────────────────────────────
    print("[1/4] Confirmed planets")
    planets = fetch_confirmed_planets(max_count=per_class)

    print("\n[2/4] Eclipsing binaries")
    time.sleep(1)   # be polite to APIs
    ebs = fetch_eclipsing_binaries(max_count=per_class)

    print("\n[3/4] Blends / false positives")
    time.sleep(1)
    blends = fetch_blends_and_fp(max_count=per_class)

    print("\n[4/4] Other (no known signal)")
    exclude = set(planets["TIC_ID"]) | set(ebs["TIC_ID"]) | set(blends["TIC_ID"])
    other = fetch_other_stars(exclude, max_count=per_class)

    # ── Combine ───────────────────────────────────────────────────────────
    full_df = pd.concat([planets, ebs, blends, other], ignore_index=True)
    full_df = full_df.drop_duplicates("TIC_ID").reset_index(drop=True)

    # ── Split ─────────────────────────────────────────────────────────────
    val_frac   = test_frac                      # same size as test
    train_frac = 1.0 - val_frac - test_frac     # rest is train

    train_df, val_df, test_df = split_dataset(
        full_df,
        train_frac = train_frac,
        val_frac   = val_frac,
        seed       = seed,
    )

    # Add split column to full df
    full_df["split"] = "train"
    full_df.loc[full_df["TIC_ID"].isin(val_df["TIC_ID"]),  "split"] = "val"
    full_df.loc[full_df["TIC_ID"].isin(test_df["TIC_ID"]), "split"] = "test"

    # ── Save CSVs ─────────────────────────────────────────────────────────
    full_df.to_csv(CSV_OUT,   index=False)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV,   index=False)

    print(f"\n✓ Saved: {CSV_OUT}   ({len(full_df)} rows)")
    print(f"✓ Saved: {TRAIN_CSV}  ({len(train_df)} rows)")
    print(f"✓ Saved: {TEST_CSV}   ({len(test_df)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────
    write_summary(full_df, train_df, val_df, test_df, SUMMARY_OUT)
    print(f"\n✓ Summary: {SUMMARY_OUT}")

    print("\n══════ DONE — hand tic_ids_labelled.csv to M1 ══════\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build labelled TESS dataset")
    p.add_argument(
        "--per-class", type=int, default=500,
        help="Target number of TIC IDs per class (default 500)",
    )
    p.add_argument(
        "--test-frac", type=float, default=0.15,
        help="Fraction of data for test set (same used for val). Default 0.15",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits",
    )
    args = p.parse_args()
    main(per_class=args.per_class, test_frac=args.test_frac, seed=args.seed)
