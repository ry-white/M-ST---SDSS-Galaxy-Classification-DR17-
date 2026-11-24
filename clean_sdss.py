"""
Clean the Kaggle SDSS DR17 'star_classification.csv' for galaxy / star / QSO work.

The file has NO magnitude errors, NO photometric flags and NO zWarning, so the
S/N cut here is a *proxy* built from the SDSS asinh-magnitude softening
parameters. For a publication-grade cut, pull the real columns from CasJobs
(see the SQL at the bottom of this file) and swap them in.

Examples
--------
    python clean_sdss.py star_classification.csv cleaned.parquet
    python clean_sdss.py star_classification.csv cleaned.csv --sn-min 10 --bands gri
    python clean_sdss.py star_classification.csv features.parquet --ml-features
"""
import argparse

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

# ---------------------------------------------------------------- constants
BANDS = list("ugriz")

# Zero-flux asinh magnitude m0 = -2.5 log10(b), b = softening parameter
# (~1-sigma sky noise in a PSF aperture, 1" seeing). Stoughton et al. 2002.
M0 = {"u": 24.63, "g": 25.11, "r": 24.80, "i": 24.36, "z": 22.83}
POGSON_A = 2.5 / np.log(10)                     # a = 1.0857

# Sane redshift window for each spectroscopic class: columns = lo, hi
Z_LIMITS = pd.DataFrame(
    {"STAR": (-0.005, 0.005),                   # |v| < 1500 km/s
     "GALAXY": (0.002, 1.2),                    # z~0 "galaxies" = bad fits
     "QSO": (0.05, 7.5)},
    index=["lo", "hi"],
).T

# Survey bookkeeping: identifies *how* an object was observed, not *what* it
# is. Leaks targeting information into a classifier, so never use as features.
BOOKKEEPING = ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID",
               "spec_obj_ID", "plate", "MJD", "fiber_ID"]
COLOURS = ["u-g", "g-r", "r-i", "i-z"]


# ---------------------------------------------------------------- helpers
def load(path):
    """Read the catalogue. IDs as str: 19-digit ints overflow float64 precision."""
    return pd.read_csv(path, dtype={"obj_ID": str, "spec_obj_ID": str})


def proxy_sn(mag, band):
    """
    Proxy S/N from an asinh magnitude, using the exact inversion
        x / b = 2 sinh((m0 - m) / a),   x = f/f0,  sigma_x ~ b.
    Returns 0 at m = m0 and negative values for negative measured flux.
    """
    return 2.0 * np.sinh((M0[band] - mag) / POGSON_A)


def add_sn_columns(df):
    """Attach sn_<band> proxy columns for every band (used by cuts + dedup)."""
    df = df.copy()
    for b in BANDS:
        valid = df[b].where(df[b].between(0, 35))     # -9999 etc. -> NaN
        df[f"sn_{b}"] = proxy_sn(valid, b)            # NaN >= x is False, so NaN fails cuts
    return df


def add_colours(df):
    df = df.copy()
    for c in COLOURS:
        a, b = c.split("-")
        df[c] = df[a] - df[b]
    return df


# ---------------------------------------------------------------- cuts
def quality_cuts(df, sn_min, sn_bands):
    """
    Row-wise quality cuts. Returns {name: boolean Series aligned on df.index};
    True = row passes. All masks are pandas Series so they align by index.
    """
    unknown = set(df["class"].unique()) - set(Z_LIMITS.index)
    if unknown:
        raise ValueError(f"No redshift window defined for class(es): {unknown}")

    lims = Z_LIMITS.loc[df["class"]].set_index(df.index)   # per-row lo/hi

    return {
        "valid mags": df[BANDS].gt(0).all(axis=1) & df[BANDS].lt(35).all(axis=1),
        f"S/N>={sn_min:g} in {''.join(sn_bands)}":
            df[[f"sn_{b}" for b in sn_bands]].ge(sn_min).all(axis=1),
        "redshift vs class": df["redshift"].between(lims["lo"], lims["hi"]),
    }


def duplicate_mask(df, radius_arcsec, rank_col):
    """
    Sky crossmatch of the catalogue against itself. Within each matched pair,
    keep the row with the higher `rank_col` (e.g. r-band S/N).
    Returns a boolean Series, True = keep.
    """
    coords = SkyCoord(df["alpha"].to_numpy() * u.deg, df["delta"].to_numpy() * u.deg)
    i, j, _, _ = search_around_sky(coords, coords, radius_arcsec * u.arcsec)

    pairs = i != j                                   # drop self-matches
    i, j = i[pairs], j[pairs]

    rank = df[rank_col].to_numpy()
    # loser of each pair = lower rank; ties broken by row position
    loser = np.where((rank[i] > rank[j]) | ((rank[i] == rank[j]) & (i < j)), j, i)

    keep = np.ones(len(df), dtype=bool)
    keep[np.unique(loser)] = False
    return pd.Series(keep, index=df.index)


# ---------------------------------------------------------------- main
def print_cutflow(df, steps):
    """steps: list of (name, boolean Series). Applied cumulatively."""
    keep = pd.Series(True, index=df.index)
    print(f"{'cut':30s}{'N left':>9s}   per class")
    print(f"{'start':30s}{len(df):9d}   {df['class'].value_counts().to_dict()}")
    for name, mask in steps:
        keep &= mask.reindex(df.index, fill_value=False)
        per = df.loc[keep, "class"].value_counts().to_dict()
        print(f"{name:30s}{keep.sum():9d}   {per}")
    return keep


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="star_classification.csv")
    p.add_argument("output", help="output file; .parquet or .csv")
    p.add_argument("--sn-min", type=float, default=5.0, help="min proxy S/N (default 5)")
    p.add_argument("--bands", default="gri", help="bands the S/N cut applies to (default gri)")
    p.add_argument("--dedup-arcsec", type=float, default=1.0,
                   help="crossmatch radius for duplicates (default 1 arcsec)")
    p.add_argument("--ml-features", action="store_true",
                   help="write only class + colours + r (+ redshift) for ML")
    return p.parse_args()


def main():
    args = parse_args()
    sn_bands = list(args.bands)
    if bad := set(sn_bands) - set(BANDS):
        raise SystemExit(f"Unknown band(s) in --bands: {bad}")

    df = add_sn_columns(load(args.input))

    # 1) row-wise quality cuts
    cuts = quality_cuts(df, args.sn_min, sn_bands)
    passed = pd.Series(True, index=df.index)
    for m in cuts.values():
        passed &= m

    # 2) dedup ONLY among rows that passed, so a good copy is never lost
    #    because its twin failed a cut
    survivors = df[passed]
    dedup = duplicate_mask(survivors, args.dedup_arcsec, rank_col="sn_r")

    steps = list(cuts.items()) + [(f"unique (<{args.dedup_arcsec:g}\" match)", dedup)]
    keep = print_cutflow(df, steps)

    clean = add_colours(df[keep])
    if args.ml_features:
        clean = clean[["class"] + COLOURS + ["r", "redshift"]]

    if args.output.endswith(".parquet"):
        clean.to_parquet(args.output, index=False)
    else:
        clean.to_csv(args.output, index=False)
    print(f"\nwrote {len(clean)} rows -> {args.output}")
    if args.ml_features:
        print("note: 'redshift' is included but is near-label leakage; "
              "drop it for photometry-only classification.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# To get REAL quality columns: upload plate, MJD, fiber_ID to CasJobs MyDB
# (e.g. as table 'kag') and run against DR17:
#
# SELECT k.plate, k.MJD, k.fiber_ID,
#        s.zWarning, s.zErr, s.snMedian_r, s.subClass,
#        p.type, p.clean,
#        p.modelMagErr_u, p.modelMagErr_g, p.modelMagErr_r,
#        p.modelMagErr_i, p.modelMagErr_z,
#        p.extinction_u, p.extinction_g, p.extinction_r,
#        p.extinction_i, p.extinction_z,
#        p.petroMag_r, p.petroR50_r, p.fracDeV_r
# INTO mydb.kag_quality
# FROM mydb.kag AS k
# JOIN SpecObj  AS s ON s.plate = k.plate AND s.mjd = k.MJD AND s.fiberID = k.fiber_ID
# JOIN PhotoObj AS p ON p.objID = s.bestObjID
#
# Then cut: zWarning = 0, clean = 1, modelMagErr < 0.2 (S/N>~5),
# and subtract extinction_* from the magnitudes before forming colours.
# ---------------------------------------------------------------------------
