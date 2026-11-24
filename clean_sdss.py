"""
Clean the Kaggle SDSS DR17 'star_classification.csv' into a galaxy sample for
red / green-valley / blue classification in the u-r vs M_r plane.

Uses ONLY the Kaggle columns. Consequences (state these in the write-up):
  * S/N is a proxy from the SDSS asinh-magnitude softening parameters, not
    real magnitude errors.
  * No Galactic-extinction correction (no dust map / extinction column).
  * No k-correction here; M_r is distance modulus only. The redshift
    dependence of colours is absorbed later, in the labelling model.

Default sample: GALAXY, 0.02 < z < 0.15, r <= 17.77 (SDSS Legacy main-galaxy
limit), proxy S/N >= 5 in u, g, r.

Examples
--------
    python clean_sdss.py star_classification.csv galaxies.parquet
    python clean_sdss.py star_classification.csv galaxies.parquet --zmax 0.1
    python clean_sdss.py star_classification.csv features.parquet --ml-features
"""
import argparse

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.cosmology import FlatLambdaCDM

# ---------------------------------------------------------------- constants
BANDS = list("ugriz")

# Zero-flux asinh magnitude m0 = -2.5 log10(b), b = softening parameter
# (~1-sigma sky noise in a PSF aperture, 1" seeing). Stoughton et al. 2002.
M0 = {"u": 24.63, "g": 25.11, "r": 24.80, "i": 24.36, "z": 22.83}
POGSON_A = 2.5 / np.log(10)                     # a = 1.0857

# A cosmology is a model choice, not external data. State it in the write-up.
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

ID_COLS = ["plate", "MJD", "fiber_ID"]           # unique key; never a feature
COLOURS = ["u-g", "g-r", "r-i", "i-z", "u-r"]


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
    """Attach sn_<band> proxy columns for every band."""
    df = df.copy()
    for b in BANDS:
        valid = df[b].where(df[b].between(0, 35))     # -9999 etc. -> NaN
        df[f"sn_{b}"] = proxy_sn(valid, b)            # NaN fails every >= cut
    return df


def add_derived(df):
    """Colours and absolute magnitude M_r = r - DM(z) (no k-correction)."""
    df = df.copy()
    for c in COLOURS:
        a, b = c.split("-")
        df[c] = df[a] - df[b]
    df["M_r"] = df["r"] - COSMO.distmod(df["redshift"].to_numpy()).value
    return df


# ---------------------------------------------------------------- cuts
def quality_cuts(df, args):
    """
    Row-wise cuts, applied in the order listed (matters only for the cut-flow
    printout). Returns {name: boolean Series aligned on df.index}.
    """
    sn_cols = [f"sn_{b}" for b in args.bands]
    return {
        "class == GALAXY": df["class"].eq("GALAXY"),
        "valid mags": df[BANDS].gt(0).all(axis=1) & df[BANDS].lt(35).all(axis=1),
        f"{args.zmin:g} < z < {args.zmax:g}":
            df["redshift"].gt(args.zmin) & df["redshift"].lt(args.zmax),
        f"r <= {args.rmax:g}": df["r"].le(args.rmax),
        f"S/N>={args.sn_min:g} in {''.join(args.bands)}":
            df[sn_cols].ge(args.sn_min).all(axis=1),
    }


def duplicate_mask(df, radius_arcsec, rank_col):
    """
    Sky crossmatch of the catalogue against itself. Within each matched pair,
    keep the row with the higher `rank_col`. Returns boolean Series, True = keep.
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
    print(f"{'cut':30s}{'N left':>9s}")
    print(f"{'start':30s}{len(df):9d}")
    for name, mask in steps:
        keep &= mask.reindex(df.index, fill_value=False)
        print(f"{name:30s}{keep.sum():9d}")
    return keep


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="star_classification.csv")
    p.add_argument("output", help="output file; .parquet or .csv")
    p.add_argument("--zmin", type=float, default=0.02,
                   help="min redshift; below ~0.02 peculiar velocities bias distances (default 0.02)")
    p.add_argument("--zmax", type=float, default=0.15, help="max redshift (default 0.15)")
    p.add_argument("--rmax", type=float, default=17.77,
                   help="faint r limit; 17.77 = SDSS main-galaxy limit (default)")
    p.add_argument("--sn-min", type=float, default=5.0, help="min proxy S/N (default 5)")
    p.add_argument("--bands", default="ugr", help="bands the S/N cut applies to (default ugr)")
    p.add_argument("--dedup-arcsec", type=float, default=1.0,
                   help="crossmatch radius for duplicates (default 1 arcsec)")
    p.add_argument("--ml-features", action="store_true",
                   help="write only IDs + redshift + r + M_r + colours")
    args = p.parse_args()

    args.bands = list(args.bands)
    if bad := set(args.bands) - set(BANDS):
        p.error(f"unknown band(s) in --bands: {bad}")
    if not args.zmin < args.zmax:
        p.error("--zmin must be < --zmax")
    return args


def main():
    args = parse_args()
    df = add_sn_columns(load(args.input))

    # 1) row-wise quality cuts
    cuts = quality_cuts(df, args)
    passed = pd.Series(True, index=df.index)
    for m in cuts.values():
        passed &= m

    # 2) dedup ONLY among rows that passed, keeping the higher r-band S/N copy
    dedup = duplicate_mask(df[passed], args.dedup_arcsec, rank_col="sn_r")

    steps = list(cuts.items()) + [(f"unique (<{args.dedup_arcsec:g}\" match)", dedup)]
    keep = print_cutflow(df, steps)

    clean = add_derived(df[keep])
    if args.ml_features:
        clean = clean[ID_COLS + ["redshift", "r", "M_r"] + COLOURS]

    if args.output.endswith(".parquet"):
        clean.to_parquet(args.output, index=False)
    else:
        clean.to_csv(args.output, index=False)
    print(f"\nwrote {len(clean)} galaxies -> {args.output}")


if __name__ == "__main__":
    main()
