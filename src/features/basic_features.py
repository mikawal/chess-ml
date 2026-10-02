"""Basic game features for the split datasets.

Reads the raw train/test files written by `split_players --write-games` and adds
the cheap header-derived features. Model features afterwards: result,
termination, half_moves, moves_san, clocks_sec, opening_family, tc_increment,
tc_base.

Usage:
    python -m src.features.basic_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUTDIR = ROOT / "data" / "features"

# 9 sampling bins -> 7 modelling bins. The two bins below 1000 (47 and 22k games
# in the pool) and the two above 2000 are merged to get usable group sizes.
MERGE_BINS = {
    "<800": "<1000",
    "[800, 1000)": "<1000",
    "[2000, 2200)": ">=2000",
    ">=2200": ">=2000",
}

# Lichess names this opening differently from its family; fold it in.
OPENING_ALIASES = {"Robatsch (Modern) Defense": "Modern Defense"}


def add_basic_features(df: pl.DataFrame) -> pl.DataFrame:
    """Merge Elo bins, derive opening_family and the time-control parts."""
    tc = pl.col("time_control").str.extract_groups(r"^(\d+)\+(\d+)$")
    return df.with_columns(
        pl.col("elo_bin").replace(MERGE_BINS),
        pl.col("opening")
        .replace(OPENING_ALIASES)
        .str.split_exact(":", 1).struct.field("field_0")
        .str.split_exact(",", 1).struct.field("field_0")
        .str.strip_chars()
        .alias("opening_family"),
        tc.struct.field("2").cast(pl.Int64).alias("tc_increment"),
        tc.struct.field("1").cast(pl.Int64).alias("tc_base"),
    ).drop("eco", "opening")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        df = add_basic_features(pl.read_parquet(args.splits_dir / f"{name}.parquet"))
        if df.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
            raise SystemExit(f"{name}: nulls after feature engineering")
        df.write_parquet(args.outdir / f"{name}.parquet")
        print(f"wrote {name}.parquet {df.shape}")


if __name__ == "__main__":
    main()
