"""Player-disjoint train/test (or train/val/test) split.

Each player gets one split; a game is kept only if both players share it.
Players are shuffled within rating strata. Expected retention is sum(p_i^2),
71% at 0.827/0.173.

Usage:
    python -m src.splitting.split_players
    python -m src.splitting.split_players --fractions 0.7 0.15 0.15 --write-games   # 3-way
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "sampled" / "sampled_games_enriched.parquet"
OUTDIR = ROOT / "data" / "splits"

SPLIT_NAMES = {2: ("train", "test"), 3: ("train", "val", "test")}


def load_games(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = df.with_columns(
        pl.col("white").str.to_lowercase(),
        pl.col("black").str.to_lowercase(),
    )
    return df.filter(pl.col("white") != pl.col("black"))


def build_player_table(df: pl.DataFrame) -> pl.DataFrame:
    long = pl.concat(
        [
            df.select(
                pl.col("white").alias("player"),
                pl.col("white_elo").alias("elo"),
            ),
            df.select(
                pl.col("black").alias("player"),
                pl.col("black_elo").alias("elo"),
            ),
        ]
    )
    return (
        long.group_by("player")
        .agg(
            pl.col("elo").mean().alias("mean_elo"),
            pl.len().alias("n_games"),
        )
        .sort("player")
    )


def assign_strata(players: pl.DataFrame, n_strata: int) -> pl.DataFrame:
    # quantiles of player mean elo; elo_bin is per game, not per player
    elo = players["mean_elo"].to_numpy()
    edges = np.quantile(elo, np.linspace(0, 1, n_strata + 1)[1:-1])
    stratum = np.digitize(elo, edges)
    return players.with_columns(pl.Series("stratum", stratum))


def assign_splits(
    players: pl.DataFrame, fractions: dict[str, float], seed: int
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    strata = players["stratum"].to_numpy()
    labels = np.empty(players.height, dtype=object)

    names = list(fractions)
    cum = np.cumsum([fractions[k] for k in names])[:-1]

    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        for part, name in zip(np.split(idx, (cum * len(idx)).astype(int)), names):
            labels[part] = name

    return players.with_columns(pl.Series("split", labels))


def apply_split(df: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    lut = players.select("player", "split")
    return (
        df.join(lut, left_on="white", right_on="player")
        .rename({"split": "split_white"})
        .join(lut, left_on="black", right_on="player")
        .rename({"split": "split_black"})
        .with_columns(
            pl.when(pl.col("split_white") == pl.col("split_black"))
            .then(pl.col("split_white"))
            .otherwise(None)
            .alias("split")
        )
        .drop("split_white", "split_black")
    )


def verify(tagged: pl.DataFrame, fractions: dict[str, float]) -> list[str]:
    lines: list[str] = []
    kept = tagged.filter(pl.col("split").is_not_null())

    theoretical = sum(v * v for v in fractions.values())
    lines.append("## Retention\n")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| games in | {tagged.height:,} |")
    lines.append(f"| games kept | {kept.height:,} ({kept.height / tagged.height:.1%}) |")
    lines.append(f"| games dropped | {tagged.height - kept.height:,} |")
    lines.append(f"| predicted retention sum(p^2) | {theoretical:.1%} |")

    lines.append("\n## Split sizes\n")
    lines.append("| split | games | share of kept | players |")
    lines.append("|---|---|---|---|")

    rosters: dict[str, set[str]] = {}
    for name in fractions:
        part = kept.filter(pl.col("split") == name)
        roster = set(part["white"].to_list()) | set(part["black"].to_list())
        rosters[name] = roster
        lines.append(
            f"| {name} | {part.height:,} | {part.height / kept.height:.1%} "
            f"| {len(roster):,} |"
        )

    lines.append("\n## Player disjointness\n")
    clean = True
    names = list(fractions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = rosters[a] & rosters[b]
            status = "PASS" if not shared else f"FAIL ({len(shared):,} shared)"
            clean &= not shared
            lines.append(f"- {a} vs {b}: {status}")
    lines.append(f"\nOverall: {'PASS' if clean else 'FAIL'}")

    lines.append("\n## Rating distribution by split\n")
    lines.append("| split | mean | p10 | p25 | p50 | p75 | p90 |")
    lines.append("|---|---|---|---|---|---|---|")
    samples: dict[str, np.ndarray] = {}
    for name in fractions:
        elo = kept.filter(pl.col("split") == name)["avg_elo"].to_numpy()
        samples[name] = elo
        q = np.percentile(elo, [10, 25, 50, 75, 90])
        lines.append(
            f"| {name} | {elo.mean():.0f} | "
            + " | ".join(f"{x:.0f}" for x in q)
            + " |"
        )

    lines.append("\n## Distribution match against train\n")
    lines.append("| comparison | KS statistic | p value |")
    lines.append("|---|---|---|")
    for name in fractions:
        if name == "train":
            continue
        stat, p = ks_2samp(samples["train"], samples[name])
        lines.append(f"| train vs {name} | {stat:.4f} | {p:.3f} |")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.827, 0.173],
                        help="2 (train test) or 3 (train val test) player fractions")
    parser.add_argument("--n-strata", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-games", action="store_true")
    args = parser.parse_args()

    total = sum(args.fractions)
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"fractions must sum to 1, got {total}")

    if len(args.fractions) not in SPLIT_NAMES:
        raise SystemExit("--fractions takes 2 or 3 values")
    fractions = dict(zip(SPLIT_NAMES[len(args.fractions)], args.fractions))
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_games(args.parquet)
    print(f"games: {df.height:,}")

    players = build_player_table(df)
    players = assign_strata(players, args.n_strata)
    players = assign_splits(players, fractions, args.seed)
    print(f"players: {players.height:,}")

    tagged = apply_split(df, players)
    kept = tagged.filter(pl.col("split").is_not_null())
    print(f"kept: {kept.height:,} ({kept.height / df.height:.1%})")

    players.write_parquet(args.outdir / "player_splits.parquet")
    tagged.select("game_id", "split").write_parquet(
        args.outdir / "game_splits.parquet"
    )

    if args.write_games:
        for name in fractions:
            part = kept.filter(pl.col("split") == name)
            part.write_parquet(args.outdir / f"{name}.parquet")
            print(f"  wrote {name}.parquet ({part.height:,} games)")

    header = [
        "# Split report\n",
        f"- source: `{args.parquet}`",
        f"- player fractions: "
        + ", ".join(f"{k} {v:.3f}" for k, v in fractions.items()),
        f"- strata: {args.n_strata}",
        f"- seed: {args.seed}\n",
    ]
    report = args.outdir / "split_report.md"
    report.write_text("\n".join(header + verify(tagged, fractions)) + "\n")
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()