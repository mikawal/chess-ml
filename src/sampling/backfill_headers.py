#!/usr/bin/env python3
"""
Backfill extra PGN headers into an existing sampled pool.

Reads the game_ids already in the parquet, rescans the .pgn.zst dumps for
exactly those ids, and merges the requested headers back in. No re-sampling:
the pool stays byte-identical apart from the new columns.

Fast because it never builds a game tree. It scans raw bytes, decodes only
the header lines of games it actually wants, and skips everything else after
looking at the Site line.

Usage:
    python backfill_headers.py \
        --pool data/sampled/sampled_games.parquet \
        --input data/raw/lichess_db_standard_rated_2017-{04,05,06}.pgn.zst \
        --output data/sampled/sampled_games_enriched.parquet
"""

import argparse
import io
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import zstandard as zstd

# Header
WANTED = {
    b"White": "white",
    b"Black": "black",
    b"UTCDate": "utc_date",
    b"UTCTime": "utc_time",
    b"ECO": "eco",
    b"Opening": "opening",
}


PREFIXES = sorted(
    ((b"[" + k + b' "', col) for k, col in WANTED.items()),
    key=lambda t: -len(t[0]),
)

SITE_PREFIX = b'[Site "'
EVENT_PREFIX = b"[Event "


def scan_file(path: str, wanted_ids: frozenset) -> list[dict]:
    """Return one dict per matched game. Order follows the file, not the pool."""
    rows: list[dict] = []
    cur: dict | None = None
    n_lines = 0

    with open(path, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        buf = io.BufferedReader(reader, buffer_size=1 << 22)

        for line in buf:
            n_lines += 1
            if not line.startswith(b"["):
                continue

            if line.startswith(EVENT_PREFIX):
                if cur is not None:
                    rows.append(cur)
                    cur = None
                continue

            if line.startswith(SITE_PREFIX):
                gid = line[line.rindex(b"/") + 1 : line.rindex(b'"')].decode("ascii")
                cur = {"game_id": gid} if gid in wanted_ids else None
                continue

            if cur is None:
                continue

            for prefix, col in PREFIXES:
                if line.startswith(prefix):
                    cur[col] = line[len(prefix) : line.rindex(b'"')].decode(
                        "utf-8", "replace"
                    )
                    break

    if cur is not None:
        rows.append(cur)

    print(f"  {Path(path).name}: {n_lines:,} lines, {len(rows):,} matches")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="existing sampled parquet")
    ap.add_argument("--input", required=True, nargs="+", help=".pgn.zst dumps")
    ap.add_argument("--output", required=True, help="enriched parquet")
    ap.add_argument("--workers", type=int, default=0, help="0 = one per input file")
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    wanted = frozenset(pool["game_id"])
    print(f"pool: {len(pool):,} games, {len(wanted):,} unique ids")

    workers = args.workers or len(args.input)
    t0 = time.time()

    if workers == 1:
        results = [scan_file(p, wanted) for p in args.input]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(scan_file, args.input, [wanted] * len(args.input)))

    rows = [r for chunk in results for r in chunk]
    print(f"scan: {time.time() - t0:.1f}s, {len(rows):,} matches total")

    extra = pd.DataFrame(rows)
    for col in WANTED.values():
        if col not in extra.columns:
            extra[col] = pd.NA
    extra = extra[["game_id"] + list(WANTED.values())]

    dupes = extra["game_id"].duplicated().sum()
    if dupes:
        raise SystemExit(f"{dupes} game_ids matched in more than one file")

    missing = len(pool) - len(extra)
    if missing:
        raise SystemExit(f"{missing} games from the pool were not found in the dumps")

    out = pool.merge(extra, on="game_id", how="left", validate="one_to_one")
    assert len(out) == len(pool)
    for col in WANTED.values():
        n_null = out[col].isna().sum()
        print(f"  {col}: {n_null} missing")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"written: {args.output}  {out.shape}")


if __name__ == "__main__":
    main()