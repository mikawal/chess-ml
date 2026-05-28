#!/usr/bin/env python3
"""
Lichess PGN sampling pipeline - stratified reservoir version.

Streams .pgn.zst dumps, applies filters, and performs per-bin reservoir
sampling to produce a roughly balanced pool of games for downstream
analysis (PGN-feature extraction + selective Stockfish evaluation).

Filters (unchanged from previous version):
  - Rated only
  - Rapid only (Lichess formula: 480 <= base + 40*inc <= 1499)
  - No BOT players
  - No Abandoned / Rules infraction
  - Rating difference <= ±max_elo_diff (default 100)
  - Minimum half-moves (default 20)
  - Both Elo headers present and parseable
  - NO eval filter (all games kept regardless of Lichess eval annotations)

Sampling:
  - 9 Elo bins keyed off avg_elo:
      <800, [800,1000), [1000,1200), [1200,1400), [1400,1600),
      [1600,1800), [1800,2000), [2000,2200), >=2200
  - Vitter's Algorithm R reservoir sampler per bin, capped at
    --per_bin_target (default 50000). Bins with fewer eligible
    games than the cap simply retain everything seen.
  - All input files are scanned in full (no early break).
"""

import argparse
import bisect
import io
import random
import re
import sys
from collections import Counter
from pathlib import Path

import chess.pgn
import pandas as pd
import zstandard as zstd
from tqdm import tqdm


RAPID_PATTERN = re.compile(r"^(\d+)\+(\d+)$")
CLK_PATTERN = re.compile(r"\[%clk\s+(\d+:\d+:\d+)\]")

# Bin edges and labels. BIN_EDGES has length N; BIN_NAMES has length N+1.
# bisect_right(BIN_EDGES, x) returns an index in [0, N] that maps directly
# into BIN_NAMES, giving "<800" for x<800 and ">=2200" for x>=2200.
BIN_EDGES = [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200]
BIN_NAMES = (
    ["<800"]
    + [f"[{lo}, {hi})" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    + [">=2200"]
)


def get_elo_bin(avg_elo: float) -> str:
    """Map avg_elo to a bin label using bisect_right semantics."""
    idx = bisect.bisect_right(BIN_EDGES, avg_elo)
    return BIN_NAMES[idx]


class ReservoirSampler:
    """
    Vitter's Algorithm R: yields a uniform random sample of size k
    from a stream of unknown length, in a single pass, using O(k) memory.

    For each incoming item (1-indexed count i):
      - If i <= k: append to buffer.
      - If i  > k: draw j in [0, i-1]; if j < k, replace buffer[j] with item.
    """
    def __init__(self, k: int, rng: random.Random):
        self.k = k
        self.buffer = []
        self.count = 0  # number of items seen (not buffered)
        self.rng = rng

    def add(self, item) -> None:
        self.count += 1
        if len(self.buffer) < self.k:
            self.buffer.append(item)
        else:
            j = self.rng.randint(0, self.count - 1)
            if j < self.k:
                self.buffer[j] = item

    def __len__(self) -> int:
        return len(self.buffer)


def is_rapid(time_control: str) -> bool:
    """Check if time control qualifies as Rapid per Lichess definition."""
    if not time_control:
        return False
    match = RAPID_PATTERN.match(time_control)
    if not match:
        return False
    base = int(match.group(1))
    inc = int(match.group(2))
    estimated = base + 40 * inc
    return 480 <= estimated <= 1499


def extract_game_id(site_header: str) -> str:
    """Extract Lichess game ID from Site header."""
    if site_header and "/" in site_header:
        return site_header.rstrip("/").split("/")[-1]
    return ""


def extract_moves_and_clocks(game: chess.pgn.Game):
    """
    Extract SAN move list and clock values from the main line.
    Returns (moves_san: list[str], clocks_sec: list[int|None]).
    clocks_sec contains remaining time in seconds after each half-move,
    or None if no clock annotation was present.
    """
    moves = []
    clocks = []
    node = game
    while node.variations:
        node = node.variations[0]
        moves.append(node.san())
        clk_match = CLK_PATTERN.search(node.comment) if node.comment else None
        if clk_match:
            parts = clk_match.group(1).split(":")
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            clocks.append(seconds)
        else:
            clocks.append(None)
    return moves, clocks


def stream_games_from_zst(pgn_zst_path: Path):
    """Yield chess.pgn.Game objects from a .pgn.zst file without full decompression."""
    with open(pgn_zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text_io = io.TextIOWrapper(stream_reader, encoding="utf-8")
        while True:
            game = chess.pgn.read_game(text_io)
            if game is None:
                break
            yield game


def main():
    parser = argparse.ArgumentParser(
        description="Stratified reservoir sampling of Lichess Rapid games."
    )
    parser.add_argument("--input", required=True, nargs="+",
                        help="Path(s) to .pgn.zst file(s)")
    parser.add_argument("--per_bin_target", type=int, default=50000,
                        help="Reservoir cap per Elo bin (default 50000)")
    parser.add_argument("--max_elo_diff", type=int, default=100,
                        help="Maximum |WhiteElo - BlackElo| (default 100)")
    parser.add_argument("--min_half_moves", type=int, default=20,
                        help="Minimum half-moves per game (default 20)")
    parser.add_argument("--output_dir", default="data/sampled",
                        help="Directory for outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reservoir sampling (default 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    samplers = {name: ReservoirSampler(args.per_bin_target, rng) for name in BIN_NAMES}

    filter_stats = Counter()
    
    # pre sampling
    bin_seen = Counter()  

    print(f"Configuration:")
    print(f"  Inputs: {len(args.input)} file(s)")
    print(f"  Per-bin target: {args.per_bin_target}")
    print(f"  Max |Elo diff|: {args.max_elo_diff}")
    print(f"  Min half-moves: {args.min_half_moves}")
    print(f"  Seed: {args.seed}")
    print(f"  Bins: {BIN_NAMES}")

    for input_path in args.input:
        print(f"\nStreaming: {input_path}")
        pbar = tqdm(desc="Scanning games", unit="games")

        for game in stream_games_from_zst(Path(input_path)):
            filter_stats["total_read"] += 1
            pbar.update(1)

            # 1. Rated?
            if "Rated" not in game.headers.get("Event", ""):
                filter_stats["reject_not_rated"] += 1
                continue

            # 2. Rapid?
            if not is_rapid(game.headers.get("TimeControl", "")):
                filter_stats["reject_not_rapid"] += 1
                continue

            # 3. No bots
            if (game.headers.get("WhiteTitle") == "BOT"
                    or game.headers.get("BlackTitle") == "BOT"):
                filter_stats["reject_bot"] += 1
                continue

            # 4. No abandoned / rules infraction
            termination = game.headers.get("Termination", "")
            if termination in ("Abandoned", "Rules infraction"):
                filter_stats["reject_termination"] += 1
                continue

            # 5. Elo headers parseable and within diff threshold
            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
            except (ValueError, TypeError):
                filter_stats["reject_missing_elo"] += 1
                continue
            if white_elo == 0 or black_elo == 0:
                filter_stats["reject_missing_elo"] += 1
                continue
            if abs(white_elo - black_elo) > args.max_elo_diff:
                filter_stats["reject_elo_diff"] += 1
                continue

            # 6. Extract moves + clocks, check minimum length
            moves, clocks = extract_moves_and_clocks(game)
            if len(moves) < args.min_half_moves:
                filter_stats["reject_too_short"] += 1
                continue

            # All filters passed: bin and feed to reservoir
            avg_elo = (white_elo + black_elo) / 2
            bin_name = get_elo_bin(avg_elo)
            bin_seen[bin_name] += 1
            filter_stats["passed_filters"] += 1

            game_id = extract_game_id(game.headers.get("Site", ""))
            game_dict = {
                "game_id": game_id,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "avg_elo": avg_elo,
                "rating_diff": white_elo - black_elo,
                "time_control": game.headers.get("TimeControl", ""),
                "result": game.headers.get("Result", ""),
                "termination": termination,
                "half_moves": len(moves),
                "moves_san": ",".join(moves),
                "clocks_sec": ",".join(str(c) if c is not None else "" for c in clocks),
                "elo_bin": bin_name,
            }
            samplers[bin_name].add(game_dict)

            # Periodic progress update on the tqdm bar
            if filter_stats["total_read"] % 50000 == 0:
                total_kept = sum(len(s) for s in samplers.values())
                min_bin = min(BIN_NAMES, key=lambda n: len(samplers[n]))
                pbar.set_postfix({
                    "kept": total_kept,
                    "min": f"{min_bin}={len(samplers[min_bin])}",
                })

        pbar.close()
        print(f"  After {Path(input_path).name}:")
        for name in BIN_NAMES:
            kept = len(samplers[name])
            seen = bin_seen[name]
            print(f"    {name:>14}: kept={kept:>6}  seen={seen:>8}")

    # Concatenate kept games across bins
    kept_games = []
    for name in BIN_NAMES:
        kept_games.extend(samplers[name].buffer)

    if not kept_games:
        print("\nNo games survived all filters.")
        sys.exit(1)

    df = pd.DataFrame(kept_games)
    parquet_path = output_dir / "sampled_games.parquet"
    df.to_parquet(parquet_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"SAMPLING COMPLETE")
    print(f"{'='*60}")
    print(f"Games kept (across all bins): {len(df)}")

    print(f"\nFilter statistics:")
    keys_in_order = [
        "total_read", "reject_not_rated", "reject_not_rapid", "reject_bot",
        "reject_termination", "reject_missing_elo", "reject_elo_diff",
        "reject_too_short", "passed_filters",
    ]
    for k in keys_in_order:
        if filter_stats[k]:
            print(f"  {k}: {filter_stats[k]}")

    print(f"\nPer-bin counts (kept / total eligible seen):")
    for name in BIN_NAMES:
        kept = len(samplers[name])
        seen = bin_seen[name]
        ratio = (kept / seen * 100) if seen else 0.0
        print(f"  {name:>14}: {kept:>6} / {seen:>8}  ({ratio:5.2f}% retained)")

    print(f"\nRating distribution (avg_elo):")
    print(f"  Min:    {df['avg_elo'].min():.0f}")
    print(f"  Q1:     {df['avg_elo'].quantile(0.25):.0f}")
    print(f"  Median: {df['avg_elo'].median():.0f}")
    print(f"  Q3:     {df['avg_elo'].quantile(0.75):.0f}")
    print(f"  Max:    {df['avg_elo'].max():.0f}")

    termination_counter = df["termination"].value_counts()
    print(f"\nTermination reasons:")
    for reason, cnt in termination_counter.items():
        print(f"  {reason}: {cnt} ({100*cnt/len(df):.1f}%)")

    print(f"\nHalf-moves: median={df['half_moves'].median():.0f}, "
          f"mean={df['half_moves'].mean():.1f}, "
          f"min={df['half_moves'].min()}, max={df['half_moves'].max()}")

    print(f"\nSaved to {parquet_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()