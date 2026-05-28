# Chess Elo Prediction from Single Games — Project Report

## Goal

Predict the Elo rating of both players from a single chess game. Regression task, target variable is the average Elo of both players.

## Data Source

Lichess monthly database dumps (database.lichess.org). PGN files, zstandard-compressed. Currently using three months from 2017: `lichess_db_standard_rated_2017-04.pgn.zst`, `2017-05.pgn.zst`, `2017-06.pgn.zst`. ~34.5M games total across all three. Multiple months are needed to populate the rare tail bins (<800 and ≥2200), which contain too few games per month for meaningful stratified sampling.

## Key Design Decisions

### Own Stockfish analysis instead of Lichess evals

Lichess annotates only ~6-12% of games with Stockfish evaluations. Those games are not a random sample — they are games where a user requested analysis, biasing toward more engaged players and possibly more dramatic games. Additionally, eval depth is heterogeneous and not recorded in the PGN dumps.

Decision: Run Stockfish locally at a fixed depth on the games selected for engine analysis. Consistent, reproducible evaluations across the dataset, no coverage bias.

Depth is not yet fixed. Initial target is 15. A calibration experiment will run first: 100 representative games (across Elo bins) analyzed at depths 12, 15, 18, and 22. Whichever depth is the lowest at which the intended features (ACPL, blunder counts, agreement rate) stabilize will be the working depth. Estimated runtime on Ryzen 5 9600X with 12 parallel single-threaded Stockfish processes:

- Depth 12: ~4–6 hours for 90k games
- Depth 15: ~17–20 hours
- Depth 18: ~3–5 days

Missed mates: only mates up to depth 4 are in scope. Deeper missed mates are out of scope (would require either much higher depth or a two-pass scheme on high-swing positions; deferred).

### Rapid only, one time control

Mixed time controls would confound the features — a blunder in bullet means something different than a blunder in rapid. Rapid was chosen over blitz because it gives more signal per game (players have time to think) and clock-based features are more meaningful.

### Both players predicted together

Each game produces one prediction (average Elo of both players). Only games where both players have similar ratings are included (±100 Elo difference). Lichess matchmaking ensures this is the case for the majority of games anyway.

Critical implementation detail: Train-test split must happen on **player level**, not game level. If the same player appears in both train and test, the model partially learns player identity instead of rating-level behavior.

### Stratified pool, training distribution as hyperparameter

The Elo distribution is concentrated around 1400–1800 with sparse tails. The original plan was natural-distribution training with inverse-density sample weights. This was revised:

- The **pool** (data available for training) is sampled with **stratified reservoir sampling per Elo bin**: Vitter's Algorithm R, 9 bins, per-bin cap of 50,000 games, across all three months. The result is a roughly balanced pool where supply permits.
- The **training distribution** (what gradients actually see) is controlled separately, at training time, via sample weights. A temperature parameter α ∈ [0, 1] interpolates between natural frequencies (α = 0) and uniform per-bin (α = 1). α is treated as a hyperparameter, grid-searched against a stratified validation metric (e.g. mean of per-bin MAEs).

This decouples two concerns. A balanced pool gives flexibility — the training distribution becomes a knob to tune. A natural-distribution pool would lock in midrange bias and prevent any post-hoc rebalancing.

Evaluation will report MAE per 200-Elo bin with bin sizes, so performance across the full range is transparent regardless of which α wins.

## Sampling Pipeline

Filters applied during streaming:

- Rated games only
- Rapid only (Lichess formula: base + 40 × increment, 480–1499 seconds)
- No bot players
- No abandoned games or rules infractions
- Rating difference ≤ ±100 between players
- Minimum 20 half-moves
- No eval filter (all games eligible regardless of Lichess `%eval` annotations)

Sampling method:

- Vitter's Algorithm R reservoir sampler, one per Elo bin
- 9 bins keyed off avg_elo: `<800`, `[800,1000)`, `[1000,1200)`, `[1200,1400)`, `[1400,1600)`, `[1600,1800)`, `[1800,2000)`, `[2000,2200)`, `>=2200`
- Per-bin cap: 50,000 games (configurable via `--per_bin_target`)
- Full streaming pass across all input files; no early termination
- Single-process implementation. Multi-process parallelism would have given a ~3-4× speedup on the available hardware but was not implemented for v1 to keep complexity low.

Output: Parquet file with game ID, both Elos, average Elo, rating difference, moves as comma-separated SAN list, clock values in seconds, termination reason, time control, half-move count, and bin label. Lichess `%eval` annotations are not extracted — there is no leakage of Lichess engine evaluations into features.

## Sampling Run Results (3 months, 2017-04 through 2017-06)

- 34,555,025 games read
- 26,917,192 rejected as non-rapid (~78%)
- 2,817,921 rejected for rating difference > ±100
- 257,169 rejected as too short
- 36,453 rejected as abandoned / rules infraction
- 4,526,290 eligible games observed across all bins
- 333,478 retained after per-bin reservoir caps

Per-bin counts (kept / total eligible):

| Bin            | Kept   | Eligible    | Retention |
|----------------|--------|-------------|-----------|
| <800           | 47     | 47          | 100.0%    |
| [800, 1000)    | 22,340 | 22,340      | 100.0%    |
| [1000, 1200)   | 50,000 | 232,583     | 21.5%     |
| [1200, 1400)   | 50,000 | 762,977     | 6.5%      |
| [1400, 1600)   | 50,000 | 1,383,785   | 3.6%      |
| [1600, 1800)   | 50,000 | 1,329,333   | 3.8%      |
| [1800, 2000)   | 50,000 | 662,901     | 7.5%      |
| [2000, 2200)   | 50,000 | 121,233     | 41.2%     |
| >=2200         | 11,091 | 11,091      | 100.0%    |

Other characteristics: avg_elo range 784–2632 (Q1 1263, median 1578, Q3 1880); 82.7% normal termination, 17.3% time forfeit; half-moves median 64, mean 70, max 314.

Total pool size is ~3.7× the planned Stockfish budget of 90k, giving headroom for distribution-shape experiments and any later per-bin sub-sampling without re-running the full scan.

Compute: 13 hours wall-clock, single-threaded, ~640 games/sec sustained. Parallelization across the three input files would have cut this to ~3-4 hours but was not pursued.

## Planned Feature Engineering Approach

Features split into two layers based on extraction cost:

**PGN-only features (cheap, computable on the full 333k pool):**

- Opening choice and theory depth, early-queen-out events
- Castling timing, development sequencing, basic center-control metrics
- Time-usage patterns from `%clk` data: per-move time, time-spike events, time pressure in endgame
- Game-phase lengths, result, termination type
- Move-time variance, half-move count

**Stockfish-derived features (expensive, computed on ~90k subset):**

- Centipawn loss statistics: mean, median, std, max (ACPL)
- Blunder / mistake / inaccuracy counts at thresholds (e.g. 200 / 100 / 50 centipawns)
- Best-move agreement rate with the engine
- Missed mates (≤ mate-in-4)
- Phase-specific versions of all of the above (opening / middlegame / endgame)

Approach: train a baseline model on PGN-only features across the full 333k pool, then measure the lift from adding Stockfish-derived features on the 90k subset. This quantifies whether engine-analysis runtime is worth its cost for this task.

Qualitative hypothesis, inspired by GothamChess "Guess the Elo" reasoning: blunder-counting features separate 800 from 1500 well but struggle to distinguish 1500 from 1800. Features capturing the *type* of decisions a player makes — planning consistency, opening understanding, time-management style — should add discriminative power in the midrange.

## Known Limitations

- **Selection bias in matchmaking:** Only games between similarly-rated players are included. The model cannot be applied to games with large rating gaps.
- **Single time control:** Results do not generalize to blitz or bullet.
- **Temporal scope:** All data from April–June 2017. Rating-pool characteristics in other periods (inflation, population shifts) are not captured. Cross-period validation would require additional dumps.
- **`<800` bin is functionally noise:** 47 games is insufficient for any per-bin signal. Likely to be merged with `[800,1000)` or dropped entirely during training. Ratings below ~1000 on Lichess are often provisional and the underlying label itself is noisy at that range.
- **Stockfish depth (target 15) weaker than Lichess server analysis (depth 18-22).** Adequate for the target rating range, but to be validated via the planned calibration experiment.
- **Includes games with Lichess `%eval` annotations:** ~6-12% of the pool are games that had post-hoc analysis requested. These are a non-random subset and slightly bias the sample. The eval annotations themselves are not used as features (no leakage), but the selection bias remains. Can be revisited with a `had_eval_annotations` flag feature if it becomes a measurable problem.
- **No player metadata:** Account age, total games played, Glicko rating deviation are not in the dumps. These would likely be strongly predictive.

## Prior Work

- **Tijhuis et al. (2023):** 30 hand-crafted features, 10-class classification, ~17% accuracy (vs 10% random baseline). Most important features: blunder counts. Used Lichess evals directly.
- **Omori & Tadepalli (2024):** CNN-LSTM, no hand-crafted features, MAE 182. Used Lichess evals. March 2021 data excluded due to datacenter fire.
- **Maia Chess (McIlroy-Young et al., 2020):** Related but predicts moves, not ratings.

## Current State

- [x] Project setup (conda env, dependencies, folder structure)
- [x] Data downloaded (3 months, ~34M games)
- [x] Exploration script run, data characteristics understood
- [x] Sampling pipeline written (stratified reservoir sampling, 9 bins)
- [x] Sampling pipeline run on full 3-month set; 333,478 games in stratified pool
- [ ] Inspection of sampled data (in progress)
- [ ] Stockfish depth calibration experiment (100 games × 4 depths)
- [ ] PGN-only feature extraction across full 333k pool
- [ ] Baseline model (PGN features only) with α grid search
- [ ] Stockfish analysis pipeline at chosen depth (multi-process, with checkpointing)
- [ ] Stockfish-feature extraction on ~90k subset of the pool
- [ ] Combined model training and evaluation, comparison vs PGN-only baseline