# Chess Elo Prediction from Single Games — Project Report

## Goal

Predict the Elo rating of both players from a single chess game. Regression task, target variable is the average Elo of both players. 

## Data Source

Lichess monthly database dumps (database.lichess.org). PGN files, zstandard-compressed, millions of games per month. Currently using `lichess_db_standard_rated_2017-04.pgn.zst` (~11M games). Two additional months downloaded as reserve (2017-05, 2017-06).

## Key Design Decisions

### Own Stockfish analysis instead of Lichess evals
Lichess annotates only ~6% of games with Stockfish evaluations. Those 6% are not a random sample — they are games where a user requested analysis, biasing toward more engaged players and possibly more dramatic games. Additionally, eval depth is heterogeneous and not recorded in the PGN dumps.

Decision: Run Stockfish locally at fixed depth 15 on all sampled games. This gives consistent, reproducible evaluations across the entire dataset and eliminates the 6% coverage bias. The trade-off is compute time (~2 days on a Ryzen 5 9600X for 90k games), but it makes the dataset scientifically cleaner than what Tijhuis (2023) and Omori (2024) used.

### Rapid only, one time control
Mixed time controls would confound the features — a blunder in bullet means something different than a blunder in rapid. Rapid was chosen over blitz because it gives more signal per game (players have time to think) and clock-based features are more meaningful.

### Both players predicted together
Each game produces one prediction (average Elo of both players). Only games where both players have similar ratings are included (±100 Elo difference). Lichess matchmaking ensures this is the case for the majority of games anyway.

Critical implementation detail: Train-test split must happen on **player level**, not game level. If the same player appears in both train and test, the model partially learns player identity instead of rating-level behavior.

### Natural distribution + sample weights
The Elo distribution is concentrated around 1400-1800 with sparse tails (<1000 and >2200). Instead of stratified sampling (which would require multiple months to fill edge bins), the approach is to sample the natural distribution and use inverse-density sample weights during training. XGBoost and Random Forest support this natively.

Evaluation will report MAE per 200-Elo bin with bin sizes, so performance across the full range is transparent.

## Sampling Pipeline

Filters applied during streaming:
- Rated games only
- Rapid only (Lichess formula: base + 40 × increment, 480-1499 seconds)
- No bot players
- No abandoned games or rules infractions
- Rating difference ≤ ±100 between players
- Minimum 20 half-moves
- No eval filter (all games eligible)

Output: Parquet file with game ID, both Elos, average Elo, moves as SAN list, clock values in seconds, termination reason, time control. ~80-90k games expected from one month.

## Exploration Results (2017-04, 50k games, preliminary)

- Rating distribution: median 1551, roughly normal, range 827-2395
- Matchmaking is tight: median rating diff 0, std 26.6
- Median game length: 61 half-moves
- 82% Normal termination, 18% Time forfeit
- 100% of rapid games have clock annotations
- ~12.5% of rapid games have Lichess eval annotations (irrelevant for our pipeline, but confirms the selection bias we're avoiding)

## Planned Feature Engineering Approach

Two categories of features, inspired by chess intuition and GothamChess "Guess the Elo" reasoning:

**Standard features (baseline):** Centipawn loss statistics (mean, median, std, max), blunder/mistake/inaccuracy counts at various thresholds, material balance trajectory, game length, result.

**Orthogonal features (intended contribution):** Opening choice and theory depth, position complexity preference, time management patterns (from clock data), endgame technique independent of blunder count, planning consistency across game phases. These are meant to capture signals that CPL-based features miss — the *type* of decisions a player makes, not just how many mistakes they make.

The hypothesis: Blunder-counting features separate 800 from 1500 well, but struggle to distinguish 1500 from 1800. The orthogonal features should add discriminative power in the midrange.

## Known Limitations

- **Selection bias in matchmaking:** Only games between similarly-rated players are included. The model cannot be applied to games with large rating gaps.
- **Single time control:** Results do not generalize to blitz or bullet.
- **Single month:** Potential temporal bias (rating pool characteristics of April 2017 may differ from other periods).
- **Stockfish depth 15:** Sufficient for the target rating range (800-2400) but weaker than Lichess server analysis (depth 18-22). An empirical comparison on a small subset can validate this.
- **No player metadata:** Account age, total games played, rating volatility (Glicko RD) are unavailable in the dumps. These would likely be strongly predictive.

## Prior Work

- **Tijhuis et al. (2023):** 30 hand-crafted features, 10-class classification, ~17% accuracy (vs 10% random baseline). Most important features: blunder counts. Used Lichess evals directly.
- **Omori & Tadepalli (2024):** CNN-LSTM, no hand-crafted features, MAE 182. Used Lichess evals. March 2021 data excluded due to datacenter fire.
- **Maia Chess (McIlroy-Young et al., 2020):** Related but predicts moves, not ratings.

## Current State

- [x] Project setup (conda env, dependencies, folder structure)
- [x] Data downloaded (3 months, ~34M games)
- [x] Exploration script run, data characteristics understood
- [x] Sampling pipeline written (ready to run on full month)
- [ ] Run sampling pipeline, produce final Parquet
- [ ] Stockfish analysis pipeline (depth 15, checkpointing)
- [ ] Feature engineering
- [ ] Model training and evaluation
