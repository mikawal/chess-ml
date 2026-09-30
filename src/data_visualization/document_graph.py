"""
sampled_games_enriched.parquet -> documentation figures + hard numbers.

Outputs (SVG, vector, drop straight into a report):
  fig_bin_matrix.svg    elo bin x elo bin pairing matrix  ("who played whom")
  fig_components.svg    connected component size distribution
  fig_degree.svg        degree distribution (opponents per player)
  bin_matrix.csv        the matrix as numbers
  graph_stats.md        every figure quoted in the captions, so nothing is hand-typed

Deps: pip install polars numpy scipy matplotlib
"""

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PARQUET = "data/sampled/sampled_games_enriched.parquet"

# Set this to your actual bin edges if you know them, e.g.
# BIN_EDGES = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
# Leave as None to infer them from the data (edges printed for you to check).
BIN_EDGES = None

plt.rcParams.update({
    "figure.dpi": 110, "savefig.bbox": "tight",
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})

# ----------------------------------------------------------------------
# 1. load
# ----------------------------------------------------------------------
df = pl.read_parquet(
    PARQUET,
    columns=["white", "black", "white_elo", "black_elo", "avg_elo", "elo_bin"],
).with_columns(
    pl.col("white").str.to_lowercase(),
    pl.col("black").str.to_lowercase(),
    pl.col("elo_bin").cast(pl.Utf8),
).filter(pl.col("white") != pl.col("black"))

n_games = df.height
print(f"games: {n_games:,}")

# ----------------------------------------------------------------------
# 2. bin edges
#    elo_bin is per GAME (derived from avg_elo), but the matrix needs a bin
#    per PLAYER. So recover the boundaries from the observed avg_elo ranges
#    and apply them to each player's own rating.
# ----------------------------------------------------------------------
bin_stats = (
    df.group_by("elo_bin")
      .agg(pl.col("avg_elo").min().alias("lo"),
           pl.col("avg_elo").max().alias("hi"),
           pl.col("avg_elo").mean().alias("m"))
      .sort("m")
)
bin_labels = bin_stats["elo_bin"].to_list()
print(f"bins ({len(bin_labels)}), low to high: {bin_labels}")

if BIN_EDGES is None:
    lo = bin_stats["lo"].to_numpy()
    hi = bin_stats["hi"].to_numpy()
    edges = [(hi[i] + lo[i + 1]) / 2 for i in range(len(bin_labels) - 1)]
    print(f"inferred bin edges: {[round(x) for x in edges]}")
    print("  ^ sanity-check these against how you built elo_bin")
else:
    edges = list(BIN_EDGES)
edges = np.asarray(edges, dtype=float)
n_bins = len(edges) + 1

# ----------------------------------------------------------------------
# 3. player table
# ----------------------------------------------------------------------
long = pl.concat([
    df.select(pl.col("white").alias("p"), pl.col("white_elo").alias("elo")),
    df.select(pl.col("black").alias("p"), pl.col("black_elo").alias("elo")),
])
players = (long.group_by("p")
             .agg(pl.col("elo").mean().alias("mean_elo"),
                  pl.len().alias("n_games"))
             .sort("p")
             .with_row_index("id"))
n_players = players.height
player_bin = np.digitize(players["mean_elo"].to_numpy(), edges)
print(f"players: {n_players:,}")

# ----------------------------------------------------------------------
# 4. edges of the graph
# ----------------------------------------------------------------------
lut = players.select("p", "id")
e = (df.join(lut, left_on="white", right_on="p").rename({"id": "s"})
       .join(lut, left_on="black", right_on="p").rename({"id": "d"})
       .select("s", "d"))
s, d = e["s"].to_numpy(), e["d"].to_numpy()
pair = np.stack([np.minimum(s, d), np.maximum(s, d)], axis=1)
uniq, weight = np.unique(pair, axis=0, return_counts=True)
degree = np.bincount(uniq.ravel(), minlength=n_players)
n_pairs = len(uniq)
print(f"pairings: {n_pairs:,}  mean degree: {2*n_pairs/n_players:.2f}")

# ----------------------------------------------------------------------
# 5. components
# ----------------------------------------------------------------------
adj = coo_matrix((np.ones(n_pairs, np.int8), (uniq[:, 0], uniq[:, 1])),
                 shape=(n_players, n_players))
n_comp, comp = connected_components(adj, directed=False)
csize = np.bincount(comp)
giant = csize.max()
games_in_giant = (comp[s] == comp[s][np.argmax(csize[comp[s]])]).sum() \
    if n_comp > 1 else n_games
giant_games = np.bincount(comp[s], minlength=n_comp).max()
print(f"components: {n_comp:,}   giant: {giant:,} players "
      f"({giant/n_players:.1%}), {giant_games:,} games ({giant_games/n_games:.1%})")

# ----------------------------------------------------------------------
# FIGURE 1 - bin x bin pairing matrix
# ----------------------------------------------------------------------
bw, bb = player_bin[s], player_bin[d]
M = np.zeros((n_bins, n_bins), dtype=np.int64)
np.add.at(M, (bw, bb), 1)
M = M + M.T - np.diag(np.diag(M))  # symmetrise; games are undirected

fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(np.where(M > 0, M, np.nan), cmap="RdYlBu_r",
               norm=LogNorm(vmin=max(M[M > 0].min(), 1), vmax=M.max()),
               origin="lower")
ax.set_xticks(range(n_bins)); ax.set_yticks(range(n_bins))
ax.set_xticklabels(bin_labels, rotation=45, ha="right")
ax.set_yticklabels(bin_labels)
ax.set_xlabel("elo bin of one player"); ax.set_ylabel("elo bin of the other")
ax.set_title(f"games played between elo bins  (n = {n_games:,})", pad=12)
for i in range(n_bins):
    for j in range(n_bins):
        if M[i, j] > 0:
            ax.text(j, i, f"{100*M[i,j]/M.sum():.1f}", ha="center", va="center",
                    fontsize=6.5, color="black")
fig.colorbar(im, ax=ax, shrink=0.75, label="games (log scale)")
ax.set_aspect("equal")
fig.savefig("fig_bin_matrix.svg"); plt.close(fig)

pl.DataFrame(M, schema=[str(b) for b in bin_labels]).write_csv("bin_matrix.csv")

# ----------------------------------------------------------------------
# FIGURE 2 - component size distribution
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
vals, counts = np.unique(csize, return_counts=True)
ax.scatter(vals, counts, s=14, color="#4575B4", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("component size (players)"); ax.set_ylabel("number of components")
ax.annotate(f"giant component\n{giant:,} players ({giant/n_players:.1%})",
            xy=(giant, 1), xytext=(giant * 0.06, 12),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
ax.set_title(f"{n_comp:,} connected components", pad=10)
ax.grid(alpha=0.25, which="both", lw=0.4)
fig.savefig("fig_components.svg"); plt.close(fig)

# ----------------------------------------------------------------------
# FIGURE 3 - degree distribution
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
dv, dc = np.unique(degree, return_counts=True)
ax.scatter(dv, dc, s=14, color="#D73027", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("distinct opponents"); ax.set_ylabel("number of players")
ax.set_title(f"degree distribution  (median {int(np.median(degree))}, "
             f"mean {degree.mean():.1f}, max {degree.max()})", pad=10)
ax.grid(alpha=0.25, which="both", lw=0.4)
fig.savefig("fig_degree.svg"); plt.close(fig)

# ----------------------------------------------------------------------
# stats file
# ----------------------------------------------------------------------
same_bin = (bw == bb).mean()
adj_bin = (np.abs(bw.astype(int) - bb.astype(int)) <= 1).mean()
repeat = (weight > 1).mean()

with open("graph_stats.md", "w") as f:
    f.write(f"""# Player co-play graph

| quantity | value |
|---|---|
| games | {n_games:,} |
| players (nodes) | {n_players:,} |
| distinct pairings (edges) | {n_pairs:,} |
| mean degree | {2*n_pairs/n_players:.2f} |
| median degree | {int(np.median(degree))} |
| max degree | {degree.max():,} |
| pairings that met more than once | {repeat:.1%} |
| connected components | {n_comp:,} |
| giant component, players | {giant:,} ({giant/n_players:.1%}) |
| giant component, games | {giant_games:,} ({giant_games/n_games:.1%}) |
| components with < 10 players | {(csize < 10).sum():,} |
| games between players in the same elo bin | {same_bin:.1%} |
| games within one bin of each other | {adj_bin:.1%} |

## Splitting

A leakage-free split assigns whole components, never individual players.
The giant component alone accounts for {giant_games/n_games:.1%} of games, so
any split target below that cannot be met by component assignment. Above that
threshold, games crossing a split boundary must be dropped.
""")

print("\nwrote fig_bin_matrix.svg, fig_components.svg, fig_degree.svg, "
      "bin_matrix.csv, graph_stats.md")