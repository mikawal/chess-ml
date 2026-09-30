"""
sampled_games_enriched.parquet  ->  players.gexf

One file. Open it in Gephi or Gephi Lite and the graph is already laid out,
colored by elo bin, with edge thickness scaled by number of games played.

Nodes      players
Edges      played at least one game together
Color      the player's elo bin (9 categories, blue = low, red = high)
Thickness  number of games between that pair
Node size  number of distinct opponents

Deps: pip install polars numpy python-igraph
"""

import numpy as np
import polars as pl
from xml.sax.saxutils import escape

PARQUET = "data/sampled/sampled_games_enriched.parquet"
OUT = "players.gexf"

MIN_DEGREE = 0          # 0 = every player. Try 2 or 3 if the file is too heavy.
PRECOMPUTE_LAYOUT = True  # False = no positions, lay it out yourself in Gephi

# blue -> red across the 9 bins, ordered low elo to high elo
PALETTE = [
    (49, 54, 149), (69, 117, 180), (116, 173, 209), (171, 217, 233),
    (255, 255, 191),
    (254, 224, 144), (253, 174, 97), (244, 109, 67), (215, 48, 39),
]

# ----------------------------------------------------------------------
# 1. load
# ----------------------------------------------------------------------
df = pl.read_parquet(
    PARQUET, columns=["white", "black", "white_elo", "black_elo", "avg_elo", "elo_bin"]
)
df = df.with_columns(
    pl.col("white").str.to_lowercase(),
    pl.col("black").str.to_lowercase(),
    pl.col("elo_bin").cast(pl.Utf8),
).filter(pl.col("white") != pl.col("black"))
print(f"games: {df.height:,}")

# ----------------------------------------------------------------------
# 2. bin order  (sort the 9 labels by the mean avg_elo of their games,
#    so this works whatever the label strings look like)
# ----------------------------------------------------------------------
bin_order = (
    df.group_by("elo_bin")
      .agg(pl.col("avg_elo").mean().alias("m"))
      .sort("m")["elo_bin"].to_list()
)
print(f"elo bins, low to high: {bin_order}")
if len(bin_order) != len(PALETTE):
    print(f"note: found {len(bin_order)} bins, palette has {len(PALETTE)} "
          f"- recycling colors")
bin_rank = {b: i for i, b in enumerate(bin_order)}

# ----------------------------------------------------------------------
# 3. nodes: one row per player, with their dominant elo bin
# ----------------------------------------------------------------------
long = pl.concat([
    df.select(pl.col("white").alias("player"),
              pl.col("white_elo").alias("elo"), "elo_bin"),
    df.select(pl.col("black").alias("player"),
              pl.col("black_elo").alias("elo"), "elo_bin"),
])

nodes = (
    long.group_by("player")
        .agg(
            pl.col("elo").mean().round(0).cast(pl.Int32).alias("mean_elo"),
            pl.col("elo_bin").mode().first().alias("elo_bin"),
            pl.len().alias("n_games"),
        )
        .sort("player")
        .with_row_index("id")
)
n = nodes.height
print(f"players: {n:,}")

# ----------------------------------------------------------------------
# 4. edges: undirected, deduplicated, weight = games played
# ----------------------------------------------------------------------
lut = nodes.select("player", "id")
e = (df.join(lut, left_on="white", right_on="player").rename({"id": "s"})
       .join(lut, left_on="black", right_on="player").rename({"id": "d"})
       .select("s", "d"))
s, d = e["s"].to_numpy(), e["d"].to_numpy()
pair = np.stack([np.minimum(s, d), np.maximum(s, d)], axis=1)
uniq, weight = np.unique(pair, axis=0, return_counts=True)
print(f"pairings: {len(uniq):,}   repeat pairings: {(weight > 1).sum():,}   "
      f"max games between one pair: {weight.max()}")

degree = np.bincount(uniq.ravel(), minlength=n)

# ----------------------------------------------------------------------
# 5. optional degree filter
# ----------------------------------------------------------------------
if MIN_DEGREE > 0:
    keep = degree >= MIN_DEGREE
    edge_keep = keep[uniq[:, 0]] & keep[uniq[:, 1]]
    uniq, weight = uniq[edge_keep], weight[edge_keep]
    old_to_new = -np.ones(n, dtype=np.int64)
    old_to_new[keep] = np.arange(keep.sum())
    uniq = old_to_new[uniq]
    nodes = nodes.filter(pl.Series(keep)).with_columns(
        pl.Series("id", np.arange(keep.sum()))
    )
    n = nodes.height
    degree = np.bincount(uniq.ravel(), minlength=n)
    print(f"after degree>={MIN_DEGREE}: {n:,} nodes / {len(uniq):,} edges")

# ----------------------------------------------------------------------
# 6. layout
# ----------------------------------------------------------------------
if PRECOMPUTE_LAYOUT:
    import igraph as ig
    g = ig.Graph(n=n, edges=[tuple(x) for x in uniq])
    print("laying out (slow part - minutes at this size)...")
    coords = np.asarray(g.layout_drl().coords, dtype=float)
    span = np.ptp(coords, axis=0).max() or 1.0
    coords = (coords - coords.mean(axis=0)) / span * 4000.0
    print("layout done")
else:
    coords = None

# ----------------------------------------------------------------------
# 7. write GEXF
# ----------------------------------------------------------------------
labels = nodes["player"].to_list()
bins = nodes["elo_bin"].to_list()
mean_elo = nodes["mean_elo"].to_list()
n_games = nodes["n_games"].to_list()

node_size = 3.0 + 12.0 * np.sqrt(degree) / max(np.sqrt(degree.max()), 1.0)
edge_thick = 0.3 + 2.7 * np.minimum(weight, 10) / 10.0

with open(OUT, "w", encoding="utf-8") as f:
    f.write(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gexf xmlns="http://www.gexf.net/1.2draft" '
        'xmlns:viz="http://www.gexf.net/1.2draft/viz" version="1.2">\n'
        '<meta><description>Lichess player co-play network</description></meta>\n'
        '<graph mode="static" defaultedgetype="undirected">\n'
        '<attributes class="node">\n'
        '  <attribute id="0" title="elo_bin" type="string"/>\n'
        '  <attribute id="1" title="mean_elo" type="integer"/>\n'
        '  <attribute id="2" title="n_games" type="integer"/>\n'
        '  <attribute id="3" title="degree" type="integer"/>\n'
        '</attributes>\n<nodes>\n'
    )
    for i in range(n):
        r, gc, b = PALETTE[bin_rank[bins[i]] % len(PALETTE)]
        f.write(f'<node id="{i}" label="{escape(str(labels[i]))}">')
        f.write(f'<viz:color r="{r}" g="{gc}" b="{b}"/>')
        f.write(f'<viz:size value="{node_size[i]:.2f}"/>')
        if coords is not None:
            f.write(f'<viz:position x="{coords[i,0]:.1f}" '
                    f'y="{coords[i,1]:.1f}" z="0.0"/>')
        f.write('<attvalues>'
                f'<attvalue for="0" value="{escape(str(bins[i]))}"/>'
                f'<attvalue for="1" value="{mean_elo[i]}"/>'
                f'<attvalue for="2" value="{n_games[i]}"/>'
                f'<attvalue for="3" value="{degree[i]}"/>'
                '</attvalues></node>\n')

    f.write('</nodes>\n<edges>\n')
    for j in range(len(uniq)):
        f.write(f'<edge id="{j}" source="{uniq[j,0]}" target="{uniq[j,1]}" '
                f'weight="{weight[j]}">'
                f'<viz:thickness value="{edge_thick[j]:.2f}"/></edge>\n')
    f.write('</edges>\n</graph>\n</gexf>\n')

print(f"wrote {OUT}")
print("open it at https://gephi.org/gephi-lite/ (browser, no install) "
      "or in the Gephi desktop app")