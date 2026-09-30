# Player co-play graph

| quantity | value |
|---|---|
| games | 333,478 |
| players (nodes) | 91,620 |
| distinct pairings (edges) | 314,595 |
| mean degree | 6.87 |
| median degree | 3 |
| max degree | 317 |
| pairings that met more than once | 4.3% |
| connected components | 1,179 |
| giant component, players | 89,151 (97.3%) |
| giant component, games | 331,882 (99.5%) |
| components with < 10 players | 1,178 |
| games between players in the same elo bin | 67.8% |
| games within one bin of each other | 99.5% |

## Splitting

A leakage-free split assigns whole components, never individual players.
The giant component alone accounts for 99.5% of games, so
any split target below that cannot be met by component assignment. Above that
threshold, games crossing a split boundary must be dropped.
