# egodiversity validation report

Cache: `egodiversity/cache/features.npz` (76 episodes, 741 features).

## Experiment A — sanity: random vs near-duplicates

- Vendi(12 random distinct) = **4.665**
- Vendi(6 real + 6 jittered near-duplicates) = **3.253**
- Expected: duplicates < random → **PASS**

## Experiment B — growth and saturation

- Vendi at 2 episodes: 1.642; at 76: 10.023 (curve rises and bends).
- Adding up to 20 near-duplicates of one episode to a 10-episode base moves Vendi from 4.262 to 2.816 (flat).
- See `experiment_b_growth.png`.

## Experiment C — subset ranking (k=15, 5 seeds)

- greedy_min: **1.639**
- random: **5.106 ± 0.040** (seeds: [5.059, 5.149, 5.154, 5.103, 5.067])
- greedy_max: **5.190**
- Expected: min < random < max → **PASS**

## Experiment D — bandwidth sensitivity

- multipliers (x median heuristic): [0.25, 0.5, 1.0, 2.0, 4.0]
- greedy_min: [1.996, 1.977, 1.639, 1.25, 1.082]
- random: [15.0, 13.678, 5.17, 1.917, 1.237]
- greedy_max: [15.0, 13.718, 5.19, 1.919, 1.237]
- Ranking min < random < max stable at every bandwidth → **PASS**
- See `experiment_d_bandwidth.png`.
