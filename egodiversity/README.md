# egodiversity

A quantitative, text-free diversity measurement suite for the EgoVerse
egocentric robotics dataset. It extracts kinematic motion features per episode
(no images, no LLM judges), scores the diversity of any episode subset with the
Vendi score (an effective count of distinct behaviors), ranks/selects
max- and min-diversity subsets, ships a validation report proving the score
behaves sanely, and a Dash dashboard for comparing two subsets head to head.

## Quickstart

```bash
source EgoVerse/emimic/bin/activate

# 1. Build the feature cache from local episode zarr stores (~1 min, no JPEG decode)
python -m egodiversity.features --data-dir data --out egodiversity/cache/features.npz

# 2. Run the validation experiments (prints numbers, writes report + PNGs)
python -m egodiversity.validate

# 3. Launch the dashboard on port 8051
python -m egodiversity.dashboard
```

## Design decisions

- **Shape, not scale.** Trajectories (right/left EE xyz, right EE quaternion,
  head xyz) are resampled in normalized time (T=64 / T=32), shifted to start
  at the origin, and divided by the RMS extent of the right-hand path. Two
  episodes performing the same motion in different parts of the room therefore
  map to nearby feature vectors — behavioral similarity, not workspace
  similarity. Absolute scale is preserved separately in scalar summary stats
  (log path length, duration, speed mean/std, normalized jerk), computed on
  the raw trajectory.
- **RBF kernel + median heuristic.** Features are z-score standardized; the
  RBF bandwidth is the median pairwise euclidean distance. Experiment D shows
  subset rankings are stable across a 16x bandwidth range around it.
- **Vendi score = effective number of distinct behaviors.** Exponential of the
  Shannon entropy of the kernel's normalized eigenvalue spectrum. k identical
  episodes score ~1; k mutually dissimilar episodes score ~k.
- **Validation experiments** (`python -m egodiversity.validate`):
  - A: near-duplicate subsets must score lower than random ones (sanity).
  - B: Vendi grows and bends as distinct episodes are added, flattens under
    near-duplicates (saturation).
  - C: greedy-min < random < greedy-max subsets over 5 seeds (ranking).
  - D: ranking stability across RBF bandwidths (sensitivity).
  Report + plots land in `egodiversity/report/`.

## Dashboard

`python -m egodiversity.dashboard` → http://localhost:8051. Two subset panels
(A/B), each configurable as random-N, greedy max-diversity N, or greedy
min-diversity N. Shows Vendi, mean pairwise distance, coverage radius, a PCA
scatter with A/B membership highlighted, per-subset episode tables, and a
winner verdict. Cache path via `EGODIV_CACHE` (default
`egodiversity/cache/features.npz`). The "Rescore on Modal" button is active
when `EGODIV_MODAL=1`.

## Modal

`egodiversity/modal_app.py` recomputes features remotely from R2
(`s3://rldb/processed_v3/aria/<episode_id>.zarr`) using the exact same feature
code. Requires `modal setup` and env vars `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3` (see `~/.egoverse_env`).
Also requires `uv pip install s3fs` locally for direct R2 access.

```bash
modal run egodiversity.modal_app -- <episode_id> [<episode_id> ...]
modal deploy egodiversity.modal_app   # serves the dashboard at /data cache
```

Results are written to the `egodiversity-cache` Modal volume as
`/data/features.npz`.
