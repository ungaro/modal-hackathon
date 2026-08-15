# Ego Trip (`egodiversity` package)

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

## Marginal diversity (`python -m egodiversity.marginal`)

Dataset-level Vendi answers "how diverse is this set?"; `marginal.py` answers
the collection-planning question "how much NEW behavior does this batch add?":

- **Lab build-up**: greedily add labs by marginal Vendi gain. On the 12.2k
  fold_clothes cache, after the first lab every other lab adds ≈0
  (ΔVendi −0.3…+0.8) — all sources capture the same motions.
- **Redundancy curve**: Vendi as random microagi episodes are added to a base
  of the other labs — +600 episodes buy +5.4 effective behaviors, flattening.

Caveat: Vendi re-standardizes per subset, so marginal deltas can dip slightly
negative — read that as "≈0 new behavior", not "negative diversity".

## Dashboard

`python -m egodiversity.dashboard` → http://localhost:8051 (deployed:
https://alp-guneysel--egodiversity-dashboard.modal.run). Six tabs:

- **Compare** — two subset panels (A/B), each configurable as random-N, greedy
  max-diversity N, or greedy min-diversity N over a lab pool, plus one-click
  curated comparisons. Shows Vendi, mean pairwise distance, coverage radius, a
  plain-English verdict, a PCA scatter with A/B membership highlighted, and
  per-subset episode tables. Hovering a scatter point shows the episode's
  actual video frame + metadata in a side panel. Random subsets use seed 0
  for A and seed 1 for B.
- **Spot check** — renders contact sheets AND inline preview videos of the
  most-similar and most-distant episode pair in subset A (via
  `egodiversity.frames`), so the score can be checked against the raw video.
- **Explore episodes** — native HTML5 preview video streamed from R2 via
  presigned URLs (`frames.get_video_url`), a 3-D hand-trajectory plot whose
  current-position markers track the video clock client-side, plus a contact
  sheet. Episodes without previews fall back to a 60-frame strip with a
  play/scrub slider (client-side, no per-frame server calls).
- **3D map** — PCA(3) WebGL scatter of all episodes, one trace per lab,
  subsets A/B highlighted; hover shows the episode's video frame thumbnail,
  click adds the full contact sheet.
- **History** — every comparison logged as JSONL (path from `EGODIV_HISTORY`,
  default `egodiversity/cache/history.jsonl`, or `/data/history.jsonl` when
  `EGODIV_CACHE` is under `/data`), debounced to one entry per config per 60s,
  tagged with the dataset label (`default` / `custom-upload`).
- **How it works** — the pipeline and validation numbers in plain English.

Cache path via `EGODIV_CACHE` (default `egodiversity/cache/features.npz`).
The "Rescore on Modal" button (active when `EGODIV_MODAL=1`, always on the
deployed instance) spawns the extraction job and polls it live, showing
elapsed time until the new cache is published (atomically) on the volume.
Thumbnails and contact sheets are disk-cached (`egodiversity/cache/thumbs|sheets`,
or `/data/...` on the volume when deployed), so repeat views are instant.

## Related work, and what's new here

The Vendi score is from [Friedman & Dieng, 2023](https://arxiv.org/abs/2210.02410);
[FAKTUAL](https://arxiv.org/html/2603.11634v1) (2026) applies Vendi-style
entropy to robotics datasets via signature kernels over trajectories. This
project's delta is not the metric:

- **A new audit target.** EgoVerse is brand new; we provide its first
  quantitative diversity audit (12,212 `fold_clothes` episodes ≈ 14 effective
  behaviors; per-episode diversity uniform across all six labs).
- **Operationalization.** A serverless scoring service (SQL manifest → Modal
  fan-out → feature cache → live dashboard), not a one-off measurement:
  12.5k episodes scored from R2 in 3.7 min.
- **A cheap kernel on purpose.** Resample→shape-normalize→RBF is cruder than
  a signature kernel but O(1) per pair after featurization — that is what
  makes the 3.7-minute scale run possible.
- **A validation harness.** The score is instrument-tested (dupes, ordering,
  bandwidth stability, saturation) before any dataset claim is made.

## Bring your own data

The dashboard can score any dataset, not just the shipped EgoVerse cache. Drop
a `.npz` on the upload box in the header (or point `EGODIV_CACHE` at it). The
format:

- `features`: float array, shape `(n_episodes, n_features)`, `n >= 4`.
- `metadata`: JSON-encoded list with one dict per row. `episode_id` is
  required; `lab`, `task_name`, `num_frames`, and `path` (s3 URI, enables
  video/thumbnail/contact-sheet previews) are optional and default to empty.

If your episodes are EgoVerse-format zarr stores on local disk, build the file
with:

```bash
python -m egodiversity.features --data-dir <dir> --out my.npz
```

If they live on S3/R2, use the Modal manifest path (`modal run -m
egodiversity.modal_app --manifest ...`; see below). Uploads are capped at
150 MB decoded; media tabs degrade to friendly placeholders for episodes
without a `path`. Note: the server is single-process and last-upload-wins —
an upload replaces the dataset for every connected viewer.

## Modal

`egodiversity/modal_app.py` recomputes features remotely from R2 using the
exact same feature code (verified bit-identical to local extraction). Episode
s3 paths vary by lab (`processed_v3/aria|mecka/flagship|microagi|...`); the
manifest JSON carries full paths. Requires `modal setup` and env vars
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3`
(see `~/.egoverse_env`).

```bash
modal run -m egodiversity.modal_app --manifest egodiversity/cache/fold_clothes_episodes.json [--labs eth,mecka] [--limit 200]
modal run -m "egodiversity.modal_app::prewarm" --manifest egodiversity/cache/fold_clothes_episodes.json  # hover thumbnails
modal deploy -m egodiversity.modal_app   # serves the dashboard off the volume cache
```

Results are written to the `egodiversity-cache` Modal volume as
`/data/features.npz` (published atomically — tmp file + rename — so a
dashboard cold start mid-rescore never reads a partial cache). The deployed
dashboard keeps one container warm (`min_containers=1`) and shows a splash
screen while the app initializes.
