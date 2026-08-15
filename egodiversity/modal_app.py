"""Modal integration for egodiversity.

Requires Modal auth — run `modal setup` once before using this module's
remote functions. R2 credentials come from env vars (see ~/.egoverse_env):
  R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3

Episodes are referenced by full s3 URI (the `zarr_processed_path` column of the
EgoVerse SQL table), e.g. s3://rldb/processed_v3/aria/<episode_id>.zarr — the
prefix varies by lab (aria / mecka/flagship / microagi / yam / eva).

The remote feature computation reuses egodiversity.features verbatim, so
remote and local caches are directly comparable.

Usage:
  modal run -m egodiversity.modal_app \
      --manifest egodiversity/cache/fold_clothes_episodes.json \
      --labs eth,mecka --limit 200
"""

from __future__ import annotations

import io
import json
import os

import modal

app = modal.App("egodiversity")

_PIP_CORE = ["numpy", "scipy", "zarr==3.1.5", "s3fs", "tqdm"]

# NOTE: add_local_python_source must be the LAST build step (Modal requirement),
# so the two images duplicate the pip_install list instead of chaining.
feature_image = (
    modal.Image.debian_slim()
    .pip_install(*_PIP_CORE)
    .add_local_python_source("egodiversity")
)
dashboard_image = (
    modal.Image.debian_slim()
    .pip_install(*_PIP_CORE, "dash", "plotly", "scikit-learn", "pandas")
    .add_local_python_source("egodiversity")
)

_r2_keys = {
    k: os.environ.get(k)
    for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL_S3")
}
r2_secret = modal.Secret.from_dict({k: v for k, v in _r2_keys.items() if v})

cache_volume = modal.Volume.from_name("egodiversity-cache", create_if_missing=True)

REMOTE_CACHE_PATH = "/data/features.npz"


def _open_remote_group(s3_uri: str):
    """Open an episode zarr store from an s3:// URI via s3fs (runs inside Modal)."""
    import s3fs
    import zarr

    bucket_key = s3_uri.removeprefix("s3://")
    fs = s3fs.S3FileSystem(
        key=os.environ["R2_ACCESS_KEY_ID"],
        secret=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        client_kwargs={"region_name": "auto"},
    )
    return zarr.open_group(store=fs.get_mapper(bucket_key), mode="r")


@app.function(image=feature_image, secrets=[r2_secret], cpu=2, memory=4096)
def extract_remote(episode: dict) -> bytes:
    """Extract the shared kinematic feature vector for one episode (given its
    manifest dict with at least a `path` s3 URI). Returns the vector serialized
    with np.save, or b"" on failure (bad/missing arrays, robot embodiments with
    different keys) so one broken episode cannot kill the whole map."""
    import numpy as np

    from egodiversity.features import features_from_arrays

    try:
        g = _open_remote_group(episode["path"])
        fps = float(g.attrs.get("fps", 30.0))
        vec = features_from_arrays(
            g["right.obs_ee_pose"][:], g["left.obs_ee_pose"][:], g["obs_head_pose"][:], fps
        )
    except Exception as e:  # noqa: BLE001 - per-episode fault isolation
        print(f"SKIP {episode.get('episode_hash', episode.get('path'))}: {type(e).__name__}: {e}")
        return b""
    buf = io.BytesIO()
    np.save(buf, vec)
    return buf.getvalue()


@app.function(image=feature_image, secrets=[r2_secret], volumes={"/data": cache_volume},
              timeout=3600)
def extract_all(episodes: list[dict]) -> str:
    """Map extract_remote over manifest dicts and write the feature cache to
    the shared volume. Metadata keys mirror the local cache (episode_id,
    task_name, num_frames) plus lab/path for cross-lab analysis."""
    import numpy as np

    vecs, metas, failed = [], [], 0
    for ep, blob in zip(episodes, extract_remote.map(episodes)):
        if not blob:
            failed += 1
            continue
        vecs.append(np.load(io.BytesIO(blob)))
        metas.append({
            "episode_id": ep.get("episode_hash", ""),
            "task_name": ep.get("task", ""),
            "lab": ep.get("lab", ""),
            "num_frames": ep.get("num_frames", -1),
            "path": ep.get("path", ""),
        })
    if not vecs:
        raise RuntimeError(f"all {len(episodes)} episodes failed extraction")
    X = np.stack(vecs)
    np.savez(REMOTE_CACHE_PATH, features=X, metadata=json.dumps(metas))
    cache_volume.commit()
    print(f"wrote {X.shape} to {REMOTE_CACHE_PATH}; {failed} episodes skipped")
    return REMOTE_CACHE_PATH


@app.function(image=dashboard_image, volumes={"/data": cache_volume},
              secrets=[r2_secret])
@modal.wsgi_app()
def dashboard():
    """Serve the Dash dashboard from Modal, reading the volume cache."""
    os.environ.setdefault("EGODIV_CACHE", REMOTE_CACHE_PATH)
    from egodiversity.dashboard import create_app

    return create_app().server


def rescore_remote(episodes: list[dict]) -> str:
    """Local-side helper used by the dashboard's 'Rescore on Modal' button."""
    with app.run():
        path = extract_all.remote(list(episodes))
    return f"Modal recompute finished, cache written to {path} on volume 'egodiversity-cache'."


@app.local_entrypoint()
def main(manifest: str, labs: str = "", limit: int = 0, seed: int = 0) -> None:
    """Score episodes from a manifest JSON (list of dicts with path/lab/task/
    episode_hash/num_frames). --labs filters by lab (comma-separated);
    --limit caps episodes per lab (0 = no cap), sampled with --seed."""
    import random

    episodes = json.load(open(manifest))
    if labs:
        keep = set(labs.split(","))
        episodes = [e for e in episodes if e.get("lab") in keep]
    if limit > 0:
        rng = random.Random(seed)
        by_lab: dict[str, list[dict]] = {}
        for e in episodes:
            by_lab.setdefault(e.get("lab", ""), []).append(e)
        episodes = [
            e for lab_eps in by_lab.values()
            for e in rng.sample(lab_eps, min(limit, len(lab_eps)))
        ]
    print(f"extracting features for {len(episodes)} episodes")
    print(extract_all.remote(episodes))
