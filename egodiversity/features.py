"""Kinematic feature extraction for EgoVerse episodes.

Each episode is reduced to a fixed-length vector capturing the *shape* of the
bimanual + head motion, plus a few scalar summary statistics. No images are
decoded — poses only.

Design choice (shape, not scale): every trajectory is resampled in normalized
time, shifted to start at the origin, and divided by the RMS extent of the
right-hand path. This makes features invariant to where the episode happened
in the room and how large the workspace was, so two episodes doing the same
folding motion at different tables land close together. That is the defensible
notion of "behavioral similarity" we want a diversity score to measure; the
absolute scale information is preserved separately via the scalar summary
stats (log path length, duration, speed), which are computed on the *raw*
trajectory.

Keep this module importable without dash/torch — it is shared with the Modal
remote functions (numpy, zarr, tqdm only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

T_TRAJ = 64   # resample length for EE trajectories / quaternion
T_HEAD = 32   # resample length for head position
EPS = 1e-8


def _resample(traj: np.ndarray, T: int) -> np.ndarray:
    """Linear interpolation of a (N, D) trajectory to T samples over normalized time."""
    n = len(traj)
    if n == T:
        return traj.astype(np.float64, copy=True)
    t_old = np.linspace(0.0, 1.0, n)
    t_new = np.linspace(0.0, 1.0, T)
    out = np.empty((T, traj.shape[1]), dtype=np.float64)
    for d in range(traj.shape[1]):
        out[:, d] = np.interp(t_new, t_old, traj[:, d])
    return out


def _canonicalize_quat(q: np.ndarray) -> np.ndarray:
    """Sign-canonicalize (N, 4) quaternions (xyzw convention, w last).

    Two steps: enforce temporal hemisphere continuity (flip sign when the dot
    product with the previous quaternion is negative, so linear resampling
    does not cut across the sphere), then flip so w >= 0 (q and -q are the
    same rotation).
    """
    q = q.astype(np.float64, copy=True)
    dots = np.einsum("ij,ij->i", q[1:], q[:-1])
    flip = np.concatenate([[False], dots < 0])
    sign = np.where(np.cumsum(flip) % 2 == 1, -1.0, 1.0)
    q = q * sign[:, None]
    q[q[:, 3] < 0] *= -1.0
    return q


def _summary_stats(xyz: np.ndarray, fps: float) -> np.ndarray:
    """Scalar stats on the RAW right-hand trajectory (pre-normalization)."""
    diffs = np.diff(xyz, axis=0)
    step = np.linalg.norm(diffs, axis=1)
    path_len = float(step.sum())
    duration = len(xyz) / fps
    speed = step * fps
    # Normalized jerk: mean squared 3rd derivative / path length^3.
    dt = 1.0 / fps
    jerk = np.diff(xyz, n=3, axis=0) / dt**3
    norm_jerk = float((jerk**2).sum(axis=1).mean() / max(path_len, EPS) ** 3)
    return np.array(
        [np.log(max(path_len, EPS)), duration, speed.mean(), speed.std(), np.log1p(norm_jerk)],
        dtype=np.float64,
    )


def features_from_arrays(
    right: np.ndarray, left: np.ndarray, head: np.ndarray, fps: float
) -> np.ndarray:
    """Core feature computation from pose arrays.

    right/left/head: (N, 7) arrays of [x, y, z, qx, qy, qz, qw]. Only right
    xyz + quat, left xyz, and head xyz are used. Returns a 1-D float64 vector.
    """
    r_xyz, l_xyz, h_xyz = right[:, :3], left[:, :3], head[:, :3]
    r_quat = _canonicalize_quat(right[:, 3:7])

    stats = _summary_stats(r_xyz, fps)

    # RMS extent of the right-hand path around its start point; one shared
    # scale for all position channels keeps hands/head in the same units.
    scale = float(np.sqrt((np.linalg.norm(r_xyz - r_xyz[0], axis=1) ** 2).mean()))
    scale = max(scale, EPS)

    def norm_traj(xyz: np.ndarray, T: int) -> np.ndarray:
        return (_resample(xyz, T) - xyz[0]).ravel() / scale

    parts = [
        norm_traj(r_xyz, T_TRAJ),            # 3*T_TRAJ  right-hand path shape
        norm_traj(l_xyz, T_TRAJ),            # 3*T_TRAJ  left-hand path shape
        _resample(r_quat, T_TRAJ).ravel(),   # 4*T_TRAJ  right-hand orientation
        norm_traj(h_xyz, T_HEAD),            # 3*T_HEAD  head path shape
        stats,                               # 5         summary stats
    ]
    return np.concatenate(parts)


def extract_features(episode_dir: str) -> tuple[np.ndarray, dict]:
    """Extract the feature vector + metadata for one episode directory."""
    episode_dir = str(episode_dir)
    with open(Path(episode_dir) / "zarr.json") as f:
        attrs = json.load(f)["attributes"]
    fps = float(attrs.get("fps", 30.0))

    g = zarr.open_group(episode_dir, mode="r")
    right = g["right.obs_ee_pose"][:]
    left = g["left.obs_ee_pose"][:]
    head = g["obs_head_pose"][:]

    vec = features_from_arrays(right, left, head, fps)
    meta = {
        "episode_id": Path(episode_dir).name,
        "task_name": attrs.get("task_name", "unknown"),
        "num_frames": int(right.shape[0]),
        "fps": fps,
    }
    return vec, meta


def build_cache(data_dir: str, out_path: str) -> tuple[np.ndarray, list[dict]]:
    """Extract features for every episode under data_dir and save an .npz."""
    from tqdm import tqdm

    episodes = sorted(
        p for p in Path(data_dir).iterdir() if p.is_dir() and (p / "zarr.json").exists()
    )
    if not episodes:
        raise RuntimeError(f"no episode directories found under {data_dir}")

    vecs, metas = [], []
    for ep in tqdm(episodes, desc="extracting features"):
        vec, meta = extract_features(str(ep))
        vecs.append(vec)
        metas.append(meta)

    X = np.stack(vecs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=X, metadata=json.dumps(metas))
    return X, metas


def load_cache(path: str) -> tuple[np.ndarray, list[dict]]:
    """Load a cache written by build_cache."""
    z = np.load(path, allow_pickle=False)
    return z["features"], json.loads(str(z["metadata"]))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the egodiversity feature cache.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="egodiversity/cache/features.npz")
    args = ap.parse_args()

    X, metas = build_cache(args.data_dir, args.out)
    print(f"saved {X.shape[0]} episodes x {X.shape[1]} features to {args.out}")
