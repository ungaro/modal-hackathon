"""Episode media access: poses, JPEG frames, and contact sheets.

Episodes are opened from a local zarr directory first (EGODIV_DATA_DIR,
default "data", with one directory per episode_id — directories are zarr v3
stores without a .zarr suffix), falling back to the episode's s3 URI (meta
["path"]) via s3fs when it is not present locally. R2 credentials come from
the environment or ~/.egoverse_env (shell-style KEY=value lines).

All remote failures raise RuntimeError with a clear message — UI callers are
expected to catch it and render a placeholder instead of crashing.

Keep this module importable without dash/modal (PIL, s3fs, zarr, numpy only).
"""

from __future__ import annotations

import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw

POSE_SUBSAMPLE = 200  # max points returned per trajectory
SHEETS_ENV = "EGODIV_SHEETS_DIR"

_groups: dict[str, zarr.Group] = {}  # episode_id -> open zarr group
_fs = None                           # lazily created s3fs filesystem
_r2: dict[str, str] | None = None    # lazily loaded R2 credentials


def _r2_env() -> dict[str, str]:
    """R2 credentials from env vars, falling back to ~/.egoverse_env."""
    global _r2
    if _r2 is not None:
        return _r2
    keys = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL_S3")
    creds = {k: os.environ.get(k) for k in keys}
    if not all(creds.values()):
        env_file = Path.home() / ".egoverse_env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k = k.strip()
                    if not creds.get(k):  # env vars win; file fills gaps
                        creds[k] = v.strip().strip('"').strip("'")
    missing = [k for k in keys if not creds.get(k)]
    if missing:
        raise RuntimeError(
            f"R2 credentials missing ({', '.join(missing)}); set env vars or "
            f"add them to ~/.egoverse_env"
        )
    _r2 = creds
    return _r2


def _s3_fs():
    global _fs
    if _fs is None:
        import s3fs

        creds = _r2_env()
        _fs = s3fs.S3FileSystem(
            key=creds["R2_ACCESS_KEY_ID"],
            secret=creds["R2_SECRET_ACCESS_KEY"],
            endpoint_url=creds["AWS_ENDPOINT_URL_S3"],
            client_kwargs={"region_name": "auto"},
        )
    return _fs


def _open_group(meta: dict) -> zarr.Group:
    """Open the episode's zarr group: local directory first, then R2."""
    episode_id = meta["episode_id"]
    if episode_id in _groups:
        return _groups[episode_id]

    local = Path(os.environ.get("EGODIV_DATA_DIR", "data")) / episode_id
    if (local / "zarr.json").exists():
        g = zarr.open_group(str(local), mode="r")
    else:
        path = meta.get("path")
        if not path:
            raise RuntimeError(
                f"episode {episode_id} not found locally under {local.parent}/ "
                f"and metadata has no 'path' s3 URI"
            )
        try:
            store = _s3_fs().get_mapper(path.removeprefix("s3://"))
            g = zarr.open_group(store=store, mode="r")
        except Exception as exc:
            raise RuntimeError(f"failed to open {path} from R2: {exc}") from exc
    _groups[episode_id] = g
    return g


def get_poses(episode_meta: dict) -> dict[str, np.ndarray]:
    """Right/left end-effector xyz trajectories, subsampled to <=200 points.

    Returns {"right": (M, 3), "left": (M, 3)}.
    """
    try:
        g = _open_group(episode_meta)
        right = g["right.obs_ee_pose"][:, :3]
        left = g["left.obs_ee_pose"][:, :3]
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"failed to read poses for {episode_meta['episode_id']}: {exc}"
        ) from exc
    n = len(right)
    keep = np.linspace(0, n - 1, min(n, POSE_SUBSAMPLE)).astype(int)
    return {"right": right[keep], "left": left[keep]}


def _read_frame(g: zarr.Group, idx: int) -> bytes:
    arr = g["images.front_1"]
    idx = max(0, min(int(idx), arr.shape[0] - 1))
    el = arr[idx]
    # Local zarr v3 VLenBytes reads wrap the scalar in nested 0-d object
    # arrays; remote reads return np.bytes_ directly. Unwrap to bytes either
    # way.
    while isinstance(el, np.ndarray):
        el = el.item()
    return bytes(el)


def get_frames(episode_meta: dict, idxs: list[int]) -> list[bytes]:
    """Fetch JPEG frames (front_1 camera) by index, local-first then R2.

    Multiple frames are fetched in parallel (max 8 threads), preserving the
    order of idxs.
    """
    g = _open_group(episode_meta)
    try:
        if len(idxs) <= 1:
            return [_read_frame(g, i) for i in idxs]
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lambda i: _read_frame(g, i), idxs))
    except Exception as exc:
        raise RuntimeError(
            f"failed to read frames for {episode_meta['episode_id']}: {exc}"
        ) from exc


def num_frames(episode_meta: dict) -> int:
    """Number of frames in the episode's front_1 image array."""
    return int(_open_group(episode_meta)["images.front_1"].shape[0])


def _sheets_dir() -> Path:
    env = os.environ.get(SHEETS_ENV)
    if env:
        return Path(env)
    if os.environ.get("EGODIV_CACHE", "").startswith("/data"):
        return Path("/data/sheets")
    return Path(__file__).parent / "cache" / "sheets"


def contact_sheet(
    episode_meta: dict, n_frames: int = 12, thumb: tuple[int, int] = (320, 240)
) -> bytes:
    """A 4-col x 3-row JPEG grid of evenly spaced frames, episode_id labeled.

    Sheets are cached on disk keyed by episode_id (and frame count).
    """
    episode_id = episode_meta["episode_id"]
    cache_dir = _sheets_dir()
    cache_path = cache_dir / f"{episode_id}_{n_frames}.jpg"
    if cache_path.exists():
        return cache_path.read_bytes()

    total = num_frames(episode_meta)
    idxs = np.linspace(0, total - 1, n_frames).astype(int).tolist()
    frames = get_frames(episode_meta, idxs)

    cols, rows = 4, int(np.ceil(n_frames / 4))
    w, h = thumb
    sheet = Image.new("RGB", (cols * w, rows * h), color=(20, 20, 20))
    for i, blob in enumerate(frames):
        img = Image.open(io.BytesIO(blob)).convert("RGB").resize(thumb)
        sheet.paste(img, ((i % cols) * w, (i // cols) * h))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, 12 * len(episode_id) + 8, 16], fill=(0, 0, 0))
    draw.text((4, 3), episode_id, fill=(255, 255, 0))

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


THUMBS_ENV = "EGODIV_THUMBS_DIR"
def _thumbs_dir() -> Path:
    env = os.environ.get(THUMBS_ENV)
    if env:
        return Path(env)
    if os.environ.get("EGODIV_CACHE", "").startswith("/data"):
        return Path("/data/thumbs")
    return Path(__file__).parent / "cache" / "thumbs"


def get_thumbnail(episode_meta: dict, size: tuple[int, int] = (240, 180)) -> bytes:
    """A small JPEG of the episode's middle front_1 frame, disk-cached.

    Same local-then-R2 access as the other media helpers; R2 failures raise
    RuntimeError. Cached under egodiversity/cache/thumbs/ (or EGODIV_THUMBS_DIR
    / /data/thumbs, mirroring the contact-sheet cache rules).
    """
    episode_id = episode_meta["episode_id"]
    cache_dir = _thumbs_dir()
    cache_path = cache_dir / f"{episode_id}.jpg"
    if cache_path.exists():
        return cache_path.read_bytes()

    total = num_frames(episode_meta)
    blob = get_frames(episode_meta, [total // 2])[0]
    img = Image.open(io.BytesIO(blob)).convert("RGB").resize(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def get_video_url(episode_meta: dict, expires_in: int = 3600) -> str | None:
    """Presigned GET URL for the episode's preview MP4 on R2, or None.

    The preview sits at the episode's zarr key with .zarr swapped for .mp4.
    Metas without "path" (the small local cache) fall back to the aria
    prefix. Returns None when no key can be derived; credential/signing
    failures raise RuntimeError like the rest of the module. Signing is
    purely local — no network call is made here.
    """
    path = episode_meta.get("path")
    if not path:
        # Fallback for the small local eth cache (metas without "path").
        # Only derive for timestamp-style aria ids (2025-11-11-...) — custom
        # uploaded datasets with arbitrary ids must not get a bogus URL that
        # signs fine but 404s.
        episode_id = episode_meta.get("episode_id", "")
        if not re.match(r"^\d{4}-\d{2}-\d{2}-", episode_id):
            return None
        path = f"s3://rldb/processed_v3/aria/{episode_id}.zarr"
    if not path.startswith("s3://"):
        return None
    bucket, _, prefix = path.removeprefix("s3://").partition("/")
    if not bucket or not prefix.endswith(".zarr"):
        return None
    key = prefix[: -len(".zarr")] + ".mp4"

    try:
        import boto3

        creds = _r2_env()
        client = boto3.client(
            "s3",
            endpoint_url=creds["AWS_ENDPOINT_URL_S3"],
            aws_access_key_id=creds["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=creds["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"failed to sign video URL for {episode_meta.get('episode_id')}: {exc}"
        ) from exc


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="egodiversity frame utilities.")
    ap.add_argument("--prewarm", action="store_true",
                    help="generate thumbnails for all LOCAL episodes (no R2)")
    args = ap.parse_args()
    if args.prewarm:
        data_dir = Path(os.environ.get("EGODIV_DATA_DIR", "data"))
        episodes = sorted(
            p for p in data_dir.iterdir() if p.is_dir() and (p / "zarr.json").exists()
        )
        ok = 0
        for ep in episodes:
            try:
                get_thumbnail({"episode_id": ep.name})
                ok += 1
            except RuntimeError as exc:
                print(f"SKIP {ep.name}: {exc}")
        print(f"prewarmed {ok}/{len(episodes)} thumbnails into {_thumbs_dir()}")
