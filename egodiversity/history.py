"""Append-only JSONL history of dashboard subset comparisons.

Every Compare-tab update is logged as one JSON line: utc timestamp, the A/B
subset definitions (lab, strategy, n), both Vendi scores, and the plain-English
verdict. Writes are debounced — an identical configuration logged within
DEBOUNCE_SECONDS of a previous entry is skipped, so dragging a slider does not
flood the log.

Path resolution (checked at call time, so tests can set env vars late):
EGODIV_HISTORY if set; else /data/history.jsonl when EGODIV_CACHE points under
/data (the Modal volume mount); else egodiversity/cache/history.jsonl.

When the path is under /data we best-effort commit the egodiversity-cache
Modal volume after each append so the log survives container restarts.

Keep this module importable without dash/modal (stdlib only; modal is imported
lazily inside the volume commit).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEBOUNCE_SECONDS = 60
VOLUME_NAME = "egodiversity-cache"


def history_path() -> Path:
    """Resolve the history file path from the environment (see docstring)."""
    env = os.environ.get("EGODIV_HISTORY")
    if env:
        return Path(env)
    if os.environ.get("EGODIV_CACHE", "").startswith("/data"):
        return Path("/data/history.jsonl")
    return Path(__file__).parent / "cache" / "history.jsonl"


def read_history(path: Path | None = None) -> list[dict]:
    """All history entries, oldest first. Tolerant of missing/corrupt lines."""
    path = path or history_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _parse_ts(entry: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(entry["ts"])
    except (KeyError, ValueError):
        return None


def _commit_volume() -> None:
    """Best-effort commit of the Modal volume holding /data (no-op locally)."""
    try:
        import modal

        modal.Volume.from_name(VOLUME_NAME).commit()
    except Exception:
        pass


def append_history(
    a: dict, b: dict, vendi_a: float, vendi_b: float, verdict: str,
    path: Path | None = None, dataset: str = "default",
) -> bool:
    """Append one comparison record. Returns True if written, False if debounced.

    a/b are subset definitions, e.g. {"lab": "mecka", "strategy": "random",
    "n": 200}. `dataset` labels which dataset was scored ("default" or
    "custom-upload").
    """
    path = path or history_path()
    now = datetime.now(timezone.utc)
    entry = {
        "ts": now.isoformat(timespec="seconds"),
        "dataset": dataset,
        "a": {"lab": a.get("lab", ""), "strategy": a.get("strategy", ""), "n": a.get("n", 0)},
        "b": {"lab": b.get("lab", ""), "strategy": b.get("strategy", ""), "n": b.get("n", 0)},
        "vendi_a": round(float(vendi_a), 3),
        "vendi_b": round(float(vendi_b), 3),
        "verdict": verdict,
    }

    # Debounce: identical config already logged within DEBOUNCE_SECONDS.
    for prev in reversed(read_history(path)):
        ts = _parse_ts(prev)
        if ts is None:
            continue
        age = (now - ts).total_seconds()
        if age > DEBOUNCE_SECONDS:
            break  # entries are chronological; older ones cannot match either
        if prev.get("a") == entry["a"] and prev.get("b") == entry["b"]:
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    if str(path).startswith("/data"):
        _commit_volume()
    return True
