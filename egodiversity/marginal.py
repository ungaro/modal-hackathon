"""Marginal diversity scoring: what does a new batch of episodes actually add?

Dataset-level Vendi answers "how diverse is this set?". The operational
question for data collection is different: "how much NEW behavior did this
batch add on top of what we already have?" This module computes both sides:

1. Lab build-up: greedily add labs in order of marginal Vendi gain — which
   data source contributes the most new behavior, and which are redundant
   given the others.
2. Episode redundancy curve: Vendi as random episodes from one lab are added
   to a fixed base of the others — shows how quickly the marginal value of
   yet another fold_clothes episode collapses.

Run: python -m egodiversity.marginal [--cache egodiversity/cache/features_full.npz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egodiversity.diversity import vendi_score
from egodiversity.features import load_cache

LAB_SAMPLE = 150     # per-lab cap for the build-up analysis
BASE_SIZE = 200      # base set size for the redundancy curve
ADD_MAX = 600        # max episodes added in the redundancy curve
EVAL_EVERY = 25      # evaluate Vendi every k additions
SEED = 0


def _lab_pools(metas: list[dict]) -> dict[str, list[int]]:
    pools: dict[str, list[int]] = {}
    for i, m in enumerate(metas):
        lab = m.get("lab", "")
        if lab:
            pools.setdefault(lab, []).append(i)
    return pools


def _sample(pool: list[int], n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())


def lab_buildup(X: np.ndarray, pools: dict[str, list[int]]) -> list[dict]:
    """Greedily add labs by marginal Vendi gain. Returns ordered steps with
    the Vendi after each addition and the marginal gain."""
    sampled = {lab: _sample(idx, LAB_SAMPLE, SEED) for lab, idx in pools.items()}
    chosen: list[str] = []
    chosen_idx: list[int] = []
    steps = []
    current = 1.0
    remaining = set(sampled)
    while remaining:
        best_lab, best_score = None, -1.0
        for lab in remaining:
            score = vendi_score(X[chosen_idx + sampled[lab]])
            if score > best_score:
                best_lab, best_score = lab, score
        steps.append({
            "lab": best_lab,
            "n_episodes": len(chosen_idx) + len(sampled[best_lab]),
            "vendi": round(best_score, 2),
            "marginal_gain": round(best_score - current, 2),
        })
        current = best_score
        chosen_idx += sampled[best_lab]
        chosen.append(best_lab)
        remaining.discard(best_lab)
    return steps


def redundancy_curve(X: np.ndarray, pools: dict[str, list[int]],
                     target_lab: str) -> list[dict]:
    """Vendi as random target-lab episodes are added to a fixed base of the
    other labs (all standardized/scored jointly per evaluation point)."""
    rng = np.random.default_rng(SEED)
    others = [i for lab, idx in pools.items() if lab != target_lab for i in idx]
    base = _sample(others, BASE_SIZE, SEED)
    add_order = rng.permutation(pools[target_lab])[:ADD_MAX].tolist()

    points = []
    eval_at = set(range(0, len(add_order) + 1, EVAL_EVERY)) | {len(add_order)}
    for k in sorted(eval_at):
        idx = base + add_order[:k]
        points.append({"added": k, "n_total": len(idx),
                       "vendi": round(vendi_score(X[idx]), 2)})
    return points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="egodiversity/cache/features_full.npz")
    ap.add_argument("--out-dir", default="egodiversity/report")
    ap.add_argument("--target-lab", default=None,
                    help="lab for the redundancy curve (default: largest)")
    args = ap.parse_args()

    X, metas = load_cache(args.cache)
    pools = _lab_pools(metas)
    target = args.target_lab or max(pools, key=lambda lab: len(pools[lab]))
    print(f"cache: {X.shape}; labs: { {k: len(v) for k, v in pools.items()} }")

    print("\n== Lab build-up (greedy marginal Vendi gain) ==")
    steps = lab_buildup(X, pools)
    for s in steps:
        print(f"  + {s['lab']:10s} vendi={s['vendi']:6.2f}  (+{s['marginal_gain']:.2f})")

    print(f"\n== Redundancy curve: adding '{target}' episodes to a base of the other labs ==")
    curve = redundancy_curve(X, pools, target)
    for p in curve:
        print(f"  +{p['added']:4d} episodes (n={p['n_total']:4d})  vendi={p['vendi']:6.2f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "marginal_results.json").write_text(json.dumps(
        {"lab_buildup": steps, "redundancy_curve": {"target_lab": target, "points": curve}},
        indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot([p["added"] for p in curve], [p["vendi"] for p in curve], "o-")
    ax.set_xlabel(f"random '{target}' episodes added to base of other labs")
    ax.set_ylabel("Vendi score (effective # distinct behaviors)")
    ax.set_title("Marginal value of additional episodes collapses quickly")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "marginal_redundancy.png", dpi=150)
    print(f"\nwrote {out}/marginal_results.json and marginal_redundancy.png")


if __name__ == "__main__":
    main()
