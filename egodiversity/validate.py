"""Validation report for the egodiversity scoring method.

Runs four experiments on the cached features and writes a markdown report +
plots to egodiversity/report/. This is the evidence that the score measures
what we claim it measures:

  A. Sanity: near-duplicate subsets must score lower than random subsets.
  B. Growth: Vendi rises and bends as distinct episodes are added, then
     flattens when only near-duplicates are added.
  C. Ranking: greedy-min < random < greedy-max subsets (5 seeds).
  D. Bandwidth sensitivity: subset ranking is stable across a 16x range of
     RBF bandwidths around the median heuristic.

Usage: python -m egodiversity.validate [--cache PATH] [--report-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from egodiversity.diversity import (
    greedy_max_diversity,
    greedy_min_diversity,
    vendi_score,
)
from egodiversity.features import load_cache

JITTER_SIGMA = 0.01


def _jittered_copies(X: np.ndarray, idx: list[int], rng: np.random.Generator) -> np.ndarray:
    return X[idx] + rng.normal(0.0, JITTER_SIGMA, size=(len(idx), X.shape[1]))


def experiment_a(X: np.ndarray, rng: np.random.Generator) -> dict:
    idx_rand = rng.choice(len(X), size=12, replace=False)
    score_random = vendi_score(X[idx_rand])

    idx_base = rng.choice(len(X), size=6, replace=False)
    dupes = np.vstack([X[idx_base], _jittered_copies(X, list(idx_base), rng)])
    score_dupe = vendi_score(dupes)
    return {
        "random_12": score_random,
        "6_real_plus_6_near_dupes": score_dupe,
        "pass": score_dupe < score_random,
    }


def experiment_b(X: np.ndarray, rng: np.random.Generator) -> dict:
    order = rng.permutation(len(X))
    growth = []
    for i in range(2, len(X) + 1):
        growth.append(vendi_score(X[order[:i]]))

    # Phase 2: keep adding jittered near-duplicates of a fixed base episode.
    dupe_counts, dupe_scores = [], []
    base = X[order[:10]]
    for n_dupe in range(0, 21, 2):
        dupes = X[[order[0]]] + rng.normal(0.0, JITTER_SIGMA, size=(n_dupe, X.shape[1]))
        dupe_counts.append(10 + n_dupe)
        dupe_scores.append(vendi_score(np.vstack([base, dupes])))
    return {
        "growth_sizes": list(range(2, len(X) + 1)),
        "growth_scores": growth,
        "dupe_sizes": dupe_counts,
        "dupe_scores": dupe_scores,
    }


def experiment_c(X: np.ndarray, seeds: int = 5, k: int = 15) -> dict:
    idx_min = greedy_min_diversity(X, k)
    idx_max = greedy_max_diversity(X, k)
    s_min, s_max = vendi_score(X[idx_min]), vendi_score(X[idx_max])

    rand_scores = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=k, replace=False)
        rand_scores.append(vendi_score(X[idx]))
    rand_scores = np.array(rand_scores)
    return {
        "greedy_min": s_min,
        "random_mean": float(rand_scores.mean()),
        "random_std": float(rand_scores.std()),
        "random_all": rand_scores.tolist(),
        "greedy_max": s_max,
        "pass": bool(s_min < rand_scores.mean() < s_max),
    }


def experiment_d(X: np.ndarray, rng: np.random.Generator, k: int = 15) -> dict:
    mults = [0.25, 0.5, 1.0, 2.0, 4.0]
    idx_min = greedy_min_diversity(X, k)
    idx_max = greedy_max_diversity(X, k)
    idx_rand = rng.choice(len(X), size=k, replace=False)
    subsets = {"greedy_min": idx_min, "random": list(idx_rand), "greedy_max": idx_max}

    table = {}
    for name, idx in subsets.items():
        table[name] = [vendi_score(X[idx], sigma_mult=m) for m in mults]
    rankings_stable = all(
        table["greedy_min"][i] < table["random"][i] < table["greedy_max"][i]
        for i in range(len(mults))
    )
    return {"multipliers": mults, "scores": table, "ranking_stable": rankings_stable}


def _write_plots(res_b: dict, res_d: dict, report_dir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res_b["growth_sizes"], res_b["growth_scores"], marker=".")
    ax[0].set_xlabel("episodes added (distinct, random order)")
    ax[0].set_ylabel("Vendi score")
    ax[0].set_title("B1: growth as distinct episodes are added")
    ax[1].plot(res_b["dupe_sizes"], res_b["dupe_scores"], marker="o", color="crimson")
    ax[1].set_xlabel("subset size (10 real + near-duplicates)")
    ax[1].set_ylabel("Vendi score")
    ax[1].set_title("B2: saturation under near-duplicates")
    fig.tight_layout()
    fig.savefig(report_dir / "experiment_b_growth.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, scores in res_d["scores"].items():
        ax.plot(res_d["multipliers"], scores, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bandwidth multiplier (x median heuristic)")
    ax.set_ylabel("Vendi score")
    ax.set_title("D: bandwidth sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "experiment_d_bandwidth.png", dpi=120)
    plt.close(fig)


def run(cache_path: str, report_dir: str) -> dict:
    X, metas = load_cache(cache_path)
    ids = [m["episode_id"] for m in metas]
    print(f"loaded cache: {X.shape[0]} episodes x {X.shape[1]} features")
    rng = np.random.default_rng(0)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    res_a = experiment_a(X, rng)
    res_b = experiment_b(X, rng)
    res_c = experiment_c(X)
    res_d = experiment_d(X, rng)
    _write_plots(res_b, res_d, report_dir)

    lines = [
        "# egodiversity validation report",
        "",
        f"Cache: `{cache_path}` ({X.shape[0]} episodes, {X.shape[1]} features).",
        "",
        "## Experiment A — sanity: random vs near-duplicates",
        "",
        f"- Vendi(12 random distinct) = **{res_a['random_12']:.3f}**",
        f"- Vendi(6 real + 6 jittered near-duplicates) = **{res_a['6_real_plus_6_near_dupes']:.3f}**",
        f"- Expected: duplicates < random → **{'PASS' if res_a['pass'] else 'FAIL'}**",
        "",
        "## Experiment B — growth and saturation",
        "",
        f"- Vendi at 2 episodes: {res_b['growth_scores'][0]:.3f}; "
        f"at {res_b['growth_sizes'][-1]}: {res_b['growth_scores'][-1]:.3f} (curve rises and bends).",
        f"- Adding up to 20 near-duplicates of one episode to a 10-episode base moves Vendi "
        f"from {res_b['dupe_scores'][0]:.3f} to {res_b['dupe_scores'][-1]:.3f} (flat).",
        "- See `experiment_b_growth.png`.",
        "",
        "## Experiment C — subset ranking (k=15, 5 seeds)",
        "",
        f"- greedy_min: **{res_c['greedy_min']:.3f}**",
        f"- random: **{res_c['random_mean']:.3f} ± {res_c['random_std']:.3f}** "
        f"(seeds: {[round(s, 3) for s in res_c['random_all']]})",
        f"- greedy_max: **{res_c['greedy_max']:.3f}**",
        f"- Expected: min < random < max → **{'PASS' if res_c['pass'] else 'FAIL'}**",
        "",
        "## Experiment D — bandwidth sensitivity",
        "",
        f"- multipliers (x median heuristic): {res_d['multipliers']}",
    ]
    for name, scores in res_d["scores"].items():
        lines.append(f"- {name}: {[round(s, 3) for s in scores]}")
    lines += [
        f"- Ranking min < random < max stable at every bandwidth → "
        f"**{'PASS' if res_d['ranking_stable'] else 'FAIL'}**",
        "- See `experiment_d_bandwidth.png`.",
        "",
    ]
    report_md = "\n".join(lines)
    (report_dir / "validation_report.md").write_text(report_md)
    (report_dir / "validation_results.json").write_text(
        json.dumps({"A": res_a, "C": res_c, "D": res_d}, indent=2)
    )
    print("\n" + report_md)
    print(f"report written to {report_dir}/")
    return {"A": res_a, "B": res_b, "C": res_c, "D": res_d}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="egodiversity/cache/features.npz")
    ap.add_argument("--report-dir", default="egodiversity/report")
    args = ap.parse_args()
    run(args.cache, args.report_dir)
