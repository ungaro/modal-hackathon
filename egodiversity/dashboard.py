"""Dash dashboard for comparing diversity of two episode subsets.

Port 8051 (8050 is taken). Feature cache path from env EGODIV_CACHE
(default: egodiversity/cache/features.npz).

Subsets are defined by a candidate pool (lab filter) + a selection strategy
(random / greedy max-diversity / greedy min-diversity). Greedy selectors run
on a capped deterministic subsample of the pool (GREEDY_POOL_CAP) so the UI
stays interactive on 10k+ episode caches.

Run: python -m egodiversity.dashboard
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, dash_table, dcc, html
from sklearn.decomposition import PCA

from egodiversity.diversity import (
    coverage_radius,
    greedy_max_diversity,
    greedy_min_diversity,
    mean_pairwise_distance,
    vendi_score,
)
from egodiversity.features import load_cache

CACHE_PATH = os.environ.get("EGODIV_CACHE", "egodiversity/cache/features.npz")
PORT = 8051

STRATEGIES = ["random", "greedy max-diversity", "greedy min-diversity"]
GREEDY_POOL_CAP = 600   # candidate cap for greedy selectors (interactivity)
MAX_SUBSET_N = 1000     # Vendi eigendecomposition stays sub-second up to here


def _select_subset(X: np.ndarray, strategy: str, n: int, pool: list[int],
                   seed: int = 0) -> list[int]:
    """Select n episode indices (into X) from pool using strategy."""
    rng = np.random.default_rng(seed)
    pool = list(pool)
    n = min(n, len(pool))
    if strategy == "random":
        return sorted(rng.choice(pool, size=n, replace=False).tolist())
    # Greedy selectors: cap the candidate pool for interactivity.
    if len(pool) > GREEDY_POOL_CAP:
        pool = sorted(rng.choice(pool, size=GREEDY_POOL_CAP, replace=False).tolist())
    sub = X[pool]
    if strategy == "greedy max-diversity":
        chosen = greedy_max_diversity(sub, n)
    else:
        chosen = greedy_min_diversity(sub, n)
    return sorted(pool[i] for i in chosen)


def _scores_block(name: str, X: np.ndarray, idx: list[int]) -> html.Div:
    sub = X[idx]
    return html.Div(
        [
            html.H3(name),
            html.Div(f"Vendi: {vendi_score(sub):.2f}",
                     style={"fontSize": "32px", "fontWeight": "bold"}),
            html.Div(f"mean pairwise distance: {mean_pairwise_distance(sub):.2f}"),
            html.Div(f"coverage radius (k=3): {coverage_radius(sub):.2f}"),
        ],
        style={"width": "45%", "display": "inline-block", "verticalAlign": "top",
               "padding": "10px"},
    )


def _subset_table(metas: list[dict], idx: list[int]) -> dash_table.DataTable:
    df = pd.DataFrame({
        "episode_id": [metas[i]["episode_id"] for i in idx],
        "lab": [metas[i].get("lab", "") for i in idx],
        "num_frames": [metas[i].get("num_frames", "") for i in idx],
    })
    return dash_table.DataTable(
        df.to_dict("records"),
        page_size=6,
        style_table={"height": "220px", "overflowY": "auto"},
        style_cell={"fontSize": "12px", "textAlign": "left"},
    )


def create_app() -> Dash:
    X, metas = load_cache(CACHE_PATH)
    ids = [m["episode_id"] for m in metas]
    labs = sorted({m.get("lab", "") for m in metas} - {""})
    n_ep = len(X)

    # lab -> pool of indices ("all" = everything)
    pools: dict[str, list[int]] = {"all": list(range(n_ep))}
    for lab in labs:
        pools[lab] = [i for i, m in enumerate(metas) if m.get("lab") == lab]
    lab_options = ["all"] + labs

    # Deterministic 2-D PCA over all episodes, computed once.
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(X)

    slider_max = min(n_ep, MAX_SUBSET_N)

    def subset_controls(tag: str, default_strategy: str, default_lab: str) -> html.Div:
        return html.Div(
            [
                html.H3(f"Subset {tag}"),
                html.Label("Pool (lab):"),
                dcc.Dropdown(lab_options, default_lab, id=f"lab-{tag}",
                             clearable=False),
                html.Label("Strategy:"),
                dcc.Dropdown(STRATEGIES, default_strategy, id=f"strategy-{tag}",
                             clearable=False),
                html.Label("N episodes:"),
                dcc.Slider(2, slider_max, 1, value=15, id=f"n-{tag}",
                           marks={2: "2", slider_max: str(slider_max)}),
            ],
            style={"width": "45%", "display": "inline-block", "padding": "10px"},
        )

    app = Dash(__name__, title="egodiversity")
    app.layout = html.Div(
        [
            html.H1("egodiversity — EgoVerse subset diversity comparison"),
            html.Div(f"{n_ep} episodes, {X.shape[1]} features per episode "
                     f"(cache: {CACHE_PATH})"),
            html.Hr(),
            html.Div([subset_controls("A", "greedy max-diversity", "all"),
                      subset_controls("B", "greedy min-diversity", "all")]),
            html.Div(id="scores-row"),
            html.Div(id="verdict",
                     style={"fontSize": "20px", "fontWeight": "bold", "padding": "10px"}),
            dcc.Graph(id="pca-scatter"),
            html.Div([html.Div(id="table-A",
                               style={"width": "45%", "display": "inline-block"}),
                      html.Div(id="table-B",
                               style={"width": "45%", "display": "inline-block",
                                      "float": "right"})]),
            html.Hr(),
            html.Button("Rescore on Modal", id="rescore-btn", n_clicks=0),
            html.Div(id="rescore-out", style={"padding": "10px", "color": "#555"}),
        ],
        style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "sans-serif"},
    )

    @callback(
        Output("scores-row", "children"),
        Output("verdict", "children"),
        Output("pca-scatter", "figure"),
        Output("table-A", "children"),
        Output("table-B", "children"),
        Input("lab-A", "value"), Input("strategy-A", "value"), Input("n-A", "value"),
        Input("lab-B", "value"), Input("strategy-B", "value"), Input("n-B", "value"),
    )
    def update(lab_a, strat_a, n_a, lab_b, strat_b, n_b):
        idx_a = _select_subset(X, strat_a, n_a, pools[lab_a])
        idx_b = _select_subset(X, strat_b, n_b, pools[lab_b])
        v_a, v_b = vendi_score(X[idx_a]), vendi_score(X[idx_b])

        name_a, name_b = f"A ({lab_a}, {strat_a})", f"B ({lab_b}, {strat_b})"
        if v_a >= v_b:
            verdict = f"Subset {name_a} is {v_a / max(v_b, 1e-9):.1f}x more diverse than {name_b} by Vendi score"
        else:
            verdict = f"Subset {name_b} is {v_b / max(v_a, 1e-9):.1f}x more diverse than {name_a} by Vendi score"

        member = np.array(["other"] * n_ep, dtype=object)
        member[idx_b] = "B"
        member[idx_a] = "A"
        both = set(idx_a) & set(idx_b)
        for i in both:
            member[i] = "A∩B"
        df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "subset": member,
                           "episode_id": ids,
                           "lab": [m.get("lab", "") for m in metas]})
        fig = px.scatter(
            df, x="PC1", y="PC2", color="subset",
            hover_data=["episode_id", "lab"],
            category_orders={"subset": ["A", "B", "A∩B", "other"]},
            color_discrete_map={"A": "#d62728", "B": "#1f77b4",
                                "A∩B": "#9467bd", "other": "#cccccc"},
            title="PCA of all episodes (subset membership highlighted)",
        )
        fig.update_traces(marker={"size": 9, "opacity": 0.85})

        return (
            html.Div([_scores_block(f"Subset {name_a}", X, idx_a),
                      _scores_block(f"Subset {name_b}", X, idx_b)]),
            verdict,
            fig,
            _subset_table(metas, idx_a),
            _subset_table(metas, idx_b),
        )

    @callback(Output("rescore-out", "children"), Input("rescore-btn", "n_clicks"))
    def rescore(n_clicks):
        if not n_clicks:
            return ""
        if os.environ.get("EGODIV_MODAL") != "1":
            return ("Modal not configured. Set EGODIV_MODAL=1 (and run `modal setup`) "
                    "to enable remote feature extraction on Modal/R2.")
        try:
            from egodiversity.modal_app import rescore_remote

            # Map episode ids to manifest dicts (with full s3 paths) when the
            # manifest is available; otherwise assume the aria prefix.
            manifest_path = Path(__file__).parent / "cache" / "fold_clothes_episodes.json"
            by_hash = {}
            if manifest_path.exists():
                by_hash = {e["episode_hash"]: e for e in json.loads(manifest_path.read_text())}
            episodes = [
                by_hash.get(eid, {"episode_hash": eid, "lab": "eth", "task": "fold_clothes",
                                  "num_frames": -1,
                                  "path": f"s3://rldb/processed_v3/aria/{eid}.zarr"})
                for eid in ids
            ]
            return rescore_remote(episodes)
        except Exception as exc:  # demo hook — surface, don't crash
            return f"Modal rescore failed: {exc}"

    return app


if __name__ == "__main__":
    create_app().run(port=PORT, debug=False)
