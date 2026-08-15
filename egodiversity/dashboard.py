"""Dash dashboard v2 for the egodiversity suite.

Port 8051 (8050 is taken). Feature cache path from env EGODIV_CACHE
(default: egodiversity/cache/features.npz).

Six tabs aimed at a judge with zero context:

- Compare: A/B subset machinery (lab pool, selection strategy, N) with
  plain-English labels and verdict, curated one-click comparisons, Vendi +
  support metrics, PCA scatter, and per-subset tables. Every update is logged
  to the history JSONL (see egodiversity.history). Random subsets use seed 0
  for A and seed 1 for B, so "random vs random" compares two real draws. The
  "Rescore on Modal" button spawns a remote recompute and polls its status
  every RESCORE_POLL_MS while the call is in flight.
- Spot check: picks the most-similar and most-distant episode pair inside
  subset A and shows their contact sheets side by side, so the score can be
  eyeballed against the raw video. Subsets larger than SPOT_CHECK_CAP are
  subsampled deterministically for the pair search.
- Explore episodes: any single episode's contact sheet, 3-D hand
  trajectories with a live "current hand position" marker, and a play/pause
  frame scrubber. One server callback batch-fetches PLAYBACK_FRAMES frames +
  poses into dcc.Stores; playback itself is fully client-side (no per-frame
  server round-trips).
- 3D map: PCA(3) over all episodes, one trace per lab, subsets A/B
  highlighted.
- History: every comparison ever run, newest first, auto-refreshing.
- How it works: the four-step pipeline and validation numbers in plain
  English.

A splash preloader (app.index_string) covers the window until the Dash app
has rendered, and update_title=None keeps the browser tab from flipping to
"Updating…" during callbacks.

Run: python -m egodiversity.dashboard
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    Dash, Input, Output, State, callback, clientside_callback, ctx, dash_table,
    dcc, html, no_update,
)
from dash.exceptions import PreventUpdate
from flask import Response
from PIL import Image, ImageDraw
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from egodiversity import frames
from egodiversity.diversity import (
    coverage_radius,
    greedy_max_diversity,
    greedy_min_diversity,
    mean_pairwise_distance,
    vendi_score,
)
from egodiversity.features import load_cache
from egodiversity.history import append_history, read_history

CACHE_PATH = os.environ.get("EGODIV_CACHE", "egodiversity/cache/features.npz")
PORT = 8051

STRATEGY_OPTIONS = [
    {"label": "At random", "value": "random"},
    {"label": "Most diverse possible (greedy)", "value": "greedy max-diversity"},
    {"label": "Least diverse possible (greedy)", "value": "greedy min-diversity"},
]
STRATEGY_VALUES = {o["value"] for o in STRATEGY_OPTIONS}
GREEDY_POOL_CAP = 600    # candidate cap for greedy selectors (interactivity)
MAX_SUBSET_N = 1000      # Vendi eigendecomposition stays sub-second up to here
SPOT_CHECK_CAP = 300     # pair search subsample cap for the Spot check tab
PLAYBACK_FRAMES = 60     # frames batch-fetched per episode for the scrubber
PLAYBACK_TICK_MS = 150   # play/pause interval
RESCORE_POLL_MS = 5000   # rescore status poll interval

# Curated one-click comparisons: name -> (lab, strategy, n) for A and B.
# Entries referencing labs absent from the cache are dropped at startup.
CURATED = {
    "Random vs random (sanity check)": {
        "a": ("all", "random", 200),
        "b": ("all", "random", 200),
    },
    "Most vs least diverse 100 (greedy max vs min)": {
        "a": ("all", "greedy max-diversity", 100),
        "b": ("all", "greedy min-diversity", 100),
    },
    "mecka vs microagi (200 random each)": {
        "a": ("mecka", "random", 200),
        "b": ("microagi", "random", 200),
    },
    "Big lab vs small lab (microagi vs eth, 60 each)": {
        "a": ("microagi", "random", 60),
        "b": ("eth", "random", 60),
    },
}

# Splash preloader: white overlay + spinner, removed by the inline script as
# soon as Dash has rendered into #react-entry-point. Structure and
# placeholders mirror Dash's default index_string.
INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            #egodiv-splash {position: fixed; inset: 0; background: #fff;
                z-index: 9999; display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                font-family: sans-serif; color: #555;}
            #egodiv-splash .spinner {width: 42px; height: 42px;
                margin-bottom: 16px; border: 4px solid #ddd;
                border-top-color: #1f77b4; border-radius: 50%;
                animation: egodiv-spin 0.9s linear infinite;}
            @keyframes egodiv-spin {to {transform: rotate(360deg);}}
        </style>
    </head>
    <body>
        <div id="egodiv-splash">
            <div class="spinner"></div>
            <div>egodiversity — loading episode features…</div>
        </div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            (function () {
                var poll = setInterval(function () {
                    var root = document.getElementById("react-entry-point");
                    if (root && root.children.length > 0) {
                        var splash = document.getElementById("egodiv-splash");
                        if (splash) splash.parentNode.removeChild(splash);
                        clearInterval(poll);
                    }
                }, 100);
            })();
        </script>
    </body>
</html>"""


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


def _behavior_word(metas: list[dict], idx: list[int]) -> str:
    """'fold clothes behaviors' if the subset shares one task, else 'behaviors'."""
    tasks = {metas[i].get("task_name", "") for i in idx} - {""}
    if len(tasks) == 1:
        return f'"{tasks.pop().replace("_", " ")}" behaviors'
    return "behaviors"


def _plain_verdict(metas: list[dict], idx_a: list[int], idx_b: list[int],
                   v_a: float, v_b: float) -> str:
    """One-sentence plain-English comparison of the two Vendi scores."""
    word = _behavior_word(metas, idx_a)
    ratio = max(v_a, v_b) / max(min(v_a, v_b), 1e-9)
    if ratio < 1.1:
        return (f"Both subsets look about equally varied — ≈{v_a:.0f} vs ≈{v_b:.0f} "
                f"meaningfully different {word}.")
    if v_a >= v_b:
        return (f"Subset A's {len(idx_a)} episodes contain ≈{v_a:.0f} meaningfully "
                f"different {word} — about {ratio:.1f}× more variety than subset B "
                f"(≈{v_b:.0f}).")
    return (f"Subset B's {len(idx_b)} episodes contain ≈{v_b:.0f} meaningfully "
            f"different {word} — about {ratio:.1f}× more variety than subset A "
            f"(≈{v_a:.0f}).")


def _scores_block(name: str, X: np.ndarray, idx: list[int]) -> html.Div:
    sub = X[idx]
    return html.Div(
        [
            html.H3(name),
            html.Div(f"Vendi: {vendi_score(sub):.2f}",
                     style={"fontSize": "32px", "fontWeight": "bold"}),
            html.Div("≈ effective number of distinct behaviors",
                     style={"color": "#777", "fontSize": "12px"}),
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


def _data_uri(blob: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(blob).decode()


def _img_from_jpeg(blob: bytes, width: str = "48%") -> html.Img:
    return html.Img(src=_data_uri(blob), style={"width": width, "margin": "4px"})


def _sheet_panel(meta: dict) -> html.Div:
    """Contact sheet for one episode, or a friendly placeholder on R2 failure."""
    episode_id = meta["episode_id"]
    lab = meta.get("lab", "") or "unknown lab"
    try:
        blob = frames.contact_sheet(meta)
    except RuntimeError as exc:
        return html.Div(
            [
                html.Div(f"{episode_id} ({lab})", style={"fontSize": "12px"}),
                html.Div(f"Could not load video frames: {exc}",
                         style={"border": "1px dashed #aaa", "padding": "30px",
                                "color": "#a33", "width": "45%", "fontSize": "13px"}),
            ],
            style={"display": "inline-block", "verticalAlign": "top"},
        )
    return html.Div(
        [
            html.Div(f"{episode_id} ({lab})", style={"fontSize": "12px"}),
            _img_from_jpeg(blob),
        ],
        style={"display": "inline-block", "verticalAlign": "top"},
    )


def _placeholder_frame(text: str) -> bytes:
    """A JPEG placeholder with a message, for media failure paths."""
    img = Image.new("RGB", (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    y = 210
    for line in text.splitlines():
        draw.text((20, y), line, fill=(220, 180, 60))
        y += 18
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _def_str(d: dict) -> str:
    return f"{d.get('lab', '')} · {d.get('strategy', '')} · n={d.get('n', '')}"


def _rescore_running_msg(elapsed: int) -> str:
    return (f"Rescore running… {elapsed}s elapsed (full 12k rescore typically "
            f"takes ~4 minutes). Watch it live: modal.com/apps")


def create_app() -> Dash:
    X, metas = load_cache(CACHE_PATH)
    ids = [m["episode_id"] for m in metas]
    id_to_meta = {m["episode_id"]: m for m in metas}
    labs = sorted({m.get("lab", "") for m in metas} - {""})
    n_ep = len(X)

    # lab -> pool of indices ("all" = everything)
    pools: dict[str, list[int]] = {"all": list(range(n_ep))}
    for lab in labs:
        pools[lab] = [i for i, m in enumerate(metas) if m.get("lab") == lab]
    lab_options = ["all"] + labs

    # Curated presets whose labs exist in this cache.
    presets = {
        name: p for name, p in CURATED.items()
        if p["a"][0] in pools and p["b"][0] in pools
    }

    # Deterministic 3-D PCA over all episodes, computed once at startup; the
    # first two components also serve the 2-D Compare scatter.
    pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(X)

    slider_max = min(n_ep, MAX_SUBSET_N)

    def subset_controls(tag: str, default_strategy: str, default_lab: str) -> html.Div:
        return html.Div(
            [
                html.H3(f"Subset {tag}"),
                html.Label("Which episodes to consider (lab):"),
                dcc.Dropdown(lab_options, default_lab, id=f"lab-{tag}",
                             clearable=False),
                html.Label("How to pick them:"),
                dcc.Dropdown(STRATEGY_OPTIONS, default_strategy, id=f"strategy-{tag}",
                             clearable=False),
                html.Label("How many:"),
                dcc.Slider(2, slider_max, 1, value=15, id=f"n-{tag}",
                           marks={2: "2", slider_max: str(slider_max)}),
            ],
            style={"width": "45%", "display": "inline-block", "padding": "10px",
                   "verticalAlign": "top"},
        )

    compare_tab = html.Div(
        [
            html.Label("Curated comparisons (sets everything below at once):"),
            dcc.Dropdown(
                [{"label": k, "value": k} for k in presets],
                id="preset", placeholder="Pick a ready-made comparison…",
                style={"maxWidth": "500px"},
            ),
            html.Div([subset_controls("A", "greedy max-diversity", "all"),
                      subset_controls("B", "greedy min-diversity", "all")]),
            dcc.Loading(
                html.Div(
                    [
                        html.Div(id="scores-row"),
                        html.Div(id="verdict", style={"fontSize": "20px",
                                                      "fontWeight": "bold",
                                                      "padding": "10px"}),
                        html.Div(
                            [
                                html.Div(dcc.Graph(id="pca-scatter"),
                                         style={"width": "70%",
                                                "display": "inline-block",
                                                "verticalAlign": "top"}),
                                html.Div(
                                    dcc.Loading(html.Div(id="scatter-detail"),
                                                type="dot", delay_show=300),
                                    style={"width": "28%",
                                           "display": "inline-block",
                                           "verticalAlign": "top",
                                           "padding": "6px"}),
                            ]
                        ),
                        html.Div([html.Div(id="table-A",
                                           style={"width": "45%",
                                                  "display": "inline-block"}),
                                  html.Div(id="table-B",
                                           style={"width": "45%",
                                                  "display": "inline-block",
                                                  "float": "right"})]),
                    ]
                ),
                type="circle",
            ),
            html.Hr(),
            html.Button("Rescore on Modal", id="rescore-btn", n_clicks=0),
            dcc.Store(id="rescore-call"),
            dcc.Interval(id="rescore-tick", interval=RESCORE_POLL_MS, disabled=True),
            html.Div(id="rescore-out", style={"padding": "10px", "color": "#555"}),
        ]
    )

    spot_tab = html.Div(
        [
            html.P("Does the score agree with your eyes? This picks the two "
                   "episodes of subset A that our features call most similar, "
                   "and the two it calls most different, and shows you the raw "
                   "frames. Subset A is whatever you last configured on the "
                   "Compare tab."),
            html.Button("Find proof pairs", id="spot-btn", n_clicks=0),
            dcc.Loading(html.Div(id="spot-results", style={"padding": "10px"}),
                        type="circle"),
        ]
    )

    explore_tab = html.Div(
        [
            html.Label("Search for an episode (type to filter):"),
            dcc.Dropdown(
                [{"label": f"{m['episode_id']} · {m.get('lab', '') or 'unknown lab'}",
                  "value": m["episode_id"]} for m in metas],
                id="explore-episode", placeholder="Pick an episode…",
                style={"maxWidth": "600px"},
            ),
            dcc.Store(id="explore-frames"),
            dcc.Store(id="explore-poses"),
            dcc.Store(id="frame-idx", data=0),
            dcc.Loading(
                html.Div(
                    [
                        html.Div(id="explore-info",
                                 style={"padding": "6px", "color": "#555"}),
                        html.Div(id="explore-sheet"),
                        dcc.Graph(id="explore-traj"),
                        html.Hr(),
                        html.Div(
                            [
                                html.Button("Play", id="play-btn", n_clicks=0),
                                dcc.Interval(id="play-tick",
                                             interval=PLAYBACK_TICK_MS,
                                             disabled=True),
                                dcc.Slider(0, 1, 1, value=0, id="scrub",
                                           marks={0: "frame 0"},
                                           tooltip={"placement": "bottom"}),
                                html.Img(id="scrub-img",
                                         style={"maxWidth": "640px",
                                                "display": "block"}),
                            ]
                        ),
                    ]
                ),
                type="circle",
            ),
        ]
    )

    map_tab = html.Div(
        [
            html.P("Every episode as one point (PCA of the motion-shape "
                   "features, 3 components). Close points = similar movement. "
                   "Subsets A and B from the Compare tab are highlighted. "
                   "Hover a point to preview the episode; click it for the "
                   "full contact sheet."),
            html.Div(
                [
                    html.Div(
                        dcc.Loading(dcc.Graph(id="map-3d",
                                              style={"height": "75vh"}),
                                    type="circle"),
                        style={"width": "70%", "display": "inline-block",
                               "verticalAlign": "top"},
                    ),
                    html.Div(
                        dcc.Loading(
                            html.Div(id="map-detail",
                                     children="Hover a point to preview the "
                                              "episode.",
                                     style={"color": "#777",
                                            "padding": "6px"}),
                            type="dot", delay_show=300,
                        ),
                        style={"width": "28%", "display": "inline-block",
                               "verticalAlign": "top", "padding": "6px"},
                    ),
                ]
            ),
        ]
    )

    history_tab = html.Div(
        [
            html.P("Every comparison run on the Compare tab is logged here."),
            dcc.Interval(id="history-tick", interval=5000),
            html.Div(id="history-content"),
        ]
    )

    how_tab = html.Div(
        [
            html.H3("How the score works"),
            html.Ol(
                [
                    html.Li([
                        html.B("Trajectories → fixed-length motion-shape vectors. "),
                        "For every episode we take the recorded hand and head "
                        "positions over time (no video, no text), resample them "
                        "to a fixed length, and normalize away where in the room "
                        "the motion happened. Each of the "
                        f"{n_ep:,} episodes becomes one 741-number vector "
                        "describing the shape of its movement.",
                    ]),
                    html.Li([
                        html.B("Similarity kernel. "),
                        "We compare every pair of vectors with an RBF (Gaussian) "
                        "kernel: 1.0 means identical movement, 0 means nothing in "
                        "common. The bandwidth is set by the median heuristic — "
                        "rankings stay stable across a 16× range around it.",
                    ]),
                    html.Li([
                        html.B("Vendi score = effective number of distinct behaviors. "),
                        "The eigenvalue spectrum of the similarity matrix, run "
                        "through Shannon entropy and exponentiated. 100 copies of "
                        "the same motion score ≈1; 100 genuinely different motions "
                        "score ≈100.",
                    ]),
                    html.Li([
                        html.B("Validated, not just plausible. "),
                        "Near-duplicate episodes score 3.25 vs 4.67 for truly "
                        "distinct ones; greedily-picked least-diverse < random < "
                        "most-diverse subsets score 1.64 < 5.11 < 5.19; rankings "
                        "hold across a 16× kernel-bandwidth range; and the score "
                        "saturates when fed redundant data, exactly as it should.",
                    ]),
                ]
            ),
            html.H3("Limits"),
            html.P("The features capture the shape of the motion, not the scene "
                   "content — two episodes folding different shirts in the same "
                   "way look identical to this score. Visual embeddings "
                   "(learned from the video frames themselves) are the designed "
                   "extension and plug into the same kernel + Vendi machinery."),
        ],
        style={"maxWidth": "800px"},
    )

    app = Dash(__name__, title="egodiversity", update_title=None)
    app.index_string = INDEX_STRING
    app.layout = html.Div(
        [
            html.H1("egodiversity"),
            html.P("How many genuinely different behaviors are in a robot "
                   "dataset? This tool turns any subset of EgoVerse episodes "
                   "into a single number — the effective count of distinct "
                   "motion behaviors — so you can prove one subset has more "
                   "variety than another, and check the answer with your own "
                   "eyes."),
            html.Div(f"{n_ep:,} episodes · {X.shape[1]} features per episode "
                     f"· cache: {CACHE_PATH}", style={"color": "#777"}),
            dcc.Store(id="store-subsets"),
            dcc.Tabs(
                [
                    dcc.Tab(compare_tab, label="Compare", value="tab-compare"),
                    dcc.Tab(spot_tab, label="Spot check", value="tab-spot"),
                    dcc.Tab(explore_tab, label="Explore episodes",
                            value="tab-explore"),
                    dcc.Tab(map_tab, label="3D map", value="tab-map"),
                    dcc.Tab(history_tab, label="History", value="tab-history"),
                    dcc.Tab(how_tab, label="How it works", value="tab-how"),
                ],
                value="tab-compare",
            ),
        ],
        style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "sans-serif"},
    )

    # ------------------------------------------------------------------ frame
    @app.server.route("/frame/<episode_id>/<int:idx>")
    def serve_frame(episode_id: str, idx: int):
        """One JPEG frame of an episode; a placeholder image on any failure.

        Kept for deep links; the Explore scrubber no longer uses it (playback
        is client-side from a batch-fetched store).
        """
        meta = id_to_meta.get(episode_id)
        blob = None
        if meta is not None:
            try:
                blob = frames.get_frames(meta, [idx])[0]
            except Exception:
                blob = None
        if blob is None:
            blob = _placeholder_frame(f"frame unavailable\n{episode_id} #{idx}")
        return Response(blob, mimetype="image/jpeg")

    # ---------------------------------------------------------------- compare
    @callback(
        Output("lab-A", "value"), Output("strategy-A", "value"), Output("n-A", "value"),
        Output("lab-B", "value"), Output("strategy-B", "value"), Output("n-B", "value"),
        Output("preset", "value"),
        Input("preset", "value"),
    )
    def apply_preset(name):
        if not name or name not in presets:
            raise PreventUpdate
        p = presets[name]
        return (*p["a"], *p["b"], None)

    @callback(
        Output("scores-row", "children"),
        Output("verdict", "children"),
        Output("pca-scatter", "figure"),
        Output("table-A", "children"),
        Output("table-B", "children"),
        Output("store-subsets", "data"),
        Input("lab-A", "value"), Input("strategy-A", "value"), Input("n-A", "value"),
        Input("lab-B", "value"), Input("strategy-B", "value"), Input("n-B", "value"),
    )
    def update(lab_a, strat_a, n_a, lab_b, strat_b, n_b):
        idx_a = _select_subset(X, strat_a, n_a, pools[lab_a], seed=0)
        idx_b = _select_subset(X, strat_b, n_b, pools[lab_b], seed=1)
        v_a, v_b = vendi_score(X[idx_a]), vendi_score(X[idx_b])
        verdict = _plain_verdict(metas, idx_a, idx_b, v_a, v_b)

        try:
            append_history({"lab": lab_a, "strategy": strat_a, "n": len(idx_a)},
                           {"lab": lab_b, "strategy": strat_b, "n": len(idx_b)},
                           v_a, v_b, verdict)
        except Exception:
            pass  # logging must never break the UI

        name_a, name_b = f"A ({lab_a}, {strat_a})", f"B ({lab_b}, {strat_b})"
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
            custom_data=["episode_id", "lab"],
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
            {"a": [int(i) for i in idx_a], "b": [int(i) for i in idx_b]},
        )

    @callback(
        Output("rescore-out", "children"),
        Output("rescore-call", "data"),
        Output("rescore-tick", "disabled"),
        Input("rescore-btn", "n_clicks"),
    )
    def rescore(n_clicks):
        if not n_clicks:
            return "", None, True
        if os.environ.get("EGODIV_MODAL") != "1":
            return (("Modal not configured. Set EGODIV_MODAL=1 (and run "
                     "`modal setup`) to enable remote feature extraction on "
                     "Modal/R2."), None, True)
        try:
            from egodiversity.modal_app import rescore_remote

            # Build manifest dicts straight from the loaded cache metadata
            # (remote-built caches carry path/lab/num_frames); fall back to the
            # aria prefix for local-only metas.
            episodes = [
                {"episode_hash": m["episode_id"], "lab": m.get("lab", ""),
                 "task": m.get("task_name", ""), "num_frames": m.get("num_frames", -1),
                 "path": m.get("path")
                 or f"s3://rldb/processed_v3/aria/{m['episode_id']}.zarr"}
                for m in metas
            ]
            call_id = rescore_remote(episodes)
        except Exception as exc:  # demo hook — surface, don't crash
            return f"Modal rescore failed: {exc}", None, True
        return (_rescore_running_msg(0),
                {"call_id": call_id, "t0": time.time()}, False)

    @callback(
        Output("rescore-out", "children", allow_duplicate=True),
        Output("rescore-call", "data", allow_duplicate=True),
        Output("rescore-tick", "disabled", allow_duplicate=True),
        Input("rescore-tick", "n_intervals"),
        State("rescore-call", "data"),
        prevent_initial_call=True,
    )
    def poll_rescore(_n, data):
        if not data or not data.get("call_id"):
            raise PreventUpdate
        elapsed = int(time.time() - data.get("t0", time.time()))
        try:
            import modal  # lazy: local dashboard works without modal auth

            result = modal.FunctionCall.from_id(data["call_id"]).get(timeout=0)
        except TimeoutError:
            # builtin TimeoutError — what FunctionCall.get(timeout=0) raises
            # while the call is still pending (verified on modal 1.5.4).
            return _rescore_running_msg(elapsed), no_update, False
        except Exception as exc:
            if type(exc).__name__ == "TimeoutError":
                # modal.exception.TimeoutError on other modal versions
                return _rescore_running_msg(elapsed), no_update, False
            return f"Rescore failed: {exc}", None, True
        return (f"{result} — done in {elapsed}s — scores refresh on the "
                f"dashboard's next cold start."), None, True

    # -------------------------------------------------------------- spot check
    @callback(
        Output("spot-results", "children"),
        Input("spot-btn", "n_clicks"),
        State("store-subsets", "data"),
    )
    def spot_check(n_clicks, data):
        if not n_clicks:
            return "Configure subset A on the Compare tab, then press the button."
        idx = list((data or {}).get("a") or [])
        if len(idx) < 2:
            return "Subset A has fewer than 2 episodes — nothing to compare."

        note = ""
        if len(idx) > SPOT_CHECK_CAP:
            rng = np.random.default_rng(0)
            idx = sorted(rng.choice(idx, SPOT_CHECK_CAP, replace=False).tolist())
            note = (f"Subset A has more than {SPOT_CHECK_CAP} episodes, so the "
                    f"pair search ran on a deterministic subsample of "
                    f"{SPOT_CHECK_CAP}.")

        Zs = StandardScaler().fit_transform(X[idx])
        D = squareform(pdist(Zs, metric="euclidean"))
        D_off = D.copy()
        np.fill_diagonal(D_off, np.inf)
        i, j = (int(t) for t in np.unravel_index(np.argmin(D_off), D_off.shape))
        k, l = (int(t) for t in np.unravel_index(np.argmax(D), D.shape))

        def pair_block(word: str, p: int, q: int, dist: float) -> html.Div:
            return html.Div(
                [
                    html.H4(f"Our score calls these {word} — check for yourself:"),
                    html.Div(f"standardized feature distance: {dist:.2f}",
                             style={"color": "#777"}),
                    html.Div([_sheet_panel(metas[idx[p]]),
                              _sheet_panel(metas[idx[q]])]),
                ],
                style={"padding": "10px 0"},
            )

        return html.Div(
            [
                html.Div(note, style={"color": "#a60"}) if note else html.Div(),
                pair_block("NEARLY IDENTICAL", i, j, float(D[i, j])),
                html.Hr(),
                pair_block("HIGHLY DISTINCT", k, l, float(D[k, l])),
            ]
        )

    # ----------------------------------------------------------------- explore
    @callback(
        Output("explore-sheet", "children"),
        Output("explore-traj", "figure"),
        Output("explore-info", "children"),
        Output("scrub", "max"),
        Output("scrub", "value"),
        Output("explore-frames", "data"),
        Output("explore-poses", "data"),
        Input("explore-episode", "value"),
    )
    def explore(episode_id):
        if not episode_id:
            return ("", go.Figure(),
                    "No episode selected — pick one above to load its video "
                    "and trajectories.", 1, 0, [], {})
        meta = id_to_meta[episode_id]

        sheet = _sheet_panel(meta)

        fig = go.Figure()
        poses_lists: dict[str, list] = {}
        traj_note = ""
        try:
            poses = frames.get_poses(meta)
            poses_lists = {h: poses[h].tolist() for h in ("right", "left")}
            for hand, color in (("right", "#d62728"), ("left", "#1f77b4")):
                p = poses[hand]
                fig.add_trace(go.Scatter3d(
                    x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                    name=f"{hand} hand", line={"color": color, "width": 4},
                ))
            # Current-position markers, moved client-side during playback.
            # They must be the LAST TWO traces (the clientside callback
            # rewrites them by position).
            for hand, color in (("right", "#ff7f0e"), ("left", "#2ca02c")):
                p = poses[hand]
                fig.add_trace(go.Scatter3d(
                    x=[p[0, 0]], y=[p[0, 1]], z=[p[0, 2]], mode="markers",
                    name=f"{hand} hand (current)",
                    marker={"size": 6, "color": color},
                ))
            fig.update_layout(title="Hand trajectories (end-effector xyz)",
                              height=450,
                              scene={"xaxis_title": "x", "yaxis_title": "y",
                                     "zaxis_title": "z"})
        except RuntimeError as exc:
            traj_note = f" Trajectories unavailable: {exc}"

        # Batch-fetch PLAYBACK_FRAMES evenly spaced frames for client-side
        # playback (get_frames is threaded internally).
        frames_note = ""
        try:
            total = frames.num_frames(meta)
            n_fetch = min(PLAYBACK_FRAMES, total)
            idxs = np.linspace(0, total - 1, n_fetch).astype(int).tolist()
            uris = [_data_uri(b) for b in frames.get_frames(meta, idxs)]
        except RuntimeError as exc:
            total = int(meta.get("num_frames") or 0)
            frames_note = f" Frames unavailable: {exc}"
            uris = [_data_uri(_placeholder_frame(
                f"frames unavailable\n{episode_id}"))]

        info = (f"{episode_id} · {meta.get('lab', '') or 'unknown lab'} · "
                f"{total} source frames · {len(uris)} playback frames."
                f"{traj_note}{frames_note}")
        return sheet, fig, info, max(len(uris) - 1, 0), 0, uris, poses_lists

    @callback(
        Output("play-tick", "disabled"),
        Output("play-btn", "children"),
        Input("play-btn", "n_clicks"),
        State("play-tick", "disabled"),
    )
    def toggle_play(n_clicks, disabled):
        if not n_clicks:
            raise PreventUpdate
        return (not disabled), ("Pause" if disabled else "Play")

    # Client-side playback: no server round-trips while playing.
    clientside_callback(
        """
        function(n, idx, frames) {
            if (!frames || !frames.length) return window.dash_clientside.no_update;
            var i = (idx == null) ? 0 : idx;
            return (i + 1) % frames.length;
        }
        """,
        Output("frame-idx", "data"),
        Input("play-tick", "n_intervals"),
        State("frame-idx", "data"),
        State("explore-frames", "data"),
    )

    clientside_callback(
        "function(v) { return (v == null) ? 0 : v; }",
        Output("frame-idx", "data", allow_duplicate=True),
        Input("scrub", "value"),
        prevent_initial_call=True,
    )

    clientside_callback(
        """
        function(idx, frames) {
            var nu = window.dash_clientside.no_update;
            if (!frames || !frames.length) return [nu, nu];
            var i = (idx == null) ? 0 : idx;
            if (i >= frames.length) i = frames.length - 1;
            return [frames[i], i];
        }
        """,
        Output("scrub-img", "src"),
        Output("scrub", "value", allow_duplicate=True),
        Input("frame-idx", "data"),
        State("explore-frames", "data"),
        prevent_initial_call=True,
    )

    clientside_callback(
        """
        function(idx, poses, frames, fig) {
            if (!poses || !poses.right || !poses.right.length ||
                    !fig || !fig.data || fig.data.length < 4) {
                return window.dash_clientside.no_update;
            }
            var nF = (frames && frames.length) ? frames.length : 1;
            var nP = poses.right.length;
            var i = (idx == null) ? 0 : idx;
            var pi = Math.round((nF > 1 ? i / (nF - 1) : 0) * (nP - 1));
            var pts = [poses.right[pi], poses.left[pi]];
            var out = Object.assign({}, fig);
            var data = fig.data.slice();
            for (var k = 0; k < 2; k++) {
                var t = Object.assign({}, fig.data[fig.data.length - 2 + k]);
                t.x = [pts[k][0]]; t.y = [pts[k][1]]; t.z = [pts[k][2]];
                data[fig.data.length - 2 + k] = t;
            }
            out.data = data;
            return out;
        }
        """,
        Output("explore-traj", "figure", allow_duplicate=True),
        Input("frame-idx", "data"),
        State("explore-poses", "data"),
        State("explore-frames", "data"),
        State("explore-traj", "figure"),
        prevent_initial_call=True,
    )

    # -------------------------------------------------------------------- map
    @callback(Output("map-3d", "figure"), Input("store-subsets", "data"))
    def update_map(data):
        fig = go.Figure()
        by_lab: dict[str, list[int]] = {}
        for i, m in enumerate(metas):
            by_lab.setdefault(m.get("lab", "") or "unknown", []).append(i)
        palette = px.colors.qualitative.Plotly
        for c, (lab, idx) in enumerate(sorted(by_lab.items())):
            idx = np.array(idx)
            fig.add_trace(go.Scatter3d(
                x=Z[idx, 0], y=Z[idx, 1], z=Z[idx, 2], mode="markers",
                name=f"{lab} ({len(idx)})",
                marker={"size": 2, "opacity": 0.45,
                        "color": palette[c % len(palette)]},
                text=[f"{ids[i]}<br>{lab}" for i in idx], hoverinfo="text",
                customdata=[[ids[i], lab] for i in idx],
            ))
        for key, name, color in (("a", "Subset A", "#d62728"),
                                 ("b", "Subset B", "#1f77b4")):
            idx = (data or {}).get(key) or []
            if not idx:
                continue
            idx = np.array(idx)
            fig.add_trace(go.Scatter3d(
                x=Z[idx, 0], y=Z[idx, 1], z=Z[idx, 2], mode="markers",
                name=f"{name} ({len(idx)})",
                marker={"size": 5, "opacity": 0.95, "color": color,
                        "line": {"width": 1, "color": "black"}},
                text=[f"{ids[i]}<br>{metas[i].get('lab', '') or 'unknown'}<br>{name}"
                      for i in idx],
                hoverinfo="text",
                customdata=[[ids[i], metas[i].get("lab", "") or "unknown"]
                            for i in idx],
            ))
        fig.update_layout(title="All episodes in motion-shape space (PCA, 3D)",
                          legend={"itemsizing": "constant"})
        return fig

    # -------------------------------------------------------- episode previews
    def _episode_detail(episode_id: str, with_sheet: bool,
                        compact: bool) -> html.Div:
        """Thumbnail + metadata panel for one episode (contact sheet too when
        with_sheet). RuntimeError (R2 failure) -> friendly placeholder."""
        meta = id_to_meta.get(episode_id)
        if meta is None:
            return html.Div(f"unknown episode: {episode_id}",
                            style={"color": "#a33"})
        try:
            thumb = frames.get_thumbnail(meta)
            thumb_el: html.Div = html.Img(
                src=_data_uri(thumb),
                style={"width": "240px" if compact else "100%"})
        except RuntimeError as exc:
            thumb_el = html.Div(
                f"preview unavailable: {exc}",
                style={"border": "1px dashed #aaa", "padding": "20px",
                       "color": "#a33", "fontSize": "13px"})
        children = [
            thumb_el,
            html.Div(episode_id, style={"fontFamily": "monospace",
                                        "fontSize": "11px",
                                        "wordBreak": "break-all"}),
            html.Div(
                f"{meta.get('lab', '') or 'unknown lab'} · "
                f"{meta.get('task_name', '') or 'unknown task'} · "
                f"{meta.get('num_frames', '?')} frames",
                style={"fontSize": "12px", "color": "#555"}),
        ]
        if with_sheet:
            try:
                children.append(html.Img(src=_data_uri(frames.contact_sheet(meta)),
                                         style={"width": "100%",
                                                "marginTop": "6px"}))
            except RuntimeError as exc:
                children.append(html.Div(f"contact sheet unavailable: {exc}",
                                         style={"color": "#a33",
                                                "fontSize": "12px"}))
        return html.Div(children)

    @callback(
        Output("map-detail", "children"),
        Input("map-3d", "hoverData"), Input("map-3d", "clickData"),
    )
    def map_detail(hover, click):
        # Most recent event wins; click wins when both fire at once. Unhover
        # leaves the last preview in place (clear_on_unhover stays False).
        prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[-1] if ctx.triggered else ""
        event = click if (prop == "clickData" and click) else (hover or click)
        if not event or not event.get("points"):
            raise PreventUpdate
        cd = event["points"][0].get("customdata")
        if not cd:
            raise PreventUpdate
        return _episode_detail(cd[0], with_sheet=(prop == "clickData"),
                               compact=False)

    @callback(
        Output("scatter-detail", "children"),
        Input("pca-scatter", "hoverData"), Input("pca-scatter", "clickData"),
    )
    def scatter_detail(hover, click):
        prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[-1] if ctx.triggered else ""
        event = click if (prop == "clickData" and click) else (hover or click)
        if not event or not event.get("points"):
            raise PreventUpdate
        cd = event["points"][0].get("customdata")
        if not cd:
            raise PreventUpdate
        return _episode_detail(cd[0], with_sheet=False, compact=True)

    # ---------------------------------------------------------------- history
    @callback(Output("history-content", "children"),
              Input("history-tick", "n_intervals"))
    def show_history(_n):
        entries = read_history()
        if not entries:
            return html.Div(
                "No comparisons logged yet — run one on the Compare tab.",
                style={"color": "#777", "padding": "20px"},
            )
        rows = [
            {
                "time (utc)": e.get("ts", "").replace("T", " ").replace("+00:00", "Z"),
                "subset A": _def_str(e.get("a", {})),
                "subset B": _def_str(e.get("b", {})),
                "vendi A": e.get("vendi_a", ""),
                "vendi B": e.get("vendi_b", ""),
                "verdict": e.get("verdict", ""),
            }
            for e in reversed(entries)
        ]
        return dash_table.DataTable(
            rows,
            columns=[{"name": c, "id": c} for c in rows[0]],
            page_size=15,
            style_cell={"fontSize": "12px", "textAlign": "left",
                        "whiteSpace": "normal", "height": "auto"},
            style_table={"overflowX": "auto"},
        )

    return app


if __name__ == "__main__":
    create_app().run(port=PORT, debug=False)
