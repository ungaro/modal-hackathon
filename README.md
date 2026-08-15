# EgoVerse Data Optimization & Evaluation Suite — Track 2: Quantitative Diversity Measurement

![summary slide](slide.png)

**Live dashboard:** https://alp-guneysel--egodiversity-dashboard.modal.run
**Summary slide:** [`slide.html`](slide.html) (open in a browser)

How diverse is an episode subset, really? This project answers it with a
number, not an LLM judge: kinematic trajectory features → RBF similarity
kernel → **Vendi score** (effective number of distinct behaviors). All
compute runs serverlessly on [Modal](https://modal.com) against the EgoVerse
R2 bucket; the dashboard is a deployed Modal web endpoint.

Headline result: 12,212 `fold_clothes` episodes across 6 labs ≈ **~14
effective distinct behaviors**. Marginal analysis: after any one lab's
episodes, the other labs add ≈0 new behaviors (ΔVendi −0.3…+0.8) — every
source captures the same motions, so scale buys coverage, not variety.

The dashboard lets you check the score with your own eyes: hover any point on
the 3D map or compare scatter to see the episode's actual video frame, play
the full preview video with hand trajectories synced to the video clock, or
open the **Spot check** tab to see the most-similar and most-distinct episode
pairs as contact sheets + inline videos. Plus curated one-click comparisons,
a comparison history log, and a drop zone to **upload your own dataset** (any
`features.npz` — see the package README for the format).

Metric credit: Vendi score ([Friedman & Dieng 2023](https://arxiv.org/abs/2210.02410));
[FAKTUAL](https://arxiv.org/html/2603.11634v1) applies it to robotics datasets
with signature kernels. New here: the first audit of EgoVerse, marginal-diversity
scoring for collection planning, and a serverless pipeline that makes
re-auditing a ~4-minute job.

See [`egodiversity/README.md`](egodiversity/README.md) for quickstart, design
decisions, and the validation report. EgoVerse dataset:
https://github.com/GaTech-RL2/EgoVerse/
