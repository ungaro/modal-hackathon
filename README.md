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
effective distinct behaviors** — and per-episode diversity is nearly identical
across labs, so scale buys coverage, not variety.

See [`egodiversity/README.md`](egodiversity/README.md) for quickstart, design
decisions, and the validation report. EgoVerse dataset:
https://github.com/GaTech-RL2/EgoVerse/
