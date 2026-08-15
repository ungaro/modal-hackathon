# Ego Trip — demo script (~5 min)

Deck: `presentation.html` (open in any browser, arrow keys, works offline).
Dashboard: https://alp-guneysel--egodiversity-dashboard.modal.run
Modal console: https://modal.com/apps (have it open in a second tab, logged in)

## 0:00–0:30 — Hook (slide 2)
"Robot teams are collecting hundreds of thousands of egocentric videos. But
nobody can say how much of it is redundant — the current answer is LLM
judges, which are slow, expensive, and subjective. We built a number instead."

## 0:30–1:15 — Method (slide 3)
"Every episode becomes 741 numbers describing the SHAPE of the motion —
hands and head, resampled so duration doesn't matter, normalized so the
table and room don't matter. Same folding motion at a different table → same
vector. Then one similarity matrix, and the Vendi score reads it as 'the
effective number of distinct behaviors.' And it's a validated instrument —
duplicates score lower, ordering is correct, bandwidth doesn't matter."

## 1:15–1:45 — Finding (slide 4)
"We ran the first audit of EgoVerse: 12,212 fold_clothes episodes across six
labs contain about FOURTEEN effective distinct behaviors. After any one lab,
every other lab adds essentially zero new behaviors. Scale buys coverage,
not variety."

## 1:45–3:45 — Live demo (slide 5)
Do these in order; each is one or two clicks:

1. **Compare tab → curated dropdown → "Does 20× data mean 20× diversity?"**
   Wait ~2 s. Read the verdict aloud: 500 episodes are only ~1.5× more
   diverse than 25. "That's the redundancy claim, computed live."
2. **Spot check tab → "Find proof pairs"** (click twice — fresh draw each
   click). "These are the two episodes our metric calls nearly identical —
   and the two it calls most distinct. Judge with your own eyes." Let the
   videos play.
3. **Explore tab → pick any episode.** Video plays; the 3D hand trajectories
   next to it have markers tracking the video clock.
4. **Rescore on Modal** → click, then switch to the Modal console tab.
   "One click inside a web page fans out to a hundred containers." Watch the
   `extract_remote` row spin up. Switch back — the dashboard polls and shows
   live status. (You don't need to wait for it to finish.)

If the venue wifi dies: the dashboard is a Modal deployment and keeps
running; tether your phone. Worst case, the slide.png screenshots in the
repo show every screen.

## 3:45–4:30 — Modal (slide 6)
"The whole pipeline is cloud-to-cloud. 12.5k episodes in 3.7 minutes —
one `.map()`. The dashboard is a deployed Modal web endpoint; we redeployed
16 times today, about 2 seconds each. The infrastructure is one Python file
— no Docker, no YAML, no IAM."

## 4:30–5:00 — Close (slides 7–8)
"This is a measurement today; the value is what it enables: keep the k
episodes that maximize diversity and train on less; use the marginal curve
to decide when to stop collecting a task. And it works on your data — drop a
features.npz on the page. Honest limit: we see motion shape, not scene
content — visual embeddings on GPUs are the designed next step."

## Likely questions
- **"Is the metric yours?"** No — Vendi score (Friedman & Dieng), FAKTUAL
  applied it to robot data. Ours: the first EgoVerse audit, the kinematic
  representation, marginal scoring, and the operationalization. (README has
  a related-work section.)
- **"Why not just farthest-point sampling?"** In 741-dim space distances
  concentrate; farthest-point scores no better than random under Vendi.
  That's why the selector optimizes Vendi directly.
- **"What about different objects/scenes with the same motion?"** Deliberate
  scope: kinematics capture motion shape. Visual embeddings plug into the
  same kernel + Vendi machinery.
- **"Can I try it?"** The URL is live; upload box takes any features.npz.
