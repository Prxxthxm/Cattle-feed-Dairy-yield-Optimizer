# Feed & Yield Optimizer Dashboard

Interactive dashboard wiring together the LP ration formulator, the ML yield
stack, and the ICAR fallback into one pipeline: move the sliders, click
Generate, get a real LP-solved ration, a real (ML-or-ICAR) yield prediction,
profit, and a SHAP-style explanation.

## Files

```
dashboard/
├── backend.py            Flask API + server entry point
├── model_core.py          all pipeline logic (LP solve, ML stack, SHAP, ICAR fallback)
├── requirements.txt
├── data/
│   ├── Dairy_Data.csv          ML training data (INRAE/CowNflow)
│   ├── nutri_matrix_guj.csv    regional ingredient nutrient matrix
│   ├── Parent Nutrient Matrix.csv    base ingredient nutrient matrix
│   └── animal_profiles.csv     breed presets
├── static/
│   └── index.html         the dashboard UI (served by Flask, calls the API)
└── cache/                 auto-created; stores the trained model so restarts are instant
```

## How to run

1. Put all of the above in one folder.
2. Install dependencies (Python 3.9+ recommended):
   ```
   pip install -r requirements.txt
   ```
3. Start the server:
   ```
   python backend.py
   ```
   First launch trains the ML yield stack once (~5-10 seconds) and caches it
   to `cache/yield_stack.pkl`. Every launch after that is near-instant.
4. Open **http://127.0.0.1:5050** in a browser.

That's it — no separate frontend build step. Flask serves `static/index.html`
directly, and it talks to the same process's `/api/...` routes.

## What each slider control does

- **Breed preset**: fills body weight, fat %, animal type, and target yield
  from `animal_profiles.csv` (HF crossbreed, Gir, Jersey crossbreed, Kankrej,
  Sahiwal). Switch back to "Custom" to move sliders independently.
- **Body weight / Milk fat % / Animal type**: feed the ICAR 2013 requirement
  equations, which set the LP's DMI/CP/ME/Ca/P/NDF targets.
- **Diet type (ML feature)**: one of the six diet categories in
  `Dairy_Data.csv` (Maize, Fresh_herbage, Maize_Lucerne, Dehydrated_herbage,
  Maize_Fresh_herbage, Maize_Hay). This is a categorical feature the ML model
  was trained on; it does not change the LP's ingredient choices, only which
  ML training subpopulation the prediction leans on.
- **Target milk yield**: what the LP solves the ration *for* (the requirement
  it's built to sustain).
- **Milk price**: used only to convert predicted yield into revenue and
  profit; doesn't affect the ration itself.

## How yield is actually predicted

For every Generate click:
1. The LP solves the least-cost ration meeting ICAR requirements for the
   target yield, given body weight/fat%/animal type.
2. That ration's real nutritional profile (DMI, CP intake, concentrate
   proportion, digestibility, etc.) is checked against the ML model's
   training envelope (BW 430–907 kg, DMI 8–29.6 kg/day, yield 5.5–47 kg/day).
3. **In envelope** → the trained Ridge/Lasso/XGB/LGBM → Ridge-meta stack
   predicts actual yield from the ration profile, and a SHAP-style
   feature-contribution breakdown is shown.
4. **Out of envelope** → falls back to the ICAR view: since the ration was
   solved to meet the requirement for the target yield, yield = target by
   construction (no extrapolation, no meta-learner cancellation issue). The
   UI does not label which path was used, per the original spec — it just
   shows the numbers. (The SHAP panel will say no ML explanation applies for
   ICAR-fallback cases, since there's no ML prediction to explain there.)

This matches the diagnosis behind `Optimisation_Engine_v2_POC.py` (ICAR,
numerically trustworthy) vs `Optimisation_Engine_v3.py` (ML architecture
demo, flagged unreliable off-distribution) — the dashboard picks whichever
is valid for the current slider combination automatically.

## Notes on the LP

Any slider combination is solved fresh — there's no lookup table. If a
combination is infeasible (e.g. very high body weight with a very high
target yield, given the fixed ingredient set's inclusion bounds), the API
returns `feasible: false` with a message rather than crashing; the UI shows
this as an error box instead of numbers.

## Extending

- Swap in a different ingredient matrix: replace `data/nutri_matrix_guj.csv`
  (same column schema) and restart.
- Adjust the ML/ICAR envelope thresholds: `ML_ENVELOPE` in `model_core.py`.
- Retrain from scratch (e.g. after editing `Dairy_Data.csv`): delete
  `cache/yield_stack.pkl` and restart.
