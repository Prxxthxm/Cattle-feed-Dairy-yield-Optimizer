# Cattle Feed Optimization & Dairy Yield Forecasting

A decision support system for Indian dairy farmers, combining a linear-programming-based
least-cost feed formulator with a machine-learning milk yield predictor, integrated into
a live interactive dashboard.

Built during Practice School - I at CDAC CINE, Silchar (BITS Pilani), May–July 2026.

## What's here

- **LP ration formulator** (`Optimisation_Engine_v1.ipynb`) — PuLP/CBC solver, ICAR 2013
  nutrient requirement tables, 19-ingredient Gujarat/western-Maharashtra nutrient matrix,
  five cattle breed profiles.
- **ML yield predictor** (`Milk_Yield_Predictor.ipynb`) — stacked ensemble
  (Ridge/Lasso/XGBoost/LightGBM → Ridge meta-learner) trained on the CowNflow/INRAE
  dataset. OOF RMSE 2.79 kg/day, R² = 0.88.
- **Integration** (`Optimisation_Engine_v2_POC.py`, `Optimisation_Engine_v3.py`,
  `profit_optimizer.py`) — parametric profit-frontier sweep linking the two modules,
  plus a distribution-aware fallback that resolves a population mismatch between the
  ML model's training data and the rations the LP produces (see `dashboard/README.md`
  for details).
- **Dashboard** (`dashboard/`) — Flask backend + self-contained HTML/CSS/JS frontend.
  Solves the LP and runs the yield stack live for any input combination, with a
  chain-rule SHAP explanation panel and rule-based feedback.

## Quick start (dashboard)

```bash
cd dashboard
pip install -r requirements.txt
python backend.py
```
Then open `http://127.0.0.1:5050`. First launch trains the ML stack once (~5-10s,
cached after).

## Data

- `Dairy_Data.csv` — CowNflow dataset (INRAE), 414 Holstein cow-period records, used
  to train the yield model.
- `nutri_matrix_guj.csv` — regional feed ingredient nutrient matrix (Gujarat / western
  Maharashtra).
- `animal_profiles.csv` — five ICAR-derived cattle breed profiles used as presets.

## Notes

- The ML model's training population (high-input, 430-907 kg BW) differs from the
  ICAR-based ration space this LP targets (350-450 kg BW). The dashboard checks each
  ration's profile against the ML model's training envelope and falls back to an
  ICAR response-curve yield estimate when out of range — see `dashboard/README.md`.
- Ingredient costs are indicative (cross-checked against IndiaMart Ahmedabad listings)
  and should be refreshed for real deployment use.
