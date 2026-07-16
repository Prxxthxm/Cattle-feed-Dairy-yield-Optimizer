"""
Core pipeline for the Cattle Feed Optimization dashboard.

Wires together, exactly as in Optimisation_Engine_v1 / profit_optimizer.py / v2_POC / v3:
  1. LP least-cost ration formulator (PuLP/CBC, ICAR 2013 requirement tables)
  2. Stacked-ensemble ML yield predictor (Ridge/Lasso/XGB/LGBM -> Ridge meta),
     trained on Dairy_Data.csv (CowNflow/INRAE, 414 Holstein records)
  3. A silent in-distribution check: if the LP-generated ration profile falls
     inside the ML model's training envelope (BW 430-907kg, DMI 8-29.6kg/day,
     yield 5.5-47kg/day), use the ML prediction. Otherwise fall back to the
     ICAR requirement-based yield (ration solved to deliver target L kg/day
     BY CONSTRUCTION, per icar_response_integration.py's reasoning) --
     without labeling which path was used, per spec.
  4. A SHAP-style explanation via chain-rule propagation: LinearExplainer for
     the Ridge/Lasso base models, TreeExplainer for XGB/LGBM, then the base
     models' contributions are weighted by the meta-learner's coefficients
     on meta_ridge/meta_lasso/meta_xgb/meta_lgb and added to the meta
     model's own direct coefficients on the original features -- matching
     the mid-semester report's Section 10 approach.

This module is import-safe: model training happens lazily on first request
and is cached to disk (cache/yield_stack.pkl) so subsequent server restarts
don't re-train from scratch.
"""
import os
import csv
import pickle
import warnings
import numpy as np
import pandas as pd
import pulp
import shap

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DAIRY_CSV = os.path.join(DATA_DIR, "Dairy_Data.csv")
MATRIX_CSV = os.path.join(DATA_DIR, "nutri_matrix_guj.csv")
PROFILES_CSV = os.path.join(DATA_DIR, "animal_profiles.csv")
CACHE_PATH = os.path.join(CACHE_DIR, "yield_stack.pkl")

MCAL_TO_MJ = 4.184
NPN_CP_FRAC_MAX = 0.30
CONC_DMI_FRAC_MAX = 0.60

ICAR_MAINT = [
    (400, 8.64, 11.82, 436, 18, 8),
    (450, 9.72, 12.94, 476, 20, 9),
    (500, 10.80, 14.04, 515, 23, 10),
    (550, 11.88, 15.10, 553, 25, 11),
    (600, 12.96, 16.15, 591, 27, 12),
]
ICAR_PROD = [
    (4.0, 0.510, 1.20, 96, 3.2, 1.8),
    (6.0, 0.670, 1.58, 124, 4.8, 1.8),
]

# ML model's training envelope (Dairy_Data.csv, lactating subset).
# BW/DMI/yield are fixed dataset bounds; CP_intake and conc_prop are computed
# from the actual training frame in _train_and_cache (see ENVELOPE_EXTRA_COLS)
# since those are the features most likely to expose an out-of-distribution
# ICAR/Gujarat ration even when BW/DMI/yield look in-range.
ML_ENVELOPE = {
    "BW": (430.0, 907.0),
    "DMI": (8.0, 29.6),
    "yield": (5.5, 47.0),
}
ENVELOPE_EXTRA_COLS = ["CP_intake", "conc_prop"]
# Training bounds from only 402 rows are noisy at the edges; a hard cutoff at
# the exact min/max would reject plenty of rations that are genuinely close
# to in-distribution. Pad each bound outward by this fraction of its range
# before gating, so only clearly-outside-training-envelope rations get pushed
# to the ICAR fallback.
EXTRA_BOUNDS_MARGIN = 0.20

TARGET = "6.-Milk-production-(kg/day)"
DIET_TYPES = ["Maize", "Fresh_herbage", "Maize_Lucerne", "Dehydrated_herbage",
              "Maize_Fresh_herbage", "Maize_Hay"]


# ───────────────────────── LP side ─────────────────────────

def _interp_maint(bw_kg):
    if bw_kg <= ICAR_MAINT[0][0]:
        b0, d0, m0, c0, ca0, p0 = ICAR_MAINT[0]
        f = bw_kg / b0
        return d0 * f, m0 * f, c0 * f, ca0 * f, p0 * f
    if bw_kg >= ICAR_MAINT[-1][0]:
        b1, d1, m1, c1, ca1, p1 = ICAR_MAINT[-2]
        b2, d2, m2, c2, ca2, p2 = ICAR_MAINT[-1]
        t = (bw_kg - b1) / (b2 - b1)
        return d1 + (d2 - d1) * t, m1 + (m2 - m1) * t, c1 + (c2 - c1) * t, ca1 + (ca2 - ca1) * t, p1 + (p2 - p1) * t
    for i in range(len(ICAR_MAINT) - 1):
        b1, d1, m1, c1, ca1, p1 = ICAR_MAINT[i]
        b2, d2, m2, c2, ca2, p2 = ICAR_MAINT[i + 1]
        if b1 <= bw_kg <= b2:
            t = (bw_kg - b1) / (b2 - b1)
            return d1 + (d2 - d1) * t, m1 + (m2 - m1) * t, c1 + (c2 - c1) * t, ca1 + (ca2 - ca1) * t, p1 + (p2 - p1) * t


def _interp_prod(fat_pct):
    fat_pct = max(ICAR_PROD[0][0], min(ICAR_PROD[-1][0], fat_pct))
    f1, d1, m1, c1, ca1, p1 = ICAR_PROD[0]
    f2, d2, m2, c2, ca2, p2 = ICAR_PROD[-1]
    t = (fat_pct - f1) / (f2 - f1)
    return d1 + (d2 - d1) * t, m1 + (m2 - m1) * t, c1 + (c2 - c1) * t, ca1 + (ca2 - ca1) * t, p1


def compute_requirements(bw_kg, milk_kg_day, fat_pct, dmi_type="crossbred"):
    dm_m, me_m, cp_m, ca_m, p_m = _interp_maint(bw_kg)
    dm_l, me_l, cp_l, ca_l, p_l = _interp_prod(fat_pct)

    dm_total = dm_m + dm_l * milk_kg_day
    me_total = (me_m + me_l * milk_kg_day) * MCAL_TO_MJ
    cp_total = cp_m + cp_l * milk_kg_day
    ca_total = ca_m + ca_l * milk_kg_day
    p_total = p_m + p_l * milk_kg_day

    return {
        "breed": f"custom_{bw_kg}kg_{milk_kg_day}L_{fat_pct}fat",
        "bw_kg": bw_kg,
        "milk_yield_kg_day": milk_kg_day,
        "milk_fat_pct": fat_pct,
        "dmi_min_kg": round(dm_total * 0.92, 2),
        "dmi_max_kg": round(dm_total * 1.10, 2),
        "cp_req_g_day": round(cp_total, 1),
        "me_req_mj_day": round(me_total, 1),
        "ca_req_g_day": round(ca_total, 1),
        "p_req_g_day": round(p_total, 1),
        "ndf_min_frac": 0.30 if dmi_type == "indigenous" else 0.28,
        "ndf_max_frac": 0.42 if dmi_type == "indigenous" else 0.40,
        "forage_ndf_min_frac": 0.75,
        "ca_p_ratio_min": 1.5,
        "ca_p_ratio_max": 2.0,
        "dmi_type": dmi_type,
    }


def load_matrix(path=MATRIX_CSV):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                k, v = k.strip(), v.strip()
                try:
                    clean[k] = float(v)
                except ValueError:
                    clean[k] = v
            rows.append(clean)
    return rows


def solve(animal, ingredients):
    prob = pulp.LpProblem("low_cost_feed", pulp.LpMinimize)
    x = {
        ing["name"]: pulp.LpVariable(
            f"x_{ing['ingredient_id']:.0f}",
            lowBound=float(ing["inclusion_min_kg"]),
            upBound=float(ing["inclusion_max_kg"]),
        )
        for ing in ingredients
    }

    def dm_expr(ing):
        return ing["dm_pct"] / 100.0 * x[ing["name"]]

    prob += pulp.lpSum(ing["cost_inr_per_kg_asfed"] * x[ing["name"]] for ing in ingredients), "Cost"

    DMI = pulp.lpSum(dm_expr(ing) for ing in ingredients)
    prob += DMI >= animal["dmi_min_kg"], "DMI_min"
    prob += DMI <= animal["dmi_max_kg"], "DMI_max"
    CP = pulp.lpSum(dm_expr(ing) * ing["cp_pct_dm"] / 100 * 1000 for ing in ingredients)
    prob += CP >= animal["cp_req_g_day"], "CP_min"
    ME = pulp.lpSum(dm_expr(ing) * ing["me_mj_per_kg_dm"] for ing in ingredients)
    prob += ME >= animal["me_req_mj_day"], "ME_min"
    Ca = pulp.lpSum(dm_expr(ing) * ing["ca_pct_dm"] / 100 * 1000 for ing in ingredients)
    prob += Ca >= animal["ca_req_g_day"], "Ca_min"
    P = pulp.lpSum(dm_expr(ing) * ing["p_pct_dm"] / 100 * 1000 for ing in ingredients)
    prob += P >= animal["p_req_g_day"], "P_min"
    roughage_cats = {"roughage_dry", "roughage_green"}
    NDF = pulp.lpSum(dm_expr(ing) * ing["ndf_pct_dm"] / 100 for ing in ingredients)
    NDF_rough = pulp.lpSum(dm_expr(ing) * ing["ndf_pct_dm"] / 100
                            for ing in ingredients if ing["category"] in roughage_cats)
    prob += NDF >= animal["ndf_min_frac"] * DMI, "NDF_min"
    prob += NDF <= animal["ndf_max_frac"] * DMI, "NDF_max"
    prob += NDF_rough >= animal["forage_ndf_min_frac"] * NDF, "ForageNDF_min"
    prob += Ca >= animal["ca_p_ratio_min"] * P, "CaP_min"
    prob += Ca <= animal["ca_p_ratio_max"] * P, "CaP_max"
    NPN_CP = pulp.lpSum(dm_expr(ing) * ing["cp_pct_dm"] / 100 * 1000
                         for ing in ingredients if ing["category"] == "NPN")
    prob += NPN_CP <= NPN_CP_FRAC_MAX * CP, "NPN_frac"
    conc_cats = {"concentrate_energy", "concentrate_protein"}
    Conc_DM = pulp.lpSum(dm_expr(ing) for ing in ingredients if ing["category"] in conc_cats)
    prob += Conc_DM <= CONC_DMI_FRAC_MAX * DMI, "Conc_frac"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob, status, x


def ration_profile(prob, x, ingredients, animal):
    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    sol = [(ing, pulp.value(x[ing["name"]])) for ing in ingredients
           if pulp.value(x[ing["name"]]) is not None and pulp.value(x[ing["name"]]) > 1e-4]

    dmi = sum(ing["dm_pct"] / 100 * q for ing, q in sol)
    cp_g = sum(ing["dm_pct"] / 100 * ing["cp_pct_dm"] / 100 * q * 1000 for ing, q in sol)
    roughage_cats = {"roughage_dry", "roughage_green"}
    conc_cats = {"concentrate_energy", "concentrate_protein"}
    forage_dm = sum(ing["dm_pct"] / 100 * q for ing, q in sol if ing["category"] in roughage_cats)
    conc_dm = sum(ing["dm_pct"] / 100 * q for ing, q in sol if ing["category"] in conc_cats)
    tdn_weighted = sum(ing["dm_pct"] / 100 * ing["tdn_pct_dm"] / 100 * q for ing, q in sol)
    dm_digestibility = tdn_weighted / dmi if dmi > 0 else np.nan
    cost_total = pulp.value(prob.objective)

    ration_items = [
        {
            "name": ing["name"],
            "category": ing["category"],
            "kg_as_fed": round(q, 3),
            "cost_inr": round(ing["cost_inr_per_kg_asfed"] * q, 2),
        }
        for ing, q in sorted(sol, key=lambda t: -t[1])
    ]

    return {
        "BW": animal["bw_kg"],
        "age_months": 42.0,
        "lactation_week": 15.0,
        "gestation_week": 0.0,
        "DMI": dmi,
        "CP_intake": cp_g,
        "N_intake": cp_g / 6.25,
        "diet_CP_conc": cp_g / dmi if dmi > 0 else np.nan,
        "conc_prop": conc_dm / dmi if dmi > 0 else np.nan,
        "forage_prop": forage_dm / dmi if dmi > 0 else np.nan,
        "DM_digestibility": dm_digestibility,
        "N_digestibility": dm_digestibility,
        "feed_cost_inr_day": cost_total,
        "target_milk_kg_day": animal["milk_yield_kg_day"],
        "ration_items": ration_items,
    }


# ───────────────────────── ML side ─────────────────────────

def build_feature_frame(df):
    BW, AGE = df["2.-Body-weight-(kg)"], df["2.-Cow-age-(month)"]
    LW = df["2.-Lactation-week"]
    GW = df["2.-Gestation-week"].fillna(0)
    DMI = df["4.-DM-intake-(kg/day)"]
    CPI = df["4.-CP-intake-(g/day)"]
    NI = df["4.-N-intake-(g/day)"]
    CPc = df["4.-Diet-CP-concentration-(g/kg-DM)"]
    CONC = df["3.-Concentrate-proportion-in-the-diet-(kg/kg,-DM-basis)"]
    FORG = df["3.-Forage-proportion-in-the-diet-(kg/kg,-DM-basis)"]
    DMDG = df["5.-DM-digestibility-(g/g)"]
    ND = df["5.-N-digestibility-(g/g)"]

    fe = pd.DataFrame(index=df.index)
    fe["BW"] = BW
    fe["age_months"] = AGE
    fe["lactation_week"] = LW
    fe["gestation_week"] = GW
    fe["DMI"] = DMI
    fe["CP_intake"] = CPI
    fe["N_intake"] = NI
    fe["diet_CP_conc"] = CPc
    fe["conc_prop"] = CONC
    fe["forage_prop"] = FORG
    fe["DM_digestibility"] = DMDG
    fe["N_digestibility"] = ND

    fe["DMI_per_metBW"] = DMI / (BW ** 0.75)
    fe["CP_per_metBW"] = CPI / (BW ** 0.75)
    fe["dig_CP"] = CPI * ND
    fe["dig_DMI"] = DMI * DMDG
    fe["BW_x_DMI"] = BW * DMI
    fe["CP_density_x_DMI"] = CPc * DMI
    fe["conc_forage_ratio"] = CONC / (FORG + 1e-6)
    fe["lact_week_sq"] = LW ** 2
    fe["age_sq"] = AGE ** 2
    fe["conc_prop_sq"] = CONC ** 2
    fe["ln_lact_week"] = np.log1p(LW)

    diet_dummies = pd.get_dummies(df["3.-Diet-type"], prefix="diet", drop_first=False)
    fe = pd.concat([fe, diet_dummies], axis=1)
    fe[TARGET] = df[TARGET].values
    fe = fe.dropna(subset=[TARGET]).fillna(fe.median(numeric_only=True))
    return fe


class YieldStack:
    """Ridge/Lasso/XGB/LGBM -> Ridge meta-learner, matching Milk_Yield_Predictor.ipynb."""

    def __init__(self):
        self.ridge = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=0.01))])
        self.lasso = Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=0.001, max_iter=10000))])
        self.xgbm = xgb.XGBRegressor(n_estimators=700, max_depth=4, learning_rate=0.07,
                                      subsample=0.7, colsample_bytree=1.0,
                                      reg_alpha=0, reg_lambda=3, min_child_weight=3,
                                      random_state=42, n_jobs=-1, verbosity=0)
        self.lgbm = lgb.LGBMRegressor(n_estimators=700, max_depth=6, learning_rate=0.07,
                                       num_leaves=15, subsample=1.0, reg_alpha=0.05,
                                       reg_lambda=0, min_child_samples=5,
                                       random_state=42, n_jobs=-1, verbose=-1)
        self.meta = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=0.01))])
        self.feature_cols = None
        self.stack_cols = None
        self._bg = None  # background sample for SHAP explainers
        self._explainers = {}

    def fit(self, fe):
        self.feature_cols = [c for c in fe.columns if c != TARGET]
        X, y = fe[self.feature_cols].astype(float), fe[TARGET]
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        oof_ridge = cross_val_predict(self.ridge, X, y, cv=cv)
        oof_lasso = cross_val_predict(self.lasso, X, y, cv=cv)
        oof_xgb = cross_val_predict(self.xgbm, X, y, cv=cv)
        oof_lgb = cross_val_predict(self.lgbm, X, y, cv=cv)

        X_stack = X.copy()
        X_stack["meta_ridge"], X_stack["meta_lasso"] = oof_ridge, oof_lasso
        X_stack["meta_xgb"], X_stack["meta_lgb"] = oof_xgb, oof_lgb
        self.stack_cols = list(X_stack.columns)

        self.meta.fit(X_stack, y)
        self.ridge.fit(X, y)
        self.lasso.fit(X, y)
        self.xgbm.fit(X, y)
        self.lgbm.fit(X, y)

        oof_final = cross_val_predict(self.meta, X_stack, y, cv=cv)
        rmse = float(np.sqrt(np.mean((y - oof_final) ** 2)))
        r2 = float(1 - np.sum((y - oof_final) ** 2) / np.sum((y - y.mean()) ** 2))

        # background sample for SHAP (small subset for speed)
        self._bg = X.sample(min(60, len(X)), random_state=42)
        return {"rmse": rmse, "r2": r2}

    def _row_from_feat(self, feat_dict, diet_type_mode):
        row = {c: 0.0 for c in self.feature_cols}
        for k, v in feat_dict.items():
            if k in row:
                row[k] = v
        diet_col = f"diet_{diet_type_mode}"
        if diet_col in row:
            row[diet_col] = 1.0
        return pd.DataFrame([row])[self.feature_cols].astype(float)

    def predict_one(self, feat_dict, diet_type_mode):
        X = self._row_from_feat(feat_dict, diet_type_mode)
        xs = X.copy()
        xs["meta_ridge"] = self.ridge.predict(X)
        xs["meta_lasso"] = self.lasso.predict(X)
        xs["meta_xgb"] = self.xgbm.predict(X)
        xs["meta_lgb"] = self.lgbm.predict(X)
        xs = xs[self.stack_cols]
        return float(self.meta.predict(xs)[0])

    def explain_one(self, feat_dict, diet_type_mode, top_k=6):
        """Chain-rule SHAP: LinearExplainer(Ridge/Lasso) + TreeExplainer(XGB/LGBM),
        weighted by the meta-learner's coefficients on each base model's output,
        plus the meta-learner's direct coefficients on the original features."""
        X = self._row_from_feat(feat_dict, diet_type_mode)

        if "ridge" not in self._explainers:
            self._explainers["ridge"] = shap.LinearExplainer(self.ridge.named_steps["m"],
                                                               self.ridge.named_steps["s"].transform(self._bg))
            self._explainers["lasso"] = shap.LinearExplainer(self.lasso.named_steps["m"],
                                                               self.lasso.named_steps["s"].transform(self._bg))
            self._explainers["xgb"] = shap.TreeExplainer(self.xgbm)
            self._explainers["lgb"] = shap.TreeExplainer(self.lgbm)

        Xs_ridge = self.ridge.named_steps["s"].transform(X)
        Xs_lasso = self.lasso.named_steps["s"].transform(X)
        sv_ridge = self._explainers["ridge"].shap_values(Xs_ridge)[0]
        sv_lasso = self._explainers["lasso"].shap_values(Xs_lasso)[0]
        sv_xgb = self._explainers["xgb"].shap_values(X)[0]
        sv_lgb = self._explainers["lgb"].shap_values(X)[0]

        # meta model coefficients (in standardized meta-feature space); convert
        # to raw-space weights by dividing by the meta scaler's per-feature std
        meta_model = self.meta.named_steps["m"]
        meta_scaler = self.meta.named_steps["s"]
        coefs = dict(zip(self.stack_cols, meta_model.coef_ / meta_scaler.scale_))

        w_ridge = coefs.get("meta_ridge", 0.0)
        w_lasso = coefs.get("meta_lasso", 0.0)
        w_xgb = coefs.get("meta_xgb", 0.0)
        w_lgb = coefs.get("meta_lgb", 0.0)

        contributions = {}
        for i, feat in enumerate(self.feature_cols):
            direct = coefs.get(feat, 0.0) * (X.iloc[0, i] - self._bg[feat].mean())
            via_bases = (w_ridge * sv_ridge[i] + w_lasso * sv_lasso[i]
                         + w_xgb * sv_xgb[i] + w_lgb * sv_lgb[i])
            contributions[feat] = float(direct + via_bases)

        ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:top_k]
        return [{"feature": f, "contribution_kg_per_day": round(v, 4)} for f, v in ranked]


def _train_and_cache():
    df_raw = pd.read_csv(DAIRY_CSV)
    df = df_raw[df_raw["2.-Physiological-status-"] == "lactating"].copy()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    fe = build_feature_frame(df)

    model = YieldStack()
    metrics = model.fit(fe)
    diet_mode_default = df["3.-Diet-type"].mode()[0]

    # Extra in-distribution bounds computed from the real training rows, not
    # guessed -- these catch ICAR/Gujarat rations whose CP density or
    # concentrate:forage mix don't resemble the CowNflow training data even
    # when BW/DMI/target yield happen to fall inside ML_ENVELOPE. Padded by
    # EXTRA_BOUNDS_MARGIN so the gate only excludes clearly-OOD rations
    # rather than anything slightly past the observed min/max.
    extra_bounds = {}
    for col in ENVELOPE_EXTRA_COLS:
        lo, hi = float(fe[col].min()), float(fe[col].max())
        pad = (hi - lo) * EXTRA_BOUNDS_MARGIN
        extra_bounds[col] = (lo - pad, hi + pad)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics, "diet_mode_default": diet_mode_default,
                     "extra_bounds": extra_bounds}, f)
    return model, metrics, diet_mode_default, extra_bounds


_STATE = {}


def get_model():
    """Lazily train (or load cached) the ML yield stack. Cached in-process too."""
    if "model" in _STATE:
        return _STATE["model"], _STATE["metrics"], _STATE["diet_mode_default"], _STATE["extra_bounds"]
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                d = pickle.load(f)
            extra_bounds = d.get("extra_bounds")
            if extra_bounds is None:
                # cache from before this fix -- retrain so bounds are populated
                raise KeyError("extra_bounds missing from cache; retraining")
            _STATE.update(model=d["model"], metrics=d["metrics"], diet_mode_default=d["diet_mode_default"],
                          extra_bounds=extra_bounds)
            return d["model"], d["metrics"], d["diet_mode_default"], extra_bounds
        except Exception:
            pass
    model, metrics, diet_mode_default, extra_bounds = _train_and_cache()
    _STATE.update(model=model, metrics=metrics, diet_mode_default=diet_mode_default, extra_bounds=extra_bounds)
    return model, metrics, diet_mode_default, extra_bounds


def in_distribution(bw_kg, dmi, target_yield, cp_intake=None, conc_prop=None, extra_bounds=None):
    b_lo, b_hi = ML_ENVELOPE["BW"]
    d_lo, d_hi = ML_ENVELOPE["DMI"]
    y_lo, y_hi = ML_ENVELOPE["yield"]
    ok = (b_lo <= bw_kg <= b_hi) and (d_lo <= dmi <= d_hi) and (y_lo <= target_yield <= y_hi)
    if not ok or not extra_bounds:
        return ok
    if cp_intake is not None and "CP_intake" in extra_bounds:
        lo, hi = extra_bounds["CP_intake"]
        ok = ok and (lo <= cp_intake <= hi)
    if conc_prop is not None and "conc_prop" in extra_bounds:
        lo, hi = extra_bounds["conc_prop"]
        ok = ok and (lo <= conc_prop <= hi)
    return ok


def load_breed_presets():
    presets = []
    with open(PROFILES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            presets.append({
                "breed": row["breed"],
                "description": row["description"],
                "bw_kg": float(row["bw_kg"]),
                "milk_yield_kg_day": float(row["milk_yield_kg_day"]),
                "milk_fat_pct": float(row["milk_fat_pct"]),
                "dmi_type": row["dmi_type"],
            })
    return presets


def generate_feedback(bw_kg, fat_pct, dmi_type, target_yield, predicted_yield,
                       feed_cost, profit, conc_prop, marginal_note):
    notes = []

    gap = predicted_yield - target_yield
    if abs(gap) < 0.15:
        notes.append(f"The ration delivers close to your {target_yield:.1f} kg/day target "
                      f"({predicted_yield:.1f} kg/day predicted) -- the feed plan is well matched to this animal.")
    elif gap < 0:
        notes.append(f"Predicted actual yield ({predicted_yield:.1f} kg/day) trails the "
                      f"{target_yield:.1f} kg/day target by {abs(gap):.1f} kg -- consider a richer "
                      f"energy/protein mix or re-checking the DMI ceiling for this body weight.")
    else:
        notes.append(f"Predicted actual yield ({predicted_yield:.1f} kg/day) comes in above the "
                      f"{target_yield:.1f} kg/day target, a good sign the ration has some headroom.")

    if conc_prop is not None:
        if conc_prop > 0.55:
            notes.append(f"Concentrate makes up {conc_prop*100:.0f}% of dry matter intake, near the "
                          f"60% ceiling used here to protect rumen health -- pushing yield further will "
                          f"likely hit this cap before cost does.")
        elif conc_prop < 0.15:
            notes.append(f"Concentrate is only {conc_prop*100:.0f}% of the ration -- there is room to "
                          f"trade in more concentrate for extra yield if the economics justify it.")

    if profit < 0:
        notes.append("At the current milk price, feed cost exceeds revenue for this target -- "
                      "either the target yield or the milk price assumption may need revisiting.")

    if dmi_type == "indigenous":
        notes.append("Indigenous-breed NDF bounds (30-42%) were applied, reflecting typical "
                      "roughage tolerance for this animal type.")
    else:
        notes.append("Crossbred NDF bounds (28-40%) were applied for this animal type.")

    if marginal_note:
        notes.append(marginal_note)

    return notes


def compute_full(bw_kg, fat_pct, dmi_type, target_yield, milk_price, diet_type_mode=None):
    """Full pipeline for one slider combination. Silently picks ML vs ICAR yield source."""
    model, metrics, diet_mode_default, extra_bounds = get_model()
    diet_type_mode = diet_type_mode or diet_mode_default

    ingredients = load_matrix()
    animal = compute_requirements(bw_kg, target_yield, fat_pct, dmi_type)
    prob, status, x = solve(animal, ingredients)

    if pulp.LpStatus[prob.status] != "Optimal":
        return {"feasible": False,
                "message": "No feasible ration exists for this combination of body weight, "
                           "fat %, and target yield under the current ingredient set and "
                           "physiological constraints. Try a lower target yield or a higher DMI ceiling."}

    prof = ration_profile(prob, x, ingredients, animal)
    feed_cost = prof["feed_cost_inr_day"]

    use_ml = in_distribution(bw_kg, prof["DMI"], target_yield,
                              cp_intake=prof["CP_intake"], conc_prop=prof["conc_prop"],
                              extra_bounds=extra_bounds)
    if use_ml:
        predicted_yield = model.predict_one(prof, diet_type_mode)
        shap_expl = model.explain_one(prof, diet_type_mode)
    else:
        # ICAR fallback: the ration was solved to meet the requirement for
        # target_yield, so by the same physiological model, yield = target.
        predicted_yield = target_yield
        shap_expl = []

    revenue = predicted_yield * milk_price
    profit = revenue - feed_cost

    marginal_note = None
    if prof["conc_prop"] is not None and prof["conc_prop"] >= CONC_DMI_FRAC_MAX - 1e-6:
        marginal_note = ("This ration is bound by the concentrate-fraction ceiling rather than "
                          "a cost/price crossover -- pushing the target higher will hit this "
                          "physical limit before it becomes unprofitable.")

    feedback = generate_feedback(bw_kg, fat_pct, dmi_type, target_yield, predicted_yield,
                                  feed_cost, profit, prof["conc_prop"], marginal_note)

    return {
        "feasible": True,
        "inputs": {
            "bw_kg": bw_kg, "fat_pct": fat_pct, "dmi_type": dmi_type,
            "target_yield_kg_day": target_yield, "milk_price_inr_per_kg": milk_price,
            "diet_type_mode": diet_type_mode,
        },
        "predicted_yield_kg_day": round(predicted_yield, 2),
        "feed_cost_inr_day": round(feed_cost, 2),
        "revenue_inr_day": round(revenue, 2),
        "profit_inr_day": round(profit, 2),
        "dmi_kg_day": round(prof["DMI"], 2),
        "cp_intake_g_day": round(prof["CP_intake"], 1),
        "conc_prop": round(prof["conc_prop"], 3) if prof["conc_prop"] is not None else None,
        "forage_prop": round(prof["forage_prop"], 3) if prof["forage_prop"] is not None else None,
        "dm_digestibility": round(prof["DM_digestibility"], 3) if prof["DM_digestibility"] is not None else None,
        "ration_items": prof["ration_items"],
        "shap_explanation": shap_expl,
        "feedback": feedback,
    }