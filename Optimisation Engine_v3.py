"""
Profit-maximizing feed optimizer.

Integrates:
  1. LP least-cost ration formulator (Optimisation_Engine_v1.ipynb)
  2. Stacked-ensemble milk yield predictor (Milk_Yield_Predictor.ipynb)

Why not a single joint LP/QP:
  The yield model is a nonlinear stack (XGB/LGBM trees + Ridge meta), so it
  can't be embedded as a linear/convex objective in the LP. Instead we use a
  PARAMETRIC FRONTIER SWEEP:
    for each candidate production target L (kg milk/day):
        solve LP -> least-cost ration that satisfies ICAR requirements for L
        derive the ration's actual nutritional profile (DMI, CP, digestibility...)
        feed that profile into the ML model -> predicted ACTUAL yield y_hat(L)
        profit(L) = y_hat(L) * milk_price - feed_cost(L)
    pick L* = argmax profit(L)


================================ IMPORTANT ==================================
STATUS: ARCHITECTURE PROOF-OF-CONCEPT ONLY. NUMBERS ARE NOT RELIABLE.

The yield model (Milk_Yield_Predictor.ipynb) is trained on Dairy_Data.csv,
whose cows (BW 430-907kg, DMI 8-30kg/day, yield 5.5-47kg/day, high-input
system) are a materially different population from the ICAR Gujarat/
crossbred cows this LP is built for (BW ~350-450kg, DMI ~10-16kg/day,
yield ~5-10kg/day). LP-generated rations fall outside the training data's
feature envelope, so the model is extrapolating, not predicting.

Diagnosed failure mode: the stacking meta-learner's coefficients on the
Ridge/Lasso base-model outputs are NEGATIVE (learned during training to
discount linear-model disagreement with the trees). Off-distribution, the
tree members (XGB/LGBM) flatline entirely (can't extrapolate past leaf
boundaries) while Ridge/Lasso do extrapolate -- but the meta-learner's
negative weighting on them cancels that movement instead of amplifying it.
Net result: predicted yield barely changes across very different rations,
and the "profit-maximizing" ration this script reports is not meaningful.

USE THIS FILE ONLY to demonstrate that the LP -> ration-profile -> ML ->
profit-sweep pipeline runs end to end. For an actual profit number, see
icar_response_integration.py instead.
===============================================================================
"""
import csv
import numpy as np
import pandas as pd
import pulp
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

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


# ───────────────────────── LP side (unchanged from Optimisation_Engine_v1) ─────────────────────────

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


def load_matrix(path):
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
    """Derive the ML model's required feature set from a solved LP ration."""
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
    # TDN%/100 as a digestibility proxy (no direct digestibility column in the nutrient matrix)
    tdn_weighted = sum(ing["dm_pct"] / 100 * ing["tdn_pct_dm"] / 100 * q for ing, q in sol)
    dm_digestibility = tdn_weighted / dmi if dmi > 0 else np.nan
    cost_total = pulp.value(prob.objective)

    return {
        "BW": animal["bw_kg"],
        "age_months": 42.0,          # not modeled by LP side; herd-median placeholder
        "lactation_week": 15.0,      # placeholder mid-lactation
        "gestation_week": 0.0,
        "DMI": dmi,
        "CP_intake": cp_g,
        "N_intake": cp_g / 6.25,
        "diet_CP_conc": cp_g / dmi if dmi > 0 else np.nan,
        "conc_prop": conc_dm / dmi if dmi > 0 else np.nan,
        "forage_prop": forage_dm / dmi if dmi > 0 else np.nan,
        "DM_digestibility": dm_digestibility,
        "N_digestibility": dm_digestibility,  # proxy: no independent N-digestibility signal from LP side
        "feed_cost_inr_day": cost_total,
        "target_milk_kg_day": animal["milk_yield_kg_day"],
    }


# ───────────────────────── ML side (re-trained stack, matches Milk_Yield_Predictor.ipynb) ─────────────────────────

TARGET = "6.-Milk-production-(kg/day)"

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
    """Ridge/Lasso/XGB/LGBM -> Ridge meta-learner, matching Milk_Yield_Predictor.ipynb's architecture."""

    def __init__(self):
        self.ridge = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10.0))])
        self.lasso = Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=10000))])
        self.xgbm = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                      subsample=0.85, colsample_bytree=0.85,
                                      random_state=42, n_jobs=-1, verbosity=0)
        self.lgbm = lgb.LGBMRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                       num_leaves=15, random_state=42, n_jobs=-1, verbose=-1)
        self.meta = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
        self.feature_cols = None

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
        self.stack_cols = X_stack.columns

        self.meta.fit(X_stack, y)
        self.ridge.fit(X, y)
        self.lasso.fit(X, y)
        self.xgbm.fit(X, y)
        self.lgbm.fit(X, y)

        oof_final = cross_val_predict(self.meta, X_stack, y, cv=cv)
        rmse = float(np.sqrt(np.mean((y - oof_final) ** 2)))
        r2 = float(1 - np.sum((y - oof_final) ** 2) / np.sum((y - y.mean()) ** 2))
        return {"rmse": rmse, "r2": r2}

    def predict_one(self, feat_dict, diet_type_mode):
        row = {c: 0.0 for c in self.feature_cols}
        for k, v in feat_dict.items():
            if k in row:
                row[k] = v
        diet_col = f"diet_{diet_type_mode}"
        if diet_col in row:
            row[diet_col] = 1.0
        X = pd.DataFrame([row])[self.feature_cols].astype(float)
        xs = X.copy()
        xs["meta_ridge"] = self.ridge.predict(X)
        xs["meta_lasso"] = self.lasso.predict(X)
        xs["meta_xgb"] = self.xgbm.predict(X)
        xs["meta_lgb"] = self.lgbm.predict(X)
        xs = xs[self.stack_cols]
        return float(self.meta.predict(xs)[0])


# ───────────────────────── Profit sweep ─────────────────────────

def profit_frontier(bw_kg, fat_pct, dmi_type, milk_price_inr_per_kg,
                     matrix_path, model, diet_type_mode,
                     milk_lo=4.0, milk_hi=18.0, step=0.5):
    ingredients = load_matrix(matrix_path)
    rows = []
    for L in np.arange(milk_lo, milk_hi + 1e-9, step):
        animal = compute_requirements(bw_kg, round(float(L), 2), fat_pct, dmi_type)
        prob, status, x = solve(animal, ingredients)
        if pulp.LpStatus[prob.status] != "Optimal":
            continue
        prof = ration_profile(prob, x, ingredients, animal)
        y_hat = model.predict_one(prof, diet_type_mode)
        revenue = y_hat * milk_price_inr_per_kg
        profit = revenue - prof["feed_cost_inr_day"]
        rows.append({
            "target_milk_L": L,
            "predicted_milk_kg": y_hat,
            "feed_cost_inr": prof["feed_cost_inr_day"],
            "revenue_inr": revenue,
            "profit_inr_day": profit,
            "DMI_kg": prof["DMI"],
            "CP_intake_g": prof["CP_intake"],
            "conc_prop": prof["conc_prop"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 70)
    print("ARCHITECTURE PROOF-OF-CONCEPT -- yield numbers are NOT reliable.")
    print("See module docstring / icar_response_integration.py for the")
    print("numerically-trustworthy version of this pipeline.")
    print("=" * 70)

    print("\nLoading dairy dataset and training yield stack "
          "(fast-config, not the full grid search)...")
    df_raw = pd.read_csv("/mnt/user-data/uploads/Dairy_Data.csv")
    df = df_raw[df_raw["2.-Physiological-status-"] == "lactating"].copy()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    fe = build_feature_frame(df)

    model = YieldStack()
    metrics = model.fit(fe)
    print(f"  5-fold OOF (on Dairy_Data.csv's own distribution): "
          f"RMSE={metrics['rmse']:.3f} kg/day, R2={metrics['r2']:.3f}")
    print("  ^ this R2 is real, but only within Dairy_Data.csv's own feature range,")
    print("    which does NOT cover the ICAR ration profiles generated below.")

    diet_mode = df["3.-Diet-type"].mode()[0]
    print(f"  Using diet type '{diet_mode}' as the LP-ration proxy (dominant class in training data)")

    MILK_PRICE = 46.0  # INR/kg, flat-price assumption -- ADJUST to your actual procurement price
    frontier = profit_frontier(
        bw_kg=450, fat_pct=4.0, dmi_type="crossbred",
        milk_price_inr_per_kg=MILK_PRICE,
        matrix_path="/mnt/user-data/uploads/nutri_matrix_guj.csv",
        model=model, diet_type_mode=diet_mode,
    )

    print(f"\nProfit frontier (HF_crossbreed, 450kg, milk price INR {MILK_PRICE}/kg):")
    print("[predicted_milk_kg / profit_inr_day are extrapolated -- NOT trustworthy magnitudes]")
    print(frontier.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    best = frontier.loc[frontier["profit_inr_day"].idxmax()]
    print("\n=== Reported 'optimum' (architecture demo only, not a real recommendation) ===")
    print(best.to_string())

    frontier.to_csv("poc_frontier_UNRELIABLE.csv", index=False)
    print("\nSaved: poc_frontier_UNRELIABLE.csv")
