"""
Profit-maximizing feed optimizer -- ICAR response-curve version.

Integrates:
  1. LP least-cost ration formulator (Optimisation_Engine_v1.ipynb)
  2. ICAR 2013 marginal production requirement tables (already embedded in
     that same LP) AS the yield-response model, instead of the ML stack.

Why this is different from v3:
  ICAR_PROD gives the extra ME (Mcal) and CP (g) required PER extra kg of
  milk, for a given fat%. That is itself a marginal milk-response function --
  it says "to sustain L kg/day of milk on top of maintenance, you need to
  feed at least this much ME/CP." The LP already solves for the least-cost
  ration that supplies >= that ME/CP for a chosen target L.

  So instead of predicting yield from an externally-trained model (which
  requires extrapolating onto out-of-distribution rations, see
  poc_ml_integration.py's warnings), we exploit the fact that the LP's own
  requirement construction IS the response curve: a ration solved to meet
  the ICAR requirement for target L delivers (by the same physiological
  model used to build the requirement) yield = L. No second model, no
  extrapolation, no mismatched dataset. The tradeoff: we inherit whatever
  approximation error ICAR's own linear per-kg tables carry, and we lose
  the individual-cow-level nuance (age, lactation week, digestibility, etc.)
  the ML side was trying to capture.

Result: profit(L) = L * milk_price - feed_cost(L), swept over L, LP-solved
at each point. Fully within the same requirement system already validated
in Optimisation_Engine_v1.ipynb -- no cross-dataset generalization claim.
"""
import numpy as np
import pandas as pd
import pulp

from profit_optimizer import load_matrix, compute_requirements, solve, ration_profile


def icar_profit_frontier(bw_kg, fat_pct, dmi_type, milk_price_inr_per_kg,
                          matrix_path, milk_lo=4.0, milk_hi=18.0, step=0.5):
    ingredients = load_matrix(matrix_path)
    rows = []
    for L in np.arange(milk_lo, milk_hi + 1e-9, step):
        L = round(float(L), 2)
        animal = compute_requirements(bw_kg, L, fat_pct, dmi_type)
        prob, status, x = solve(animal, ingredients)
        if pulp.LpStatus[prob.status] != "Optimal":
            rows.append({"target_milk_L": L, "feasible": False})
            continue
        prof = ration_profile(prob, x, ingredients, animal)
        cost = prof["feed_cost_inr_day"]
        revenue = L * milk_price_inr_per_kg  # ICAR-consistent: ration meets exactly the
                                              # requirement built to sustain L kg/day
        profit = revenue - cost
        rows.append({
            "target_milk_L": L,
            "feasible": True,
            "feed_cost_inr": cost,
            "revenue_inr": revenue,
            "profit_inr_day": profit,
            "marginal_cost_per_L": np.nan,  # filled below
            "DMI_kg": prof["DMI"],
            "CP_intake_g": prof["CP_intake"],
            "conc_prop": prof["conc_prop"],
        })
    df = pd.DataFrame(rows)
    df.loc[df["feasible"], "marginal_cost_per_L"] = df.loc[df["feasible"], "feed_cost_inr"].diff() / step
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("ICAR RESPONSE-CURVE VERSION -- numerically trustworthy within the")
    print("same ICAR 2013 requirement system your LP is already built on.")
    print("No externally-trained ML model, no extrapolation risk.")
    print("=" * 70)

    MILK_PRICE = 46.0  # INR/kg, flat-price assumption -- ADJUST to your actual procurement price
    frontier = icar_profit_frontier(
        bw_kg=450, fat_pct=4.0, dmi_type="crossbred",
        milk_price_inr_per_kg=MILK_PRICE,
        matrix_path="/mnt/user-data/uploads/nutri_matrix_guj.csv",
    )

    feasible = frontier[frontier["feasible"]].reset_index(drop=True)
    print(f"\nProfit frontier (HF_crossbreed, 450kg, milk price INR {MILK_PRICE}/kg):")
    print(feasible.drop(columns="feasible").to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    best = feasible.loc[feasible["profit_inr_day"].idxmax()]
    print("\n=== Profit-maximizing operating point ===")
    print(best.to_string())

    # Where marginal cost per extra litre exceeds milk price, further pushing yield destroys profit -- that crossover is the economically correct stopping point.
    crossover = feasible[feasible["marginal_cost_per_L"] > MILK_PRICE]
    if len(crossover):
        print(f"\nMarginal feed cost exceeds milk price (INR {MILK_PRICE}/kg) starting at "
              f"target_milk_L = {crossover.iloc[0]['target_milk_L']} -- pushing yield past this "
              f"point costs more in feed than it earns in milk.")

    feasible.to_csv("icar_profit_frontier.csv", index=False)
    print("\nSaved: icar_profit_frontier.csv")
