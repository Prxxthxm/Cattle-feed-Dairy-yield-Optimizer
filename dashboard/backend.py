"""
Flask API for the Cattle Feed Optimization & Dairy Yield dashboard.

Run:  python backend.py
Then open http://127.0.0.1:5050 in a browser.
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

import model_core as mc

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/presets")
def presets():
    return jsonify(mc.load_breed_presets())


@app.route("/api/diet_types")
def diet_types():
    return jsonify(mc.DIET_TYPES)


@app.route("/api/compute", methods=["POST"])
def compute():
    body = request.get_json(force=True)
    try:
        bw_kg = float(body["bw_kg"])
        fat_pct = float(body["fat_pct"])
        dmi_type = body.get("dmi_type", "crossbred")
        target_yield = float(body["target_yield"])
        milk_price = float(body["milk_price"])
        diet_type_mode = body.get("diet_type_mode")
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"feasible": False, "message": f"Invalid input: {e}"}), 400

    if dmi_type not in ("crossbred", "indigenous"):
        dmi_type = "crossbred"
    if diet_type_mode not in mc.DIET_TYPES:
        diet_type_mode = None

    # sane slider clamps so nothing sent from the UI can crash the LP/ICAR math
    bw_kg = max(250.0, min(1000.0, bw_kg))
    fat_pct = max(3.0, min(7.0, fat_pct))
    target_yield = max(1.0, min(45.0, target_yield))
    milk_price = max(1.0, min(200.0, milk_price))

    result = mc.compute_full(bw_kg, fat_pct, dmi_type, target_yield, milk_price, diet_type_mode)
    return jsonify(result)


@app.route("/api/warmup", methods=["POST"])
def warmup():
    """Optional: pre-train the ML stack on server start so the first slider
    move doesn't pay the ~5s training cost."""
    model, metrics, diet_mode_default, extra_bounds = mc.get_model()
    return jsonify({"status": "ready", "metrics": metrics, "default_diet_type": diet_mode_default})


if __name__ == "__main__":
    print("Pre-training yield stack (one-time, ~5-10s, cached after)...")
    mc.get_model()
    print("Ready. Open http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)