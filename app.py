from flask import Flask, request, render_template, redirect, url_for, session
from joblib import load
import os
import pandas as pd

from data_preprocessing import load_data, preprocess
from train_model import compute_suitability_score
from recommend import filter_by_rules

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace_with_a_random_secret_key"

# Load & preprocess once
DF = preprocess(load_data("completed_bike_dataset.xlsx"))
DF["suitability_score"] = compute_suitability_score(DF)

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "bike_suitability_rf.joblib")
MODEL = load(MODEL_PATH)

# Geography codes
GEO_CODE = {
    "mountain": 0, "coastal": 1, "city": 2,
    "small_town": 3, "highway": 4, "back_roads": 5
}

# --- Budget-only re-ranking bonus (no other behavior changed) ---
# Tunable via env var, defaults to 0.08 (i.e., +8% max nudging)
BUDGET_BONUS_MAX = float(os.environ.get("BUDGET_BONUS_MAX", "0.08"))

def budget_bonus(price: float, budget: float) -> float:
    """
    Reward bikes that are closer to the user's budget ceiling (within budget).
    r = price / budget  in [0,1]
    bonus = BUDGET_BONUS_MAX * r^2
    """
    if budget <= 0:  # guard
        return 0.0
    if price > budget:
        return 0.0
    r = max(0.0, min(1.0, float(price) / float(budget)))
    return BUDGET_BONUS_MAX * (r ** 2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/skill", methods=["GET","POST"])
def skill():
    if request.method == "POST":
        session["experience_level"] = int(request.form["experience_level"])
        return redirect(url_for("location"))
    return render_template("skill.html")

@app.route("/location", methods=["GET","POST"])
def location():
    if request.method == "POST":
        session["geography"] = GEO_CODE[request.form["geography"]]
        return redirect(url_for("budget"))
    return render_template("location.html")

@app.route("/budget", methods=["GET","POST"])
def budget():
    if request.method == "POST":
        session["budget"] = int(request.form["budget"])

        user = {
            "experience_level": session["experience_level"],
            "geography":        session["geography"],
            "budget":           session["budget"]
        }

        # 1) Rule-based filtering (unchanged)
        candidates = filter_by_rules(DF.copy(), user)
        if candidates.empty:
            return render_template("results.html", error="No bikes match your criteria.")

        # 2) Random Forest base score (unchanged)
        Xc = candidates[["engine_size","riding_style_code","price"]].values
        base = MODEL.predict(Xc)

        # 3) Budget-only re-ranking bonus (NEW, small & safe)
        budget_val = float(user["budget"])
        bonuses = candidates["price"].apply(lambda p: budget_bonus(p, budget_val)).values

        candidates = candidates.copy()
        candidates["score"] = base + bonuses

        # 4) Rank and return (unchanged)
        top5 = candidates.sort_values("score", ascending=False).head(5)
        return render_template("results.html", bikes=top5.to_dict(orient="records"))

    return render_template("budget.html")

if __name__ == "__main__":
    # Optional: tune nudging without code changes
    # os.environ["BUDGET_BONUS_MAX"] = "0.08"
    app.run(host="0.0.0.0", port=5000, debug=True)