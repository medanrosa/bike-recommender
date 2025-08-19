from flask import (
    Flask, request, render_template,
    redirect, url_for, session
)
from joblib import load
import pandas as pd

from data_preprocessing import load_data, preprocess
from train_model import compute_suitability_score
from recommend import filter_by_rules

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
app.secret_key = "replace_with_a_random_secret_key"

# Load & preprocess once
DF = preprocess(load_data("completed_bike_dataset.xlsx"))
DF["suitability_score"] = compute_suitability_score(DF)

# Load model
MODEL = load("bike_suitability_rf.joblib")

# Geography code map (keep in sync with templates/location.html)
GEO_CODE = {
    "mountain": 0, "coastal": 1, "city": 2,
    "small_town": 3, "highway": 4, "back_roads": 5
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/skill", methods=["GET","POST"])
def skill():
    if request.method == "POST":
        session["experience_level"] = int(request.form["experience_level"])
        # No 'style' step; go straight to location
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

        # Filter & score
        candidates = filter_by_rules(DF.copy(), user)
        if candidates.empty:
            return render_template("results.html", error="No bikes match your criteria.")

        Xc = candidates[["engine_size","riding_style_code","price"]].values
        candidates = candidates.copy()
        candidates["score"] = MODEL.predict(Xc)

        top5 = candidates.sort_values("score", ascending=False).head(5)
        return render_template("results.html", bikes=top5.to_dict(orient="records"))

    return render_template("budget.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)