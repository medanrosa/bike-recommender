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

# Mapping from form strings to codes
STYLE_CODE = {
    "supersport": 0, "supermoto": 1,
    "naked": 2,      "touring":    3,
    "dirtbike": 4
}
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
        return redirect(url_for("style"))
    return render_template("skill.html")


@app.route("/style", methods=["GET","POST"])
def style():
    if request.method == "POST":
        session["preferred_style"] = STYLE_CODE[request.form["preferred_style"]]
        return redirect(url_for("location"))
    return render_template("style.html")


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

        # Build user dict
        user = {
            "experience_level": session["experience_level"],
            "preferred_style":  session["preferred_style"],
            "geography":        session["geography"],
            "budget":           session["budget"]
        }

        # Filter & score
        candidates = filter_by_rules(DF.copy(), user)
        if candidates.empty:
            return render_template("results.html", error="No bikes match your criteria.")

        Xc = candidates[["engine_size","riding_style_code","price"]].values
        candidates["score"] = MODEL.predict(Xc)
        top5 = candidates.sort_values("score", ascending=False).head(5)

        return render_template(
            "results.html",
            bikes=top5.to_dict(orient="records")
        )

    return render_template("budget.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)