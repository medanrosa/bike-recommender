from flask import Flask, request, jsonify, send_from_directory
from joblib import load
from data_preprocessing import load_data, preprocess
from recommend import compute_suitability_score, filter_by_rules

app = Flask(__name__, static_folder="static", template_folder="templates")

# load once
model = load("bike_suitability_rf.joblib")
df = preprocess(load_data("completed_bike_dataset.xlsx"))
df["suitability_score"] = compute_suitability_score(df)

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/recommend", methods=["POST"])
def recommend_bike():
    user = request.get_json()
    candidates = filter_by_rules(df.copy(), user)
    if candidates.empty:
        return jsonify({"error": "No bikes match your criteria."}), 404
    X = candidates[["engine_size","riding_style_code","price"]]
    candidates["score"] = model.predict(X)
    top5 = candidates.sort_values("score", ascending=False).head(5)
    return jsonify(
        top5[["bike_name","engine_size","riding_style","price","score"]]
        .to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)