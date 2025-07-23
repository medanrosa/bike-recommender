import pandas as pd
from joblib import load
from data_preprocessing import load_data, preprocess
from ontology_questions import ask_user

def compute_suitability_score(df: pd.DataFrame) -> pd.Series:
    # same as in train_model.py
    style_weights = {
        "supersport": 0.9, "supermoto": 0.7, "naked": 0.8,
        "touring":    0.85,"dirtbike":0.6,"autocycle":0.5
    }
    df["style_weight"] = df["riding_style"].map(style_weights).fillna(0.5)
    eng_norm   = df["engine_size"] / df["engine_size"].max()
    price_norm = df["price"] / df["price"].max()
    price_suit = 1 - price_norm
    return (0.4 * eng_norm + 0.3 * price_suit + 0.3 * df["style_weight"]).clip(0.0, 1.0)

def filter_by_rules(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    # beginner → ≤500cc
    if user["experience_level"] == 0:
        df = df[df.engine_size <= 500]
    # geography rules
    geo = user["geography"]
    if geo == 4:  # highway
        df = df[~df.riding_style.isin(["dirtbike", "supermoto"])]
    elif geo in [0, 5]:  # mountain/back_roads
        df = df[df.riding_style.isin(["dirtbike","touring","supermoto"])]
    elif geo == 2:  # city
        df = df[df.riding_style.isin(["naked","supermoto"])]
    # budget cap
    df = df[df.price <= user["budget"]]
    return df

if __name__ == "__main__":
    df = load_data("completed_bike_dataset.xlsx")
    df = preprocess(df)
    df["suitability_score"] = compute_suitability_score(df)
    model = load("bike_suitability_rf.joblib")
    user  = ask_user()
    candidates = filter_by_rules(df, user)
    if candidates.empty:
        print("🚨 No bikes match your criteria—try relaxing budget or preferences.")
        exit()
    X = candidates[["engine_size","riding_style_code","price"]]
    candidates["score"] = model.predict(X)
    top5 = candidates.sort_values("score", ascending=False).head(5)
    print("\nTop 5 recommendations:\n")
    print(top5[["bike_name","engine_size","riding_style","price","score"]].to_string(index=False))