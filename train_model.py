import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump
from data_preprocessing import load_data, preprocess

def compute_suitability_score(df: pd.DataFrame) -> pd.Series:
    style_weights = {
        "supersport": 0.9, "supermoto": 0.7, "naked": 0.8,
        "touring":    0.85,"dirtbike":0.6,"autocycle":0.5
    }
    df["style_weight"] = df["riding_style"].map(style_weights).fillna(0.5)
    eng_norm   = df["engine_size"] / df["engine_size"].max()
    price_norm = df["price"] / df["price"].max()
    price_suit = 1 - price_norm
    score = 0.4 * eng_norm + 0.3 * price_suit + 0.3 * df["style_weight"]
    return score.clip(0.0, 1.0)

if __name__ == "__main__":
    # load & preprocess
    df = load_data("completed_bike_dataset.xlsx")
    df = preprocess(df)
    # compute target
    df["suitability_score"] = compute_suitability_score(df)
    X = df[["engine_size", "riding_style_code", "price"]]
    y = df["suitability_score"]
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    dump(model, "bike_suitability_rf.joblib")
    # evaluate
    y_pred = model.predict(X_test)
    print(f"✅ R²: {r2_score(y_test, y_pred):.4f} | MSE: {mean_squared_error(y_test, y_pred):.6f}")
    # feature importances
    plt.bar(["engine_size","riding_style_code","price"], model.feature_importances_)
    plt.ylabel("Relative importance")
    plt.tight_layout()
    plt.show()