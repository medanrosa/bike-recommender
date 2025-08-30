import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump

from data_preprocessing import load_data, preprocess

def compute_suitability_score(df: pd.DataFrame) -> pd.Series:
    """Heuristic target for training (unchanged semantics)."""
    style_weights = {
        "supersport": 0.90, "supermoto": 0.70, "naked": 0.80,
        "touring": 0.85, "dirtbike": 0.60, "autocycle": 0.50,
        "commuter": 0.75, "cruiser": 0.82, "adventure": 0.78,
    }
    df = df.copy()
    df["style_weight"] = df["riding_style"].map(style_weights).fillna(0.50)

    eng_norm   = df["engine_size"] / max(1, df["engine_size"].max())
    price_norm = df["price"]       / max(1, df["price"].max())
    price_suit = 1 - price_norm

    score = 0.4 * eng_norm + 0.3 * price_suit + 0.3 * df["style_weight"]
    return score.clip(0.0, 1.0)

if __name__ == "__main__":
    # 1) Data
    df = preprocess(load_data("completed_bike_dataset.xlsx"))
    df["suitability_score"] = compute_suitability_score(df)

    # 2) Features (must match app.py)
    X = df[["engine_size", "riding_style_code", "price"]].values
    y = df["suitability_score"].values

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) RandomizedSearchCV over RF (no SciPy; pure Python lists)
    param_distributions = {
        "n_estimators":      [300, 500, 700, 900],
        "max_depth":         [8, 12, 16, 24, 32, None],
        "min_samples_split": [2, 4, 6, 8, 10, 12, 16],
        "min_samples_leaf":  [1, 2, 3, 4, 5],
        "max_features":      ["auto", "sqrt", 1.0],
        "bootstrap":         [True],
    }

    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=40,          # increase for more thorough search if you like
        scoring="r2",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    # 5) Test metrics
    y_pred = best_rf.predict(X_test)
    print("=== Random Forest (tuned) ===")
    print("Best params:", search.best_params_)
    print(f"✅ R² (test): {r2_score(y_test, y_pred):.4f} | MSE (test): {mean_squared_error(y_test, y_pred):.6f}")

    # 6) Refit on full data & save
    final_rf = RandomForestRegressor(**search.best_params_, random_state=42, n_jobs=-1)
    final_rf.fit(X, y)
    dump(final_rf, "bike_suitability_rf.joblib")
    print("💾 Saved model -> bike_suitability_rf.joblib")

    # 7) Feature importances
    plt.bar(["engine_size", "riding_style_code", "price"], final_rf.feature_importances_)
    plt.ylabel("Relative importance")
    plt.tight_layout()
    plt.show()