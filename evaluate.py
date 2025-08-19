# evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from joblib import load

from data_preprocessing import load_data, preprocess
from train_model import compute_suitability_score
from recommend import filter_by_rules

def precision_at_k(y_true, y_pred, k=5):
    return np.mean([
        len(set(true) & set(pred[:k])) / float(k)
        for true, pred in zip(y_true, y_pred)
    ])

def recall_at_k(y_true, y_pred, k=5):
    return np.mean([
        len(set(true) & set(pred[:k])) / float(len(true))
        for true, pred in zip(y_true, y_pred)
    ])

if __name__ == "__main__":
    # Load dataset + model
    df = preprocess(load_data("completed_bike_dataset.xlsx"))
    df["suitability_score"] = compute_suitability_score(df)
    model = load("bike_suitability_rf.joblib")

    # Use the same features
    X = df[["engine_size","riding_style_code","price"]].values
    y = df["suitability_score"].values

    # Model regression accuracy
    y_pred = model.predict(X)
    print(f"✅ Overall R²: {r2_score(y, y_pred):.4f}")
    print(f"✅ Overall MSE: {mean_squared_error(y, y_pred):.6f}")

    # Fake evaluation set for recommendation quality
    # Here we use the top-scored items as “predicted” and the top ground-truth
    df = df.copy()
    df["pred_score"] = y_pred

    # Treat top N by heuristic score as "true" relevant, top N by RF as "pred"
    true_ranked = df.sort_values("suitability_score", ascending=False)
    pred_ranked = df.sort_values("pred_score", ascending=False)

    y_true = [set(true_ranked.head(5).index)]
    y_pred = [list(pred_ranked.head(5).index)]

    print(f"Precision@5: {precision_at_k(y_true,y_pred,5):.4f}")
    print(f"Recall@5:    {recall_at_k(y_true,y_pred,5):.4f}")
