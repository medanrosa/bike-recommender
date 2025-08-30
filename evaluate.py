import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import r2_score, mean_squared_error

from data_preprocessing import load_data, preprocess
from train_model import compute_suitability_score

def _spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without SciPy: Pearson correlation of ranked values."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.clip(np.mean(ra * rb), -1.0, 1.0))

def precision_at_k(y_true_sets, y_pred_lists, k=5):
    return np.mean([
        len(set_true & set(pred[:k])) / float(k)
        for set_true, pred in zip(y_true_sets, y_pred_lists)
    ])

def recall_at_k(y_true_sets, y_pred_lists, k=5):
    vals = []
    for set_true, pred in zip(y_true_sets, y_pred_lists):
        denom = float(len(set_true)) if len(set_true) > 0 else 1.0
        vals.append(len(set_true & set(pred[:k])) / denom)
    return np.mean(vals)

def ndcg_at_k(y_true_sets, y_pred_lists, k=5):
    def dcg(rels): return sum(rel/np.log2(i+2) for i, rel in enumerate(rels))
    scores = []
    for set_true, pred in zip(y_true_sets, y_pred_lists):
        rel = [1 if pid in set_true else 0 for pid in pred[:k]]
        ideal = sorted(rel, reverse=True)
        denom = dcg(ideal) or 1.0
        scores.append(dcg(rel)/denom)
    return float(np.mean(scores))

if __name__ == "__main__":
    if not os.path.exists("bike_suitability_rf.joblib"):
        raise FileNotFoundError("bike_suitability_rf.joblib not found. Run `python train_model.py` first.")

    df = preprocess(load_data("completed_bike_dataset.xlsx"))
    df["suitability_score"] = compute_suitability_score(df)

    model = load("bike_suitability_rf.joblib")

    X = df[["engine_size","riding_style_code","price"]].values
    y_true = df["suitability_score"].values
    y_pred = model.predict(X)

    print(f"✅ Overall R²:  {r2_score(y_true, y_pred):.6f}")
    print(f"✅ Overall MSE: {mean_squared_error(y_true, y_pred):.6f}")
    print(f"✅ Spearman rank corr: {_spearman_rank_corr(y_true, y_pred):.6f}")

    # Top-K ranking quality vs heuristic “ground truth”
    df_eval = df.copy()
    df_eval["pred_score"] = y_pred

    M, K = 20, 5  # treat top-M by heuristic as relevant; evaluate top-K by model
    gt_top = set(df_eval.sort_values("suitability_score", ascending=False).head(M).index)
    rf_top = list(df_eval.sort_values("pred_score", ascending=False).index)

    y_true_sets  = [gt_top]
    y_pred_lists = [rf_top]

    print(f"Precision@{K}: {precision_at_k(y_true_sets, y_pred_lists, K):.4f}")
    print(f"Recall@{K}:    {recall_at_k(y_true_sets, y_pred_lists, K):.4f}")
    print(f"NDCG@{K}:      {ndcg_at_k(y_true_sets, y_pred_lists, K):.4f}")