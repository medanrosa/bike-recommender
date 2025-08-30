import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error

from data_preprocessing import load_data, preprocess
from train_model import compute_suitability_score

def precision_at_k(y_true_sets, y_pred_lists, k=5):
    return np.mean([
        len(set_true & set(pred[:k])) / float(k)
        for set_true, pred in zip(y_true_sets, y_pred_lists)
    ])

def recall_at_k(y_true_sets, y_pred_lists, k=5):
    return np.mean([
        len(set_true & set(pred[:k])) / float(len(set_true) if len(set_true)>0 else 1)
        for set_true, pred in zip(y_true_sets, y_pred_lists)
    ])

def ndcg_at_k(y_true_sets, y_pred_lists, k=5):
    """Binary relevance NDCG@K against a set of relevant items."""
    def dcg(relevances):
        return sum(rel/np.log2(idx+2) for idx, rel in enumerate(relevances))
    scores = []
    for true_set, pred in zip(y_true_sets, y_pred_lists):
        pred_topk = pred[:k]
        rel = [1 if pid in true_set else 0 for pid in pred_topk]
        ideal = sorted(rel, reverse=True)
        denom = dcg(ideal) or 1.0
        scores.append(dcg(rel)/denom)
    return float(np.mean(scores))

if __name__ == "__main__":
    # 1) Data & model
    df = preprocess(load_data("completed_bike_dataset.xlsx"))
    df["suitability_score"] = compute_suitability_score(df)
    model = load("bike_suitability_rf.joblib")

    # 2) Predict on full dataset (model fit quality)
    X = df[["engine_size","riding_style_code","price"]].values
    y_true = df["suitability_score"].values
    y_pred = model.predict(X)

    print(f"✅ Overall R²:  {r2_score(y_true, y_pred):.6f}")
    print(f"✅ Overall MSE: {mean_squared_error(y_true, y_pred):.6f}")
    rho, _ = spearmanr(y_true, y_pred)
    print(f"✅ Spearman rank corr: {rho:.6f}")

    # 3) Recommendation quality (top-K) vs heuristic ground truth
    df_eval = df.copy()
    df_eval["pred_score"] = y_pred

    # Treat top-M by heuristic as "relevant"; evaluate how many the RF puts in top-K
    M, K = 20, 5
    gt_top = set(df_eval.sort_values("suitability_score", ascending=False).head(M).index)
    rf_top = list(df_eval.sort_values("pred_score", ascending=False).index)

    y_true_sets = [gt_top]            # single global scenario
    y_pred_lists = [rf_top]

    print(f"Precision@{K}: {precision_at_k(y_true_sets,y_pred_lists,K):.4f}")
    print(f"Recall@{K}:    {recall_at_k(y_true_sets,y_pred_lists,K):.4f}")
    print(f"NDCG@{K}:      {ndcg_at_k(y_true_sets,y_pred_lists,K):.4f}")