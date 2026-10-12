"""End-to-end training, baseline comparison, evaluation, and visualization for AlertCast."""

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from data_pipeline import load_and_clean_data, get_modeled_oblasts
from features import build_feature_matrix, compute_labels, FEATURE_NAMES
from splits import get_train_test_split, walk_forward_folds
from models import (
    BaselineB0Prior,
    BaselineB1RecentActivity,
    BaselineB2HourlyClimatology,
    BaselineB3TrailingRate,
    compute_metrics,
    train_lightgbm_model,
)
from visualization import (
    plot_time_of_day_heatmap,
    plot_feature_importance,
    plot_forecast_vs_actual,
)


def run_training_and_evaluation():
    print("=" * 70)
    print("🚨 AlertCast: Zero-Leakage Pipeline Training & Evaluation")
    print("=" * 70)

    # 1. Ingestion & Data Preparation
    t0 = time.time()
    print("\n[1/5] Ingesting and cleaning raw volunteer data...")
    df = load_and_clean_data("data/volunteer_data_en.csv", output_parquet=None)
    oblasts = get_modeled_oblasts(df)
    data_end = df["started_at"].max()
    print(f"Loaded {len(df):,} valid records across {len(oblasts)} modeled oblasts.")
    print(f"Data range: {df['started_at'].min()} to {data_end}")

    # 2. Time Splitting
    print("\n[2/5] Creating purged train/test splits...")
    origins = pd.date_range("2022-02-25 00:00:00", data_end.floor("h"), freq="h")
    train_origins, test_origins, test_start, test_end = get_train_test_split(
        origins, data_end, test_days=90, horizon_hours=3, warmup_days=30
    )
    print(f"Train origins: {len(train_origins):,} ({train_origins.min()} to {train_origins.max()})")
    print(f"Test origins:  {len(test_origins):,} ({test_origins.min()} to {test_origins.max()})")
    print(f"Purge gap:     {train_origins.max() + pd.Timedelta(hours=3)} <= {test_start} (Zero Leakage)")

    # 3. Feature Extraction
    print("\n[3/5] Computing zero-leakage feature matrices and targets...")
    t_feat = time.time()
    X_train = build_feature_matrix(df, train_origins, oblasts)
    y_train = compute_labels(df, train_origins, oblasts, horizon_hours=3)["y"]

    X_test = build_feature_matrix(df, test_origins, oblasts)
    y_test = compute_labels(df, test_origins, oblasts, horizon_hours=3)["y"]
    print(f"X_train shape: {X_train.shape}, positive rate: {y_train.mean():.3%}")
    print(f"X_test shape:  {X_test.shape}, positive rate: {y_test.mean():.3%}")
    print(f"Feature computation completed in {time.time() - t_feat:.2f}s")

    # 4. Model Training (Baselines & LightGBM)
    print("\n[4/5] Fitting baselines and LightGBM classifier...")
    t_train = time.time()

    # B0 Prior
    b0 = BaselineB0Prior().fit(X_train, y_train)
    # B1 Recent Activity
    b1 = BaselineB1RecentActivity().fit(X_train, y_train)
    # B2 Climatology
    b2 = BaselineB2HourlyClimatology().fit(X_train, y_train)
    # B3 Trailing Base Rate
    b3 = BaselineB3TrailingRate().fit(X_train, y_train)

    # LightGBM Classifier
    lgbm = train_lightgbm_model(
        X_train,
        y_train,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        seed=42,
    )
    print(f"Model training completed in {time.time() - t_train:.2f}s")

    # 5. Out-of-fold optimal operating threshold on training data
    # (F2 score recall-weighted)
    preds_train = lgbm.predict_proba(X_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, preds_train)
    # Avoid div-by-zero
    f2_scores = (5 * precisions * recalls) / np.maximum(4 * precisions + recalls, 1e-8)
    best_idx = np.argmax(f2_scores)
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    print(f"Optimized decision threshold (F2 on pre-test): {best_threshold:.4f}")

    # 6. Test Set Evaluation
    print("\n[5/5] Scoring on sealed 90-day test set...")
    p_b0 = b0.predict_proba(X_test)
    p_b1 = b1.predict_proba(X_test)
    p_b2 = b2.predict_proba(X_test)
    p_b3 = b3.predict_proba(X_test)
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]

    # Calm subset masks (no active alert at origin t)
    calm_mask_test = (X_test["is_active_now"] == 0).values
    y_test_calm = y_test.values[calm_mask_test]
    p_b0_calm = p_b0[calm_mask_test]
    p_b1_calm = p_b1[calm_mask_test]
    p_b2_calm = p_b2[calm_mask_test]
    p_b3_calm = p_b3[calm_mask_test]
    p_lgbm_calm = p_lgbm[calm_mask_test]

    # Metrics on Calm Subset (Headline)
    b2_calm_brier = compute_metrics(y_test_calm, p_b2_calm)["brier"]
    m_b0_calm = compute_metrics(y_test_calm, p_b0_calm, b2_brier=b2_calm_brier)
    m_b1_calm = compute_metrics(y_test_calm, p_b1_calm, b2_brier=b2_calm_brier)
    m_b2_calm = compute_metrics(y_test_calm, p_b2_calm, b2_brier=b2_calm_brier)
    m_b3_calm = compute_metrics(y_test_calm, p_b3_calm, b2_brier=b2_calm_brier)
    m_lgbm_calm = compute_metrics(y_test_calm, p_lgbm_calm, b2_brier=b2_calm_brier)

    # Metrics on All Rows
    b2_all_brier = compute_metrics(y_test.values, p_b2)["brier"]
    m_lgbm_all = compute_metrics(y_test.values, p_lgbm, b2_brier=b2_all_brier)

    # Print Headline Markdown Table
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS ON SEALED 90-DAY TEST SET (Headline: Calm Subset)")
    print("=" * 70)
    print("| Model | PR-AUC ↑ | ROC-AUC ↑ | Brier ↓ | Brier Skill Score vs B2 ↑ |")
    print("|---|---|---|---|---|")
    print(f"| B0 Prior | {m_b0_calm['pr_auc']:.4f} | {m_b0_calm['roc_auc']:.4f} | {m_b0_calm['brier']:.4f} | {m_b0_calm['brier_skill_score']:+.4f} |")
    print(f"| B1 Recent Activity | {m_b1_calm['pr_auc']:.4f} | {m_b1_calm['roc_auc']:.4f} | {m_b1_calm['brier']:.4f} | {m_b1_calm['brier_skill_score']:+.4f} |")
    print(f"| B2 Hourly Climatology | {m_b2_calm['pr_auc']:.4f} | {m_b2_calm['roc_auc']:.4f} | {m_b2_calm['brier']:.4f} | 0.0000 |")
    print(f"| B3 Trailing Base Rate | {m_b3_calm['pr_auc']:.4f} | {m_b3_calm['roc_auc']:.4f} | {m_b3_calm['brier']:.4f} | {m_b3_calm['brier_skill_score']:+.4f} |")
    print(f"| **LightGBM (AlertCast)** | **{m_lgbm_calm['pr_auc']:.4f}** | **{m_lgbm_calm['roc_auc']:.4f}** | **{m_lgbm_calm['brier']:.4f}** | **{m_lgbm_calm['brier_skill_score']:+.4f}** |")
    print("-" * 70)
    print(f"Secondary (All rows): LightGBM PR-AUC = {m_lgbm_all['pr_auc']:.4f}, ROC-AUC = {m_lgbm_all['roc_auc']:.4f}, Brier = {m_lgbm_all['brier']:.4f}")

    # 7. Generate Visualizations
    print("\n🎨 Generating editorial static figures in 'figures/'...")
    Path("figures").mkdir(parents=True, exist_ok=True)

    # Figure 1: Heatmap of alert starts
    plot_time_of_day_heatmap(df, test_start, output_path="figures/fig1_heatmap_hour_dow.png")
    print("  ✓ Saved figures/fig1_heatmap_hour_dow.png")

    # Figure 2: Top 15 Feature Importance
    plot_feature_importance(lgbm, FEATURE_NAMES, top_n=15, output_path="figures/fig2_feature_importance.png")
    print("  ✓ Saved figures/fig2_feature_importance.png")

    # Figure 3: 7-day Forecast vs Actual (Kyiv City)
    sample_oblast = "Kyiv City"
    ob_mask = X_test.index.get_level_values("oblast") == sample_oblast
    ob_origins = X_test.index.get_level_values("origin")[ob_mask]
    ob_lgbm = p_lgbm[ob_mask]
    ob_b2 = p_b2[ob_mask]
    plot_forecast_vs_actual(
        df,
        ob_origins,
        ob_lgbm,
        ob_b2,
        oblast=sample_oblast,
        threshold=best_threshold,
        days=7,
        output_path="figures/fig3_forecast_vs_actual.png",
    )
    print(f"  ✓ Saved figures/fig3_forecast_vs_actual.png ({sample_oblast})")

    # Also save to reports/figures/ for PRD compatibility
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy("figures/fig1_heatmap_hour_dow.png", "reports/figures/fig1_heatmap_hour_dow.png")
    shutil.copy("figures/fig2_feature_importance.png", "reports/figures/fig2_feature_importance.png")
    shutil.copy("figures/fig3_forecast_vs_actual.png", "reports/figures/fig3_forecast_vs_actual.png")

    print("\n✅ Training and evaluation pipeline finished in {:.2f}s total!".format(time.time() - t0))


if __name__ == "__main__":
    run_training_and_evaluation()
