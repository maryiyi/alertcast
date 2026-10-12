"""Comprehensive truncation and zero-leakage test suite for AlertCast Block 2."""

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from data_pipeline import (
    load_and_clean_data,
    get_modeled_oblasts,
    as_of_view,
)
from features import (
    build_feature_matrix,
    compute_labels,
    FEATURE_NAMES,
)
from splits import (
    get_train_test_split,
    walk_forward_folds,
)


def independent_truncate(raw: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Independently truncate data at origin t (PRD §6.0).
    
    Keeps rows with started_at <= t and masks finished_at > t as NaT.
    Written independently inside test suite to verify pipeline integrity.
    """
    out = raw[raw["started_at"] <= t].copy()
    out.loc[out["finished_at"] > t, "finished_at"] = pd.NaT
    return out


@pytest.fixture(scope="session")
def cleaned_alerts():
    """Load cleaned alert data once for test session."""
    return load_and_clean_data(csv_path="data/volunteer_data_en.csv", output_parquet=None)


@pytest.fixture(scope="session")
def oblasts(cleaned_alerts):
    """Retrieve modeled oblasts."""
    return get_modeled_oblasts(cleaned_alerts)


def test_data_cleaning_contract(cleaned_alerts):
    """Verify data cleaning guarantees."""
    assert not cleaned_alerts.empty
    # No Luhansk or Crimea
    assert "Luhanska oblast" not in cleaned_alerts["oblast"].values
    assert "Crimea" not in cleaned_alerts["oblast"].values
    # Valid intervals: finished_at >= started_at
    assert (cleaned_alerts["finished_at"] >= cleaned_alerts["started_at"]).all()
    # No exact duplicates
    assert not cleaned_alerts.duplicated(subset=["oblast", "started_at", "finished_at"]).any()
    # Timestamps are tz-naive
    assert cleaned_alerts["started_at"].dt.tz is None
    assert cleaned_alerts["finished_at"].dt.tz is None


def test_as_of_view_truncation_invariance(cleaned_alerts):
    """Verify as_of_view returns identical views whether fed full or truncated data."""
    test_origins = [
        pd.Timestamp("2022-03-01 12:00:00"),
        pd.Timestamp("2023-01-15 03:00:00"),
        pd.Timestamp("2024-06-20 18:00:00"),
        pd.Timestamp("2025-11-10 09:00:00"),
        pd.Timestamp("2026-08-01 22:00:00"),
    ]

    for t in test_origins:
        view_full = as_of_view(cleaned_alerts, t).reset_index(drop=True)
        truncated_raw = independent_truncate(cleaned_alerts, t)
        view_trunc = as_of_view(truncated_raw, t).reset_index(drop=True)

        assert_frame_equal(view_full, view_trunc)


def test_feature_truncation_invariance_all_features(cleaned_alerts, oblasts):
    """The Contract (PRD §6.1 / §6.2): verify all 32 features are invariant under truncation."""
    # Test sample of edge origins: alert starts, ends, and DST transition
    sample_starts = cleaned_alerts["started_at"].sample(n=5, random_state=42).tolist()
    sample_ends = cleaned_alerts["finished_at"].sample(n=5, random_state=42).tolist()
    hourly_origins = [
        pd.Timestamp("2024-03-31 01:00:00"),  # DST transition
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2026-07-20 15:00:00"),
    ]

    origins_to_test = sample_starts + sample_ends + hourly_origins

    for t in origins_to_test:
        M_full = build_feature_matrix(cleaned_alerts, [t], oblasts)
        truncated_raw = independent_truncate(cleaned_alerts, t)
        M_trunc = build_feature_matrix(truncated_raw, [t], oblasts)

        # Assert bit-for-bit equality across all 32 feature columns
        assert_frame_equal(M_full, M_trunc)


def test_target_label_contract(cleaned_alerts, oblasts):
    """Verify target calculation y(o, t): alert start in (t, t + 3h]."""
    data_end = cleaned_alerts["started_at"].max()
    test_origins = pd.date_range("2024-05-01 00:00:00", periods=5, freq="h")

    y_df = compute_labels(cleaned_alerts, test_origins, oblasts, horizon_hours=3)
    assert not y_df.empty
    assert set(y_df["y"].unique()).issubset({0, 1})
    assert y_df.index.names == ["origin", "oblast"]

    # Verify edge: origin right at the end of dataset must drop incomplete horizons
    incomplete_origin = [data_end - pd.Timedelta(hours=1)]
    y_incomplete = compute_labels(cleaned_alerts, incomplete_origin, oblasts, horizon_hours=3)
    assert y_incomplete.empty


def test_train_test_split_and_purge_guarantees(cleaned_alerts):
    """Verify zero data leakage in splitting and walk-forward cross validation."""
    data_end = cleaned_alerts["started_at"].max()
    all_origins = pd.date_range("2022-02-25 00:00:00", data_end.floor("h"), freq="h")

    train_origins, test_origins, test_start, test_end = get_train_test_split(
        all_origins, data_end, test_days=90, horizon_hours=3, warmup_days=30
    )

    # 1. Test set is strictly the last 90 days
    assert test_origins.min() == test_start
    assert test_origins.max() <= data_end - pd.Timedelta(hours=3)
    assert (test_origins.max() - test_origins.min()).days >= 89

    # 2. Train set label window is strictly purged: t_train + H <= test_start
    assert train_origins.max() + pd.Timedelta(hours=3) <= test_start

    # 3. Walk-forward cross validation folds guarantee purged training window
    folds = list(walk_forward_folds(train_origins, test_start, horizon_hours=3, n_folds=5, valid_days=28))
    assert len(folds) == 5

    for tr_fold, val_fold in folds:
        # Train fold label window fully realized before valid fold begins
        assert tr_fold.max() + pd.Timedelta(hours=3) <= val_fold.min()
        # Validation fold never touches test period
        assert val_fold.max() <= test_start - pd.Timedelta(hours=3)
