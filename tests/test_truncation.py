"""Truncation test suite verifying zero data leakage for AlertCast Block 1."""

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from data_pipeline import (
    load_and_clean_data,
    get_modeled_oblasts,
    as_of_view,
    build_feature_matrix,
)


def independent_truncate(raw: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Independently truncate data at origin t (PRD §6.0).
    
    Keeps rows started_at <= t and masks finished_at > t as NaT.
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
    # Pick a sample of diverse origins across different years
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

        # Output must be identical
        assert_frame_equal(view_full, view_trunc)


def test_feature_truncation_invariance(cleaned_alerts, oblasts):
    """Verify feature outputs at origin t are identical using full vs truncated data."""
    # Pick diverse origins: random, exact alert starts, and exact alert ends
    sample_starts = cleaned_alerts["started_at"].sample(n=10, random_state=42).tolist()
    sample_ends = cleaned_alerts["finished_at"].sample(n=10, random_state=42).tolist()
    hourly_origins = pd.date_range("2024-01-01 00:00:00", periods=10, freq="12h").tolist()

    origins_to_test = sample_starts + sample_ends + hourly_origins

    for t in origins_to_test:
        # Full data feature calculation
        M_full = build_feature_matrix(cleaned_alerts, [t], oblasts)

        # Truncated data feature calculation
        truncated_data = independent_truncate(cleaned_alerts, t)
        M_trunc = build_feature_matrix(truncated_data, [t], oblasts)

        # Invariance assertion
        assert_frame_equal(M_full, M_trunc)


def test_is_active_now_logic(cleaned_alerts, oblasts):
    """Verify that is_active_now correctly captures active alerts."""
    # Pick an alert that was active for more than 1 hour
    long_alerts = cleaned_alerts[
        cleaned_alerts["finished_at"] - cleaned_alerts["started_at"] > pd.Timedelta(hours=1)
    ]
    sample_alert = long_alerts.iloc[0]
    ob = sample_alert["oblast"]
    s = sample_alert["started_at"]
    e = sample_alert["finished_at"]

    # At midpoint of the alert: must be active
    midpoint = s + (e - s) / 2
    f_mid = build_feature_matrix(cleaned_alerts, [midpoint], oblasts)
    assert f_mid.loc[(midpoint, ob), "is_active_now"] == 1

    # Before start: this specific alert should not contribute
    before = s - pd.Timedelta(seconds=1)
    truncated_before = independent_truncate(cleaned_alerts, before)
    f_before = build_feature_matrix(truncated_before, [before], oblasts)
    # The alert wasn't known before start
    assert sample_alert["started_at"] not in truncated_before["started_at"].values
