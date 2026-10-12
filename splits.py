"""Purged zero-leakage time splitting and walk-forward CV generator."""

from typing import Iterator
import pandas as pd


def get_train_test_split(
    origins: pd.DatetimeIndex,
    data_end: pd.Timestamp,
    test_days: int = 90,
    horizon_hours: int = 3,
    warmup_days: int = 30,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    """Split hourly origins into train (pre-test) and test sets (PRD §4).
    
    Guarantees:
    - Test window: origins t in [data_end - 90 days, data_end - H]
    - Train window: origins t with t + H <= test_start (purged label horizon)
    - Warm-up: first 30 days of data excluded from train origins
    
    Returns:
        (train_origins, test_origins, test_start, test_end)
    """
    if origins.tz is not None:
        origins = origins.tz_localize(None)
    if data_end.tzinfo is not None:
        data_end = data_end.tz_localize(None)

    data_end_hour = data_end.floor("h")
    test_start = data_end_hour - pd.Timedelta(days=test_days)
    test_end = data_end_hour - pd.Timedelta(hours=horizon_hours)

    data_start = origins.min()
    warmup_end = data_start + pd.Timedelta(days=warmup_days)

    # Train origins: after warm-up, and label window fully realized before test_start
    train_mask = (origins >= warmup_end) & (origins <= test_start - pd.Timedelta(hours=horizon_hours))
    test_mask = (origins >= test_start) & (origins <= test_end)

    train_origins = origins[train_mask]
    test_origins = origins[test_mask]

    return train_origins, test_origins, test_start, test_end


def walk_forward_folds(
    train_origins: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    horizon_hours: int = 3,
    n_folds: int = 5,
    valid_days: int = 28,
    train_window_days: int | None = None,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate expanding or trailing walk-forward folds on pre-test data (PRD §5.5).
    
    Guarantees:
    - Purged label horizon: train.max() + H <= valid.min()
    - Test isolation: valid.max() <= test_start - H
    - No fold crosses into test period.
    """
    H = pd.Timedelta(hours=horizon_hours)
    step = pd.Timedelta(days=valid_days)

    for k in range(n_folds):
        valid_end_k = test_start - H - k * step
        valid_start_k = valid_end_k - step

        valid_mask = (train_origins > valid_start_k) & (train_origins <= valid_end_k)
        valid_fold = train_origins[valid_mask]

        # Train rows must have their label fully realized by valid_start_k
        train_cutoff = valid_start_k - H
        if train_window_days is not None:
            train_window_start = valid_start_k - pd.Timedelta(days=train_window_days)
            train_mask = (train_origins <= train_cutoff) & (train_origins >= train_window_start)
        else:
            train_mask = train_origins <= train_cutoff

        train_fold = train_origins[train_mask]

        if len(train_fold) > 0 and len(valid_fold) > 0:
            yield train_fold, valid_fold
