"""Data loading, cleaning, as-of filtering, and sanity feature extraction for AlertCast."""

import os
from pathlib import Path
from typing import Sequence
import pandas as pd
import numpy as np

# Excluded administrative units per PRD (permanent siren / degenerate target)
EXCLUDED_REGIONS = [
    "Luhanska oblast",
    "Crimea",
    "Autonomous Republic of Crimea",
    "Sevastopol",
]


def load_and_clean_data(
    csv_path: str = "data/volunteer_data_en.csv",
    output_parquet: str | None = "data/interim/alerts.parquet",
) -> pd.DataFrame:
    """Load raw volunteer alert data and apply strictly row-local cleaning.
    
    Cleaning rules (PRD §3.2):
    - Row-local and deterministic (no future peeking or cross-row interval merging)
    - Parse started_at, finished_at as UTC -> tz-naive datetime64[ns]
    - Drop exact duplicates (oblast, started_at, finished_at)
    - Drop rows with finished_at < started_at
    - Impute missing finished_at with started_at + 30 min (naive=True)
    - Exclude permanent-siren regions (Luhansk, Crimea)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column naming
    if "region" in df.columns:
        df = df.rename(columns={"region": "oblast"})

    required_cols = {"oblast", "started_at", "finished_at"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Exclude permanent-siren regions
    df = df[~df["oblast"].isin(EXCLUDED_REGIONS)].copy()

    # Parse timestamps to UTC and strip timezone to tz-naive datetime64[ns]
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True).dt.tz_localize(None)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True).dt.tz_localize(None)

    # Impute missing finished_at (started_at + 30m, naive=True)
    missing_end = df["finished_at"].isna()
    if missing_end.any():
        df.loc[missing_end, "finished_at"] = df.loc[missing_end, "started_at"] + pd.Timedelta(minutes=30)
        df.loc[missing_end, "naive"] = True

    if "naive" not in df.columns:
        df["naive"] = False
    else:
        df["naive"] = df["naive"].fillna(False).astype(bool)

    # Drop rows where finished_at < started_at (invalid chronologies)
    valid_interval = df["finished_at"] >= df["started_at"]
    df = df[valid_interval].copy()

    # Drop exact duplicate rows
    df = df.drop_duplicates(subset=["oblast", "started_at", "finished_at"]).copy()

    # Type casting and sorting
    df["oblast"] = df["oblast"].astype("category")
    df = df.sort_values(by=["started_at", "oblast"]).reset_index(drop=True)

    columns_order = ["oblast", "started_at", "finished_at", "naive"]
    df = df[columns_order]

    if output_parquet:
        out_path = Path(output_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    return df


def get_modeled_oblasts(df: pd.DataFrame) -> list[str]:
    """Return sorted list of modeled oblast names."""
    return sorted(list(df["oblast"].unique()))


def as_of_view(raw: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Return only what was knowable at forecast origin time t (PRD §5.2).
    
    Rules:
    - keep rows with started_at <= t
    - mask finished_at > t as NaT (alert is still open as far as knowable at t)
    - tz-naive UTC timestamps only
    
    Returns:
        DataFrame with columns: [oblast, started_at, finished_at, naive]
    """
    t = pd.to_datetime(t)
    if t.tzinfo is not None:
        t = t.tz_localize(None)

    # Alert start must have happened on or before t
    view = raw[raw["started_at"] <= t].copy()

    # Mask any future end time as NaT (ongoing / open alert at origin t)
    future_end_mask = view["finished_at"] > t
    if future_end_mask.any():
        view.loc[future_end_mask, "finished_at"] = pd.NaT

    return view[["oblast", "started_at", "finished_at", "naive"]]


def build_feature_matrix(
    raw: pd.DataFrame,
    origins: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    oblasts: Sequence[str],
) -> pd.DataFrame:
    """Compute minimal sanity feature matrix (is_active_now) from as_of_view for Block 1.
    
    Index: MultiIndex (origin, oblast)
    Columns: ['is_active_now'] (int: 1 if active at t, 0 otherwise)
    """
    rows = []
    for origin in origins:
        t = pd.to_datetime(origin)
        if t.tzinfo is not None:
            t = t.tz_localize(None)

        view = as_of_view(raw, t)

        # An alert is active at t iff it started <= t and its end is NaT (masked because end > t)
        # or end is strictly greater than t.
        active_mask = view["finished_at"].isna() | (view["finished_at"] > t)
        active_oblasts = set(view.loc[active_mask, "oblast"].unique())

        for o in oblasts:
            rows.append({
                "origin": t,
                "oblast": o,
                "is_active_now": 1 if o in active_oblasts else 0,
            })

    result = pd.DataFrame(rows)
    if result.empty:
        index = pd.MultiIndex.from_tuples([], names=["origin", "oblast"])
        return pd.DataFrame(columns=["is_active_now"], index=index)

    return result.set_index(["origin", "oblast"]).sort_index()
