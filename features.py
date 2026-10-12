"""Vectorized zero-leakage feature engineering and target calculation for AlertCast."""

from typing import Sequence
import pandas as pd
import numpy as np

# Adjacency map of modeled Ukrainian administrative units (land borders).
# Kyiv City borders Kyivska oblast. Luhanska oblast and Crimea are excluded.
ADJACENCY = {
    "Kyiv City": ["Kyivska oblast"],
    "Kyivska oblast": [
        "Kyiv City",
        "Chernihivska oblast",
        "Poltavska oblast",
        "Cherkaska oblast",
        "Vinnytska oblast",
        "Zhytomyrska oblast",
    ],
    "Cherkaska oblast": [
        "Kyivska oblast",
        "Poltavska oblast",
        "Kirovohradska oblast",
        "Vinnytska oblast",
    ],
    "Chernihivska oblast": ["Kyivska oblast", "Sumska oblast", "Poltavska oblast"],
    "Chernivetska oblast": [
        "Ivano-Frankivska oblast",
        "Ternopilska oblast",
        "Khmelnytska oblast",
        "Vinnytska oblast",
    ],
    "Dnipropetrovska oblast": [
        "Poltavska oblast",
        "Kharkivska oblast",
        "Donetska oblast",
        "Zaporizka oblast",
        "Khersonska oblast",
        "Mykolaivska oblast",
        "Kirovohradska oblast",
    ],
    "Donetska oblast": [
        "Kharkivska oblast",
        "Dnipropetrovska oblast",
        "Zaporizka oblast",
    ],
    "Ivano-Frankivska oblast": [
        "Lvivska oblast",
        "Ternopilska oblast",
        "Chernivetska oblast",
        "Zakarpatska oblast",
    ],
    "Kharkivska oblast": [
        "Sumska oblast",
        "Poltavska oblast",
        "Dnipropetrovska oblast",
        "Donetska oblast",
    ],
    "Khersonska oblast": [
        "Mykolaivska oblast",
        "Dnipropetrovska oblast",
        "Zaporizka oblast",
    ],
    "Khmelnytska oblast": [
        "Rivnenska oblast",
        "Zhytomyrska oblast",
        "Vinnytska oblast",
        "Chernivetska oblast",
        "Ternopilska oblast",
    ],
    "Kirovohradska oblast": [
        "Cherkaska oblast",
        "Poltavska oblast",
        "Dnipropetrovska oblast",
        "Mykolaivska oblast",
        "Odeska oblast",
        "Vinnytska oblast",
    ],
    "Lvivska oblast": [
        "Volynska oblast",
        "Rivnenska oblast",
        "Ternopilska oblast",
        "Ivano-Frankivska oblast",
        "Zakarpatska oblast",
    ],
    "Mykolaivska oblast": [
        "Odeska oblast",
        "Kirovohradska oblast",
        "Dnipropetrovska oblast",
        "Khersonska oblast",
    ],
    "Odeska oblast": [
        "Vinnytska oblast",
        "Kirovohradska oblast",
        "Mykolaivska oblast",
    ],
    "Poltavska oblast": [
        "Chernihivska oblast",
        "Sumska oblast",
        "Kharkivska oblast",
        "Dnipropetrovska oblast",
        "Kirovohradska oblast",
        "Cherkaska oblast",
        "Kyivska oblast",
    ],
    "Rivnenska oblast": [
        "Volynska oblast",
        "Zhytomyrska oblast",
        "Khmelnytska oblast",
        "Ternopilska oblast",
        "Lvivska oblast",
    ],
    "Sumska oblast": ["Chernihivska oblast", "Poltavska oblast", "Kharkivska oblast"],
    "Ternopilska oblast": [
        "Rivnenska oblast",
        "Khmelnytska oblast",
        "Chernivetska oblast",
        "Ivano-Frankivska oblast",
        "Lvivska oblast",
    ],
    "Vinnytska oblast": [
        "Zhytomyrska oblast",
        "Kyivska oblast",
        "Cherkaska oblast",
        "Kirovohradska oblast",
        "Odeska oblast",
        "Chernivetska oblast",
        "Khmelnytska oblast",
    ],
    "Volynska oblast": ["Rivnenska oblast", "Lvivska oblast"],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
    "Zaporizka oblast": [
        "Dnipropetrovska oblast",
        "Donetska oblast",
        "Khersonska oblast",
    ],
    "Zhytomyrska oblast": [
        "Kyivska oblast",
        "Vinnytska oblast",
        "Khmelnytska oblast",
        "Rivnenska oblast",
    ],
}

FEATURE_NAMES = [
    # Identity & Calendar
    "oblast_id",
    "hour_local",
    "dow_local",
    "is_weekend",
    "hour_utc",
    # State
    "is_active_now",
    "current_alert_age_min",
    "n_open_alerts_now",
    # Recency
    "mins_since_last_start",
    "mins_since_last_end",
    "last_alert_duration_min",
    # History
    "starts_1h",
    "starts_3h",
    "starts_6h",
    "starts_12h",
    "starts_24h",
    "starts_72h",
    "starts_7d",
    "alert_min_6h",
    "alert_min_24h",
    "alert_min_7d",
    "start_rate_30d",
    # Neighbors
    "nbr_active_cnt",
    "nbr_active_share",
    "nbr_starts_1h",
    "nbr_starts_3h",
    "nbr_starts_6h",
    "nbr_min_since_last_start",
    # National
    "nat_active_cnt",
    "nat_starts_1h",
    "nat_starts_3h",
    "nat_start_rate_30d",
]


def _compute_calendar_features(origins: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Compute local Kyiv and UTC calendar features deterministically from origins."""
    s = pd.Series(origins)
    hour_utc = origins.hour.to_numpy(dtype=np.int32)
    kyiv_times = origins.tz_localize("UTC").tz_convert("Europe/Kyiv")
    hour_local = kyiv_times.hour.to_numpy(dtype=np.int32)
    dow_local = kyiv_times.dayofweek.to_numpy(dtype=np.int32)
    is_weekend = (dow_local >= 5).astype(np.int32)

    return {
        "hour_utc": hour_utc,
        "hour_local": hour_local,
        "dow_local": dow_local,
        "is_weekend": is_weekend,
    }


def build_feature_matrix(
    raw: pd.DataFrame,
    origins: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    oblasts: Sequence[str],
    cap_minutes: float = 20160.0,
) -> pd.DataFrame:
    """Build full zero-leakage feature matrix for given origins and oblasts.
    
    All features are derived strictly from information knowable as-of each origin t.
    Returns DataFrame indexed by MultiIndex (origin, oblast) with columns FEATURE_NAMES.
    """
    if len(origins) == 0 or len(oblasts) == 0:
        index = pd.MultiIndex.from_tuples([], names=["origin", "oblast"])
        return pd.DataFrame(columns=FEATURE_NAMES, index=index)

    origins_idx = pd.DatetimeIndex(origins)
    if origins_idx.tz is not None:
        origins_idx = origins_idx.tz_localize(None)

    origins_arr = origins_idx.to_numpy(dtype="datetime64[ns]")
    N_origins = len(origins_arr)
    sorted_oblasts = sorted(list(oblasts))
    oblast_id_map = {ob: idx for idx, ob in enumerate(sorted_oblasts)}

    # Global calendar features (same for all oblasts at each origin)
    cal = _compute_calendar_features(origins_idx)

    # Windows in nanoseconds
    win_deltas = {
        "1h": np.timedelta64(1, "h"),
        "3h": np.timedelta64(3, "h"),
        "6h": np.timedelta64(6, "h"),
        "12h": np.timedelta64(12, "h"),
        "24h": np.timedelta64(24, "h"),
        "72h": np.timedelta64(72, "h"),
        "7d": np.timedelta64(7, "D"),
        "30d": np.timedelta64(30, "D"),
    }

    # Per-oblast intermediate feature arrays
    per_oblast_data = {}

    for ob in sorted_oblasts:
        df_o = raw[raw["oblast"] == ob]
        starts = np.sort(df_o["started_at"].to_numpy(dtype="datetime64[ns]"))

        # Ends filtering for known ends <= t
        ends_raw = df_o["finished_at"].to_numpy(dtype="datetime64[ns]")
        starts_raw = df_o["started_at"].to_numpy(dtype="datetime64[ns]")
        valid_end_mask = ~np.isnat(ends_raw)
        
        valid_ends = ends_raw[valid_end_mask]
        valid_end_starts = starts_raw[valid_end_mask]
        
        # Sort by finished_at for recency search
        sort_by_end_idx = np.argsort(valid_ends)
        known_ends = valid_ends[sort_by_end_idx]
        known_starts_for_ends = valid_end_starts[sort_by_end_idx]
        known_durations = (known_ends - known_starts_for_ends) / np.timedelta64(1, "m")

        # 1. Starts in windows
        r_idx = np.searchsorted(starts, origins_arr, side="right")
        starts_counts = {}
        for w_name, w_dt in win_deltas.items():
            l_idx = np.searchsorted(starts, origins_arr - w_dt, side="right")
            starts_counts[w_name] = (r_idx - l_idx).astype(np.float32)

        start_rate_30d = starts_counts["30d"] / 30.0

        # 2. Recency: mins_since_last_start
        mins_since_start = np.full(N_origins, cap_minutes, dtype=np.float32)
        has_prev_start = r_idx > 0
        if np.any(has_prev_start):
            prev_starts = starts[r_idx[has_prev_start] - 1]
            diff_m = (origins_arr[has_prev_start] - prev_starts) / np.timedelta64(1, "m")
            mins_since_start[has_prev_start] = np.clip(diff_m, 0.0, cap_minutes)

        # 3. Recency: mins_since_last_end & last_alert_duration_min
        mins_since_end = np.full(N_origins, cap_minutes, dtype=np.float32)
        last_duration = np.full(N_origins, np.nan, dtype=np.float32)
        r_end_idx = np.searchsorted(known_ends, origins_arr, side="right")
        has_prev_end = r_end_idx > 0
        if np.any(has_prev_end):
            prev_ends = known_ends[r_end_idx[has_prev_end] - 1]
            diff_end_m = (origins_arr[has_prev_end] - prev_ends) / np.timedelta64(1, "m")
            mins_since_end[has_prev_end] = np.clip(diff_end_m, 0.0, cap_minutes)
            last_duration[has_prev_end] = known_durations[r_end_idx[has_prev_end] - 1]

        # 4. State: is_active_now, current_alert_age_min, n_open_alerts_now
        is_active = np.zeros(N_origins, dtype=np.int32)
        n_open = np.zeros(N_origins, dtype=np.int32)
        alert_age = np.zeros(N_origins, dtype=np.float32)

        # An alert is active at origin t iff started_at <= t and (finished_at > t or NaT)
        for s, e in zip(starts_raw, ends_raw):
            t_s = np.searchsorted(origins_arr, s, side="left")
            if np.isnat(e):
                t_e = N_origins
            else:
                t_e = np.searchsorted(origins_arr, e, side="left")
            
            if t_e > t_s:
                n_open[t_s:t_e] += 1
                cur_active = is_active[t_s:t_e]
                is_first = cur_active == 0
                is_active[t_s:t_e] = 1
                # calculate age from start
                age_delta = (origins_arr[t_s:t_e] - s) / np.timedelta64(1, "m")
                # update age for earliest start
                cur_age = alert_age[t_s:t_e]
                cur_age[is_first] = age_delta[is_first]
                alert_age[t_s:t_e] = np.maximum(cur_age, age_delta)

        # 5. Alert minutes in windows (6h, 24h, 7d)
        alert_mins = {
            "6h": np.zeros(N_origins, dtype=np.float32),
            "24h": np.zeros(N_origins, dtype=np.float32),
            "7d": np.zeros(N_origins, dtype=np.float32),
        }

        # Check if origins are regular hourly grid
        is_regular_hourly = False
        if N_origins > 250:
            diffs = np.diff(origins_arr)
            is_regular_hourly = np.all(diffs == np.timedelta64(1, "h"))

        if is_regular_hourly:
            # Vectorized hourly binning via cumulative sums
            grid_min = origins_idx.min() - pd.Timedelta(days=7)
            full_grid = pd.date_range(grid_min, origins_idx.max(), freq="h").to_numpy(dtype="datetime64[ns]")
            G = len(full_grid)
            hourly_mins = np.zeros(G, dtype=np.float64)

            # Accumulate overlap into hourly bins
            for s, e in zip(starts_raw, ends_raw):
                eff_e = full_grid[-1] + np.timedelta64(1, "h") if np.isnat(e) else e
                si = max(0, np.searchsorted(full_grid, s, side="right") - 1)
                ei = min(G, np.searchsorted(full_grid, eff_e, side="right"))
                for b in range(si, ei):
                    b_s = full_grid[b]
                    b_e = b_s + np.timedelta64(1, "h")
                    ov = min(eff_e, b_e) - max(s, b_s)
                    if ov > np.timedelta64(0, "ns"):
                        hourly_mins[b] += ov / np.timedelta64(1, "m")

            cumsum = np.concatenate([[0.0], np.cumsum(hourly_mins)])
            grid_indices = np.searchsorted(full_grid, origins_arr)

            alert_mins["6h"] = (cumsum[grid_indices] - cumsum[grid_indices - 6]).astype(np.float32)
            alert_mins["24h"] = (cumsum[grid_indices] - cumsum[grid_indices - 24]).astype(np.float32)
            alert_mins["7d"] = (cumsum[grid_indices] - cumsum[grid_indices - 168]).astype(np.float32)
        else:
            # Direct exact calculation for each origin
            for i, t in enumerate(origins_arr):
                for w_key in ["6h", "24h", "7d"]:
                    w_dt = win_deltas[w_key]
                    win_start = t - w_dt
                    in_win_mask = (starts_raw <= t) & (np.isnat(ends_raw) | (ends_raw > win_start))
                    if np.any(in_win_mask):
                        s_sub = starts_raw[in_win_mask]
                        e_sub = ends_raw[in_win_mask]
                        # Clip ends at t for ongoing/future ends
                        clipped_ends = np.where(np.isnat(e_sub) | (e_sub > t), t, e_sub)
                        clipped_starts = np.maximum(s_sub, win_start)
                        ov = np.maximum(0, (clipped_ends - clipped_starts) / np.timedelta64(1, "m"))
                        alert_mins[w_key][i] = np.sum(ov)

        per_oblast_data[ob] = {
            "starts": starts_counts,
            "start_rate_30d": start_rate_30d,
            "mins_since_start": mins_since_start,
            "mins_since_end": mins_since_end,
            "last_duration": last_duration,
            "is_active": is_active,
            "n_open": n_open,
            "alert_age": alert_age,
            "alert_min_6h": alert_mins["6h"],
            "alert_min_24h": alert_mins["24h"],
            "alert_min_7d": alert_mins["7d"],
        }

    # 6. National aggregations across all modeled oblasts
    nat_active_cnt = np.sum([per_oblast_data[ob]["is_active"] for ob in sorted_oblasts], axis=0)
    nat_starts_1h = np.sum([per_oblast_data[ob]["starts"]["1h"] for ob in sorted_oblasts], axis=0)
    nat_starts_3h = np.sum([per_oblast_data[ob]["starts"]["3h"] for ob in sorted_oblasts], axis=0)
    nat_start_rate_30d = np.sum([per_oblast_data[ob]["start_rate_30d"] for ob in sorted_oblasts], axis=0)

    # 7. Assemble final dataframe
    records = []
    for ob in sorted_oblasts:
        ob_d = per_oblast_data[ob]
        neighbors = [n for n in ADJACENCY.get(ob, []) if n in per_oblast_data]
        n_nbrs = len(neighbors)

        if n_nbrs > 0:
            nbr_active = np.sum([per_oblast_data[n]["is_active"] for n in neighbors], axis=0)
            nbr_active_share = nbr_active / float(n_nbrs)
            nbr_starts_1h = np.sum([per_oblast_data[n]["starts"]["1h"] for n in neighbors], axis=0)
            nbr_starts_3h = np.sum([per_oblast_data[n]["starts"]["3h"] for n in neighbors], axis=0)
            nbr_starts_6h = np.sum([per_oblast_data[n]["starts"]["6h"] for n in neighbors], axis=0)
            nbr_min_since_start = np.min([per_oblast_data[n]["mins_since_start"] for n in neighbors], axis=0)
        else:
            nbr_active = np.zeros(N_origins, dtype=np.float32)
            nbr_active_share = np.zeros(N_origins, dtype=np.float32)
            nbr_starts_1h = np.zeros(N_origins, dtype=np.float32)
            nbr_starts_3h = np.zeros(N_origins, dtype=np.float32)
            nbr_starts_6h = np.zeros(N_origins, dtype=np.float32)
            nbr_min_since_start = np.full(N_origins, cap_minutes, dtype=np.float32)

        df_block = pd.DataFrame(
            {
                "origin": origins_idx,
                "oblast": ob,
                "oblast_id": oblast_id_map[ob],
                "hour_local": cal["hour_local"],
                "dow_local": cal["dow_local"],
                "is_weekend": cal["is_weekend"],
                "hour_utc": cal["hour_utc"],
                "is_active_now": ob_d["is_active"],
                "current_alert_age_min": ob_d["alert_age"],
                "n_open_alerts_now": ob_d["n_open"],
                "mins_since_last_start": ob_d["mins_since_start"],
                "mins_since_last_end": ob_d["mins_since_end"],
                "last_alert_duration_min": ob_d["last_duration"],
                "starts_1h": ob_d["starts"]["1h"],
                "starts_3h": ob_d["starts"]["3h"],
                "starts_6h": ob_d["starts"]["6h"],
                "starts_12h": ob_d["starts"]["12h"],
                "starts_24h": ob_d["starts"]["24h"],
                "starts_72h": ob_d["starts"]["72h"],
                "starts_7d": ob_d["starts"]["7d"],
                "alert_min_6h": ob_d["alert_min_6h"],
                "alert_min_24h": ob_d["alert_min_24h"],
                "alert_min_7d": ob_d["alert_min_7d"],
                "start_rate_30d": ob_d["start_rate_30d"],
                "nbr_active_cnt": nbr_active,
                "nbr_active_share": nbr_active_share,
                "nbr_starts_1h": nbr_starts_1h,
                "nbr_starts_3h": nbr_starts_3h,
                "nbr_starts_6h": nbr_starts_6h,
                "nbr_min_since_last_start": nbr_min_since_start,
                "nat_active_cnt": nat_active_cnt,
                "nat_starts_1h": nat_starts_1h,
                "nat_starts_3h": nat_starts_3h,
                "nat_start_rate_30d": nat_start_rate_30d,
            }
        )
        records.append(df_block)

    full_df = pd.concat(records, ignore_index=True)
    return full_df.set_index(["origin", "oblast"]).sort_index()[FEATURE_NAMES]


def compute_labels(
    raw: pd.DataFrame,
    origins: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    oblasts: Sequence[str],
    horizon_hours: int = 3,
) -> pd.DataFrame:
    """Compute target y(o, t) according to PRD §2.3:
    
    y(o, t) = 1 if there exists an alert start s in oblast o with t < s <= t + H, else 0.
    
    Note:
    - This is the ONLY forward-looking calculation, isolated from feature code.
    - Origins with t + H > data_end are incomplete and dropped.
    Returns DataFrame indexed by MultiIndex (origin, oblast) with column ['y'].
    """
    if len(origins) == 0:
        index = pd.MultiIndex.from_tuples([], names=["origin", "oblast"])
        return pd.DataFrame(columns=["y"], index=index)

    origins_idx = pd.DatetimeIndex(origins)
    if origins_idx.tz is not None:
        origins_idx = origins_idx.tz_localize(None)

    data_end = raw["started_at"].max()
    horizon_delta = pd.Timedelta(hours=horizon_hours)

    # Drop origins with incomplete horizon window
    valid_origins_mask = origins_idx + horizon_delta <= data_end
    valid_origins = origins_idx[valid_origins_mask]

    if len(valid_origins) == 0:
        index = pd.MultiIndex.from_tuples([], names=["origin", "oblast"])
        return pd.DataFrame(columns=["y"], index=index)

    origins_arr = valid_origins.to_numpy(dtype="datetime64[ns]")
    target_end_arr = (valid_origins + horizon_delta).to_numpy(dtype="datetime64[ns]")

    records = []
    for ob in sorted(list(oblasts)):
        df_o = raw[raw["oblast"] == ob]
        starts = np.sort(df_o["started_at"].to_numpy(dtype="datetime64[ns]"))

        # Number of starts with s <= t + H
        idx_right = np.searchsorted(starts, target_end_arr, side="right")
        # Number of starts with s <= t
        idx_left = np.searchsorted(starts, origins_arr, side="right")

        # Starts with t < s <= t + H is (idx_right - idx_left)
        y = (idx_right > idx_left).astype(np.int32)

        df_ob = pd.DataFrame({
            "origin": valid_origins,
            "oblast": ob,
            "y": y,
        })
        records.append(df_ob)

    res = pd.concat(records, ignore_index=True)
    return res.set_index(["origin", "oblast"]).sort_index()
