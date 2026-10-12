"""Modern editorial visualizations for AlertCast using Matplotlib and Seaborn."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Color palette definition for editorial clarity
PALETTE = {
    "primary": "#1f77b4",
    "accent": "#d62728",
    "dark": "#1a1d20",
    "gray": "#555555",
    "light_gray": "#f8f9fa",
    "grid": "#e9ecef",
    # Feature category colors
    "Recency": "#2ca02c",
    "State": "#1f77b4",
    "History": "#ff7f0e",
    "Neighbors": "#9467bd",
    "National": "#e377c2",
    "Calendar": "#7f7f7f",
    "Identity": "#8c564b",
}


def _get_feature_category(feat: str) -> str:
    """Classify feature into high-level group for color-coding."""
    if feat.startswith("mins_since") or feat.startswith("last_alert"):
        return "Recency"
    elif feat.startswith("is_active") or feat.startswith("current_alert") or feat.startswith("n_open"):
        return "State"
    elif feat.startswith("starts_") or feat.startswith("alert_min") or feat.startswith("start_rate"):
        return "History"
    elif feat.startswith("nbr_"):
        return "Neighbors"
    elif feat.startswith("nat_"):
        return "National"
    elif feat in ["hour_local", "dow_local", "is_weekend", "hour_utc"]:
        return "Calendar"
    else:
        return "Identity"


def plot_time_of_day_heatmap(
    raw: pd.DataFrame,
    test_start: pd.Timestamp,
    output_path: str = "figures/fig1_heatmap_hour_dow.png",
) -> None:
    """Figure 1: Time-of-Day x Day-of-Week heatmap of alert starts (Pre-test data only)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Use training data only (pre-test period)
    train_data = raw[raw["started_at"] < test_start].copy()

    # Convert started_at to Europe/Kyiv
    kyiv_times = train_data["started_at"].dt.tz_localize("UTC").dt.tz_convert("Europe/Kyiv")
    hours = kyiv_times.dt.hour
    dows = kyiv_times.dt.dayofweek

    counts = pd.crosstab(dows, hours)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts.index = [day_labels[i] for i in counts.index]

    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    sns.heatmap(
        counts,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        cbar_kws={"label": "Total Alert Starts"},
        linewidths=0.5,
        linecolor="#ffffff",
        ax=ax,
    )

    ax.set_title("Air Raid Alert Starts by Time of Day and Day of Week", fontsize=14, fontweight="bold", loc="left", pad=24)
    ax.text(0.0, 1.025, "Nationwide aggregated alert starts (Europe/Kyiv time) • Pre-test training period only",
            transform=ax.transAxes, fontsize=10, color=PALETTE["gray"])

    ax.set_xlabel("Hour of Day (Local Time, 00:00 – 23:00)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Day of Week", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    output_path: str = "figures/fig2_feature_importance.png",
) -> None:
    """Figure 2: LightGBM Feature Importance (Top N by Gain)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    raw_gains = model.booster_.feature_importance(importance_type="gain")
    total_gain = raw_gains.sum()
    pct_gains = (raw_gains / total_gain) * 100.0

    df_imp = pd.DataFrame({"feature": feature_names, "importance": pct_gains})
    df_imp["category"] = df_imp["feature"].apply(_get_feature_category)
    df_imp = df_imp.sort_values(by="importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = [PALETTE.get(cat, "#333333") for cat in df_imp["category"]]
    bars = ax.barh(df_imp["feature"], df_imp["importance"], color=colors, height=0.65, edgecolor="none")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=9.5,
            color=PALETTE["dark"],
            fontweight="bold",
        )

    ax.set_xlim(0, max(df_imp["importance"]) * 1.15)
    ax.set_title("LightGBM Feature Importance (Top 15 by Gain)", fontsize=14, fontweight="bold", loc="left", pad=24)
    ax.text(0.0, 1.025, "Normalized % of total split gain • Key predictive drivers for 3-hour alert starts",
            transform=ax.transAxes, fontsize=10, color=PALETTE["gray"])

    ax.set_xlabel("Importance (% of Total Gain)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Feature", fontsize=11, fontweight="bold")

    cats_present = df_imp["category"].unique()
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[cat], label=cat) for cat in cats_present if cat in PALETTE
    ]
    ax.legend(handles=legend_elements, title="Group", loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_forecast_vs_actual(
    raw: pd.DataFrame,
    test_origins: pd.DatetimeIndex,
    preds_test: np.ndarray,
    b2_preds_test: np.ndarray,
    oblast: str,
    threshold: float = 0.5,
    days: int = 7,
    output_path: str = "figures/fig3_forecast_vs_actual.png",
) -> None:
    """Figure 3: 7-Day Forecast vs Actual probability plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    end_time = test_origins.max()
    start_time = end_time - pd.Timedelta(days=days)

    window_mask = (test_origins >= start_time) & (test_origins <= end_time)
    sub_origins = test_origins[window_mask]

    sub_df = pd.DataFrame({
        "origin": sub_origins,
        "pred": preds_test[window_mask],
        "b2": b2_preds_test[window_mask],
    }).sort_values("origin")

    df_ob = raw[raw["oblast"] == oblast]
    alerts_in_win = df_ob[
        (df_ob["started_at"] <= end_time + pd.Timedelta(hours=3)) &
        (df_ob["finished_at"] >= start_time)
    ]

    fig, ax = plt.subplots(figsize=(14, 5.8), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Shaded actual alert bands
    first_alert = True
    for _, alert in alerts_in_win.iterrows():
        s = max(alert["started_at"], start_time)
        e = min(alert["finished_at"], end_time)
        ax.axvspan(
            s,
            e,
            color="#ff4d4d",
            alpha=0.25,
            label="Actual Active Alert" if first_alert else "",
        )
        first_alert = False

    # 2. Predicted probability trace
    ax.plot(
        sub_df["origin"],
        sub_df["pred"],
        color="#1f77b4",
        linewidth=2.2,
        label="LightGBM P(Start in next 3h)",
    )

    # 3. B2 baseline trace
    ax.plot(
        sub_df["origin"],
        sub_df["b2"],
        color="#777777",
        linewidth=1.3,
        linestyle="--",
        label="B2 Hourly Climatology",
    )

    # 4. Actual alert start markers
    starts_in_win = alerts_in_win[
        (alerts_in_win["started_at"] >= start_time) & (alerts_in_win["started_at"] <= end_time)
    ]["started_at"]
    if len(starts_in_win) > 0:
        ax.scatter(
            starts_in_win,
            [0.95] * len(starts_in_win),
            color="#d62728",
            marker="v",
            s=80,
            zorder=5,
            label="Actual Alert Start",
        )

    # 5. Operating threshold line
    ax.axhline(
        threshold,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label=f"Decision Threshold ({threshold:.2f})",
    )

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(start_time, end_time)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:00"))

    ax.set_title(f"3-Hour Alert Start Forecast vs. Actual Events — {oblast}", fontsize=14, fontweight="bold", loc="left", pad=24)
    ax.text(0.0, 1.025, f"Most recent {days} days of test set • Forecast issued at origin t covers (t, t + 3h]",
            transform=ax.transAxes, fontsize=10, color=PALETTE["gray"])

    ax.set_xlabel("Forecast Origin Time (UTC)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Forecast Probability", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
