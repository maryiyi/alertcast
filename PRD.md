# PRD: AlertCast — Zero-Leakage 3-Hour Air Raid Alert Start Forecasting (Ukraine)

| | |
|---|---|
| **Status** | Draft v1.0, ready for implementation |
| **Audience** | AI coding agent (primary), human owner (reviewer) |
| **Language / stack** | Python 3.11+, pandas, NumPy, scikit-learn, LightGBM, Plotly, pytest |
| **Time budget** | Single evening. Strict priority tiers in §13. |
| **Working name** | `alertcast` (rename freely) |

> **How to use this document.** This is the master specification. Treat every statement containing **MUST / MUST NOT** as an acceptance criterion. If something is ambiguous, choose the interpretation that is *safer against data leakage*, write the decision to `docs/DECISIONS.md`, and continue. Do not ask for clarification unless blocked.

---

## 0. Non-negotiables (read first)

1. **Zero data leakage.** Nothing used as a model input, training row, feature, label, scaler, encoder, threshold, hyperparameter choice, or oblast-selection decision may depend on information from after the forecast origin time `t` at which a prediction is made. The only permitted use of post-`t` data is to compute the *evaluation label* for scoring.
2. **The test suite is the contract.** The leakage tests in §6 MUST exist, MUST pass, and MUST NOT be weakened, skipped, or edited to make the pipeline pass. If a leakage test fails, fix the pipeline, never the test.
3. **Reproducibility.** Fixed seeds, pinned dependencies, pinned data snapshot (hash recorded). One command reproduces everything: `make all`.
4. **Honesty over flattery.** If LightGBM does not beat the baselines, the report says so. No cherry-picked windows, oblasts or metrics.
5. **Scope discipline.** Text/Telegram/news data is **out of scope**. Do not build it. It appears only as "Future Phase 2" in the README.

---

## 1. Problem statement

Civilians and defense-adjacent services need advance indication of where air raid alerts are likely to be issued. We build an end-to-end pipeline that, at the top of every hour, outputs for each Ukrainian oblast the probability that **a new air raid alert will start in that oblast within the next 3 hours**, using only information available at that moment, and evaluate it on the most recent 3 months of data.

This is a pet-project MVP. It is **not** an operational warning system (see §14 disclaimer).

### 1.1 Goals
- G1: Working, reproducible Python pipeline: ingest → clean → features → train → evaluate → report.
- G2: Provable leakage safety via automated tests (the "Zero Data Leakage Architecture").
- G3: Honest evaluation against strong baselines on a strictly unseen, most-recent 3-month window.
- G4: Three presentation-ready static visualizations (§10), no HTML.

### 1.2 Non-goals
- Text/media/Telegram signals, graph neural networks, deep learning, a web service, real-time inference, sub-oblast (raion/hromada) forecasting.

---

## 2. Formal problem definition

### 2.1 Entities and time conventions
- **Oblast set `O`**: modeled administrative units (defined in §3.4).
- **Time zone**: all raw timestamps are **UTC** (per dataset documentation). Internally use **tz-naive UTC `datetime64[ns]`**. Local-time features use `Europe/Kyiv` conversion (handles DST).
- **Grid**: hourly forecast origins `t` at `HH:00:00 UTC`.
- **Horizon** `H = 3 hours` (config `horizon_hours`, default 3).

### 2.2 Information set (the as-of rule)
At origin `t`, the information set `I(t)` consists of:
- every alert **start event** with `started_at <= t`;
- every alert **end event** with `finished_at <= t`.

An alert interval `[s, e)` is **active at `t`** iff `s <= t` and (`e` is unknown/`NaT` **or** `e > t`).
An end time `e > t` is **future information and MUST be treated as unknown (`NaT`) at time `t`.**

### 2.3 Label
For oblast `o` and origin `t`:

```
y(o, t) = 1  if there exists a raw alert start s in oblast o with  t < s <= t + H
          0  otherwise
```

- The label is computed **only** in `labels.py`, from the *full* data, and is used **only** as a training target (for origins whose label window is fully in the training period, see §6.5) and as an evaluation target.
- Origins with `t + H > data_end` (incomplete label window) MUST be dropped.
- No merging or "de-duplication" of overlapping alerts is allowed before labeling (it would use future end times; see §3.3).

### 2.2b Row eligibility flags
Each `(o, t)` row carries:
- `is_active_now` (feature; from the as-of state).
- `calm = ~is_active_now`.

**Headline evaluation subset = `calm` rows** ("no alert currently active in `o`; will one start within 3h?"), as this is the actionable warning question. Metrics are **also** reported on all rows as secondary. The model is trained on all rows with `is_active_now` as a feature.

---

## 3. Data

### 3.1 Source
Repository: `Vadimkin/ukrainian-air-raid-sirens-dataset` (MIT license, flat CSV, updated daily, all times in UTC).
We STRICTLY use `datasets/volunteer_data_en.csv`. Rows with no end message are flagged `naive=True` with `finished_at = started_at + 30 min`.

### 3.2 Cleaning rules (row-local only)
Cleaning MUST be **row-local and deterministic**: it may use only the fields of the row itself (or exact-duplicate detection). It MUST NOT use other rows' future information.

- Parse `started_at`, `finished_at` as UTC → tz-naive.
- Drop exact duplicate rows `(oblast, started_at, finished_at)`.
- Drop rows with `finished_at < started_at` (log the count).
- Rows with missing `finished_at`: impute `started_at + 30 min` (same rule the dataset uses for `naive=True`). Log the count.
- Normalize oblast names through an explicit mapping dict; **fail loudly** on unknown names.
- **MUST NOT** merge overlapping intervals, close gaps, or resample alerts on the full dataset. (Merging uses future end times; it would make features at `t` differ between full and truncated data and fail the truncation test.) Overlaps are handled at feature time through the as-of event log.
- Output: `data/interim/alerts.parquet` with columns `oblast: category, started_at: datetime64[ns], finished_at: datetime64[ns], naive: bool`.

### 3.3 Oblast set
- Exclude units with a **permanent siren** (per dataset README: Luhansk Oblast, Crimea) as the target is degenerate.
- Exclude any oblast with fewer than `min_train_starts` (config, default 50) alert starts **in the training period only** (data before `test_start − H`). The selection is computed from training-period data only and frozen.
- The final oblast list is written to `artifacts/oblasts.json` and reused unchanged everywhere, including the test period.
- Kyiv City and Kyiv Oblast are separate units if the dataset separates them.

### 3.4 Data vintage and provenance
- `scripts/fetch_data.py` clones or downloads the CSV and writes `data/MANIFEST.json`: source URL, git commit SHA, download timestamp, SHA-256 of the file, row count.
- **Known, unfixable limitation (document in README):** the dataset is a retrospectively corrected snapshot; real-time data would have had gaps/latency. Naive 30-minute end imputation is also a deterministic row-local approximation of the real signal.

---

## 4. Time splits

Let `data_end` = `max(started_at)` in the snapshot, floored to the hour.

| Period | Definition |
|---|---|
| **Test (unseen)** | Origins `t ∈ [data_end − 90 days, data_end − H]` (last 3 months, hourly) |
| **Pre-test** | all origins `t < test_start − H` (training + hyperparameter tuning via walk-forward) |
| **Warm-up** | the first 30 days of data supply feature history only and are not used as origins |

where `test_start = data_end − 90 days`.

**Hard rule:** the test period MUST NOT be touched by any of: hyperparameter search, feature selection, oblast selection, threshold selection, calibration, scaler/encoder fitting, or manual inspection-driven code changes. The test set is evaluated **once**, at the end, via `scripts/run_test_eval.py`, and the result is logged with a timestamp. (If you iterate after seeing test results, say so in the README.)

---

## 5. Zero-leakage architecture

### 5.1 Layered design

```
raw alerts (full) ──► clean (row-local) ──► alerts.parquet
                                              │
              ┌───────────────────────────────┼────────────────────────────────┐
              ▼                               ▼                                ▼
     labels.py (uses FUTURE,           as_of.py: as_of_view(raw, t)      splits.py (purged
     only for y; isolated)             features/*.py (see ONLY view)     walk-forward folds)
              │                               │                                │
              └───────────────► training matrix (X from past, y from future) ◄─┘
```

Rules:
- `features/` MUST NOT import from `labels.py`, `splits.py`, `models/`, or `evaluation/`. Enforced by a test (§6.6).
- `labels.py` is the only module allowed to look forward.
- No module may read wall-clock time (`datetime.now`, `pd.Timestamp.now`, `date.today`) or use dataset-wide maxima (`raw.started_at.max()`) as a reference time inside feature code. The **only** time reference in feature code is the argument `t`.

### 5.2 `as_of_view` (the single gate)

```python
# src/alertcast/as_of.py
def as_of_view(raw: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Return only what was knowable at time t.
    - keep rows with started_at <= t
    - mask finished_at > t as NaT (alert still open as far as we know)
    Returns columns: oblast, started_at, finished_at (NaT = open), naive.
    """
```

### 5.3 `build_features` (Single Production Implementation)
Implement a single, robust feature builder mapping exact operational bounds:

```python
# src/alertcast/features/builder.py
def build_feature_matrix(raw: pd.DataFrame, origins: pd.DatetimeIndex, oblasts: Sequence[str], cfg: FeatureConfig) -> pd.DataFrame:
    """Index: MultiIndex (origin, oblast). Columns: FEATURE_NAMES."""
```

### 5.4 `build_feature_matrix` (production, vectorized)

The reference function is too slow for ~1M rows. Implement:

```python
# src/alertcast/features/vectorized.py
def build_feature_matrix(raw: pd.DataFrame, origins: pd.DatetimeIndex,
                         oblasts: Sequence[str], cfg: FeatureConfig) -> pd.DataFrame:
    """Index: MultiIndex (origin, oblast). Columns: FEATURE_NAMES."""
```

Implementation hints (non-binding): per-oblast sorted start arrays + `np.searchsorted` for `mins_since_last_start`/counts; running max of end over start-sorted intervals for `is_active_now` (active iff `max_end_known_at_t > t` or any open interval); cumulative-sum of hourly start counts for window counts. Whatever you do, **end times > t must be treated as unknown**.

**MUST** satisfy the equivalence test (§6.2) against the reference implementation, and the truncation tests (§6.1, §6.3) on **both** implementations.

### 5.5 Splitting and label purging

A training row at origin `t_train` has a label that looks until `t_train + H`. For a validation/test block starting at `cutoff`:

```
train origins:      t_train + H <= cutoff     (label window fully realized by cutoff)
validation origins: t_valid >= cutoff
```

Implementation `splits.py`:

```python
def walk_forward_folds(origins, test_start, H, n_folds=5, valid_days=28, train_window_days=None):
    """Yield (train_idx, valid_idx) with expanding (or trailing) train windows.
    valid_end_k = test_start - H - k*valid_days  (k = 0..n_folds-1)
    valid_start_k = valid_end_k - valid_days
    train: origins with t + H <= valid_start_k  (and t >= valid_start_k - train_window_days if set)
    valid: origins in (valid_start_k, valid_end_k]
    Invariant: max(valid_end_k) <= test_start - H.
    """
```

Final model: train origins `t + H <= test_start`. Test origins `t >= test_start`.

Training-window length (`train_window_days ∈ {180, 365, 730, None}`) is a **tuned hyperparameter** (concept drift: tactics change).

**Primary protocol: frozen model.** Train once pre-test, predict the whole test window with no retraining. *(P2 stretch: monthly walk-forward retraining inside the test window, with each retrain using only data ≤ its cutoff − H.)*

---

## 6. Leakage test suite (`tests/leakage/`): THE CONTRACT

All tests run under `pytest -m leakage`. A fast subset (`n=30`) runs as a **runtime gate** at the start of `scripts/run_pipeline.py`: the pipeline MUST refuse to train if the gate fails.

### 6.0 Shared test helpers (written independently inside tests, NOT imported from `src/`)

```python
def independent_truncate(raw, t):
    out = raw[raw["started_at"] <= t].copy()
    out.loc[out["finished_at"] > t, "finished_at"] = pd.NaT
    return out

def poison_future(raw, t, oblasts, rng, n_fake=500):
    base = independent_truncate(raw, t)
    # (a) still-open alerts get random FUTURE end times
    open_mask = base["finished_at"].isna()
    base.loc[open_mask, "finished_at"] = t + pd.to_timedelta(
        rng.uniform(1, 60 * 24 * 30, open_mask.sum()), unit="m")
    # (b) append fabricated alerts that start strictly after t
    start = t + pd.to_timedelta(rng.uniform(1, 60 * 24 * 30, n_fake), unit="m")
    fake = pd.DataFrame({
        "oblast": rng.choice(oblasts, n_fake),
        "started_at": start,
        "finished_at": start + pd.to_timedelta(rng.uniform(5, 600, n_fake), unit="m"),
        "naive": False})
    return pd.concat([base, fake], ignore_index=True)

def sample_origins(origins, n, seed):
    """Stratified: 1/3 from test period, 1/3 from pre-test, 1/3 'edge' origins:
       - t equal to some alert's started_at floored/ceilinged to the hour
       - t equal to some alert's finished_at (hour-aligned)
       - an origin 1 hour after a start with a long open alert
       - the first valid origin (after warm-up)
       - origins around DST transitions (Kyiv last Sunday of March/October)"""
```

### 6.1 Automated Truncation Invariance (Production Implementation)

```python
@pytest.mark.leakage
def test_truncation_invariance(raw, oblasts, cfg):
    rng = np.random.default_rng(1)
    M_full = build_feature_matrix(raw, all_origins, oblasts, cfg)   # computed ONCE on the full data
    for t in sample_origins(all_origins, n=200, seed=1):
        for variant in (independent_truncate(raw, t), poison_future(raw, t, oblasts, rng)):
            M_t = build_feature_matrix(variant, pd.DatetimeIndex([t]), oblasts, cfg)
            assert_frames_close(M_full.xs(t, level="origin").loc[oblasts],
                                M_t.xs(t, level="origin").loc[oblasts], atol=1e-9)
```

### 6.2 Truncation invariance (production implementation) — the strongest test

```python
@pytest.mark.leakage
def test_truncation_invariance_production(raw, oblasts, cfg):
    rng = np.random.default_rng(1)
    M_full = build_feature_matrix(raw, all_origins, oblasts, cfg)   # computed ONCE on the full data
    for t in sample_origins(all_origins, n=200, seed=1):
        for variant in (independent_truncate(raw, t), poison_future(raw, t, oblasts, rng)):
            M_t = build_feature_matrix(variant, pd.DatetimeIndex([t]), oblasts, cfg)
            assert_frames_close(M_full.xs(t, level="origin").loc[oblasts],
                                M_t.xs(t, level="origin").loc[oblasts], atol=1e-9)
```

If any feature at `t` changes when the post-`t` data is removed or replaced with garbage, the feature leaks and the test fails.

### 6.3 Model prediction invariance

```python
@pytest.mark.leakage
def test_prediction_invariance(trained_model, raw, oblasts, cfg):
    for t in sample_origins(test_origins, n=50, seed=3):
        p_full  = trained_model.predict_proba(M_full.xs(t, level="origin").loc[oblasts])
        X_trunc = build_feature_matrix(independent_truncate(raw, t), pd.DatetimeIndex([t]), oblasts, cfg)
        p_trunc = trained_model.predict_proba(X_trunc.xs(t, level="origin").loc[oblasts])
        np.testing.assert_allclose(p_full, p_trunc, atol=1e-12)
```

### 6.4 Splitter and label-purge tests

```python
@pytest.mark.leakage
def test_walk_forward_no_overlap():
    for tr, va in walk_forward_folds(origins, test_start, H, ...):
        assert origins[tr].max() + H <= origins[va].min()           # labels of train fully realized
        assert origins[va].max() <= test_start - H                  # tuning never touches test
def test_final_train_purged():
    assert final_train_origins.max() + H <= test_start
def test_test_window_exact():
    assert test_origins.min() >= test_start and test_origins.max() <= data_end - H
```

### 6.5 Structural / static tests
- **Import isolation:** AST-parse every module under `features/`; assert it never imports `labels`, `splits`, `models`, `evaluation`.
- **Forbidden tokens in `features/`:** `.now(`, `.today(`, `datetime.now`, `.max()` applied to `started_at`/`finished_at`/index of the raw frame, `random`, `np.random` (regex/AST check; allowlist with justification comment `# leak-ok: <reason>` reviewed by the human owner).
- **Fit-scope test:** any object with a `fit` method (encoders, thresholds, climatology baselines, models) is fitted only on rows satisfying `t + H <= cutoff`. Implement with a wrapper/assertion in `training.py` that receives the cutoff and asserts on the data it is fed.

### 6.6 Too-good-to-be-true alarm (soft)
After evaluation, if test PR-AUC on the calm subset exceeds `0.95` or any single feature's univariate PR-AUC exceeds `0.9`, print a loud warning and add a section "Suspiciously good results: leak audit" to the report.

---

## 7. Feature specification

All features are computed from `as_of_view(raw, t)` only, per `(oblast o, origin t)`. "Window `w`" = interval `(t − w, t]`. Unknown end = open.

| Group | Feature | Definition |
|---|---|---|
| **Identity** | `oblast_id` | Static integer code of the oblast. Used as LightGBM categorical. Static mapping, not data-fitted. |
| **Calendar** | `hour_local`, `dow_local`, `is_weekend`, `hour_utc` | From `t` converted to `Europe/Kyiv` (and UTC). Deterministic functions of `t`, not leakage. |
| **State** | `is_active_now` | 1 if any interval in `o` active at `t` (§2.2). |
| | `current_alert_age_min` | If active: `t −` start of the earliest open interval; else 0. |
| | `n_open_alerts_now` | Count of simultaneously open intervals. |
| **Recency** | `mins_since_last_start` | `t −` latest `started_at <= t`; capped at `cap_minutes`; cap value if none. |
| | `mins_since_last_end` | `t −` latest known `finished_at <= t`; same cap rules. |
| | `last_alert_duration_min` | Duration of the most recent alert with known end `<= t`; NaN if none. |
| **History** | `starts_{1h,3h,6h,12h,24h,72h,7d}` | Number of starts in window. |
| | `alert_min_{6h,24h,7d}` | Minutes in window where ≥1 alert open (end clipped at `t`). |
| | `start_rate_30d` | Starts per day over the past 30 days in `o`. |
| **Neighbors** (E) | `nbr_active_cnt`, `nbr_active_share` | Count/share of adjacent oblasts active now. |
| | `nbr_starts_{1h,3h,6h}` | Sum of starts in adjacent oblasts in window. |
| | `nbr_min_since_last_start` | Min over neighbors of minutes since last start (capped). |
| | `nbr2_active_cnt` | Count of 2-hop neighbors active now (P1). |
| **National** | `nat_active_cnt`, `nat_starts_{1h,3h}` | Number of oblasts active now; nationwide starts in window (wave indicator). |
| | `nat_start_rate_30d` | Nationwide starts/day over the past 30 days (regime/drift proxy). |

**Neighbor map:** a static adjacency dict in `src/alertcast/geo.py` (land borders between oblasts; treat Kyiv City as adjacent to Kyiv Oblast; ignore Crimea/Luhansk if excluded). Geography is time-invariant, so this is not leakage. Add a unit test: adjacency is symmetric and covers every modeled oblast. Document the source/assumption in `docs/DECISIONS.md`.

Do NOT use: tsfresh, centered rolling windows, global target encoding, global normalization, forward/backward fill across `t`, or any statistic computed over the full dataset.

---

## 8. Modeling pipeline

### 8.1 Baselines (mandatory; all evaluated on identical rows)
| ID | Baseline | Score |
|---|---|---|
| B0 | **Prior** | Per-oblast training-period base rate of `y` (fit on `t + H <= cutoff`). |
| B1 | **Recent activity** | `1` if `starts_3h > 0` else `0` (score = `starts_3h`, tie-broken by recency). |
| B2 | **Hourly climatology** | Per `(oblast, hour_local)` empirical rate of `y` from training rows, with additive (Laplace) smoothing. |
| B3 | **Trailing base rate** | As-of per-oblast rate of 3h-windows with a start over the past 14 days (computed from `starts_*`-style as-of data). |

### 8.2 Primary model
- `LightGBMClassifier` (sklearn API), objective `binary`, one **pooled** model over all oblasts with `oblast_id` as categorical.
- Determinism: `seed`, `deterministic=True`, `force_row_wise=True`, fixed `n_jobs`.
- No `scale_pos_weight` by default (preserves probability calibration); optional ablation.
- **No early stopping against any validation or test data.** `n_estimators` is in the search space.

### 8.3 Hyperparameter search (walk-forward, pre-test only)
- `sklearn.model_selection.ParameterSampler`, `n_iter=25` (reduce to 10 if short on time), seed fixed.
- Space: `num_leaves ∈ {15,31,63}`, `learning_rate ∈ {0.02,0.05,0.1}`, `n_estimators ∈ {200,400,800}`, `min_child_samples ∈ {20,50,100}`, `feature_fraction ∈ {0.6,0.8,1.0}`, `reg_lambda ∈ {0,1,10}`, `train_window_days ∈ {180,365,730,None}`.
- Objective: **mean PR-AUC on the calm subset across the walk-forward validation folds** (tie-break: lower mean Brier).
- All folds from `walk_forward_folds` (§5.5); log per-fold metrics to `artifacts/tuning_results.csv`.

### 8.4 Final fit, threshold, calibration
1. Refit the best config on all pre-test rows with `t + H <= test_start`.
2. **Operating threshold** chosen on **out-of-fold validation predictions only** (maximize F2, recall-weighted; also store the thresholds for recall = 0.8 and 0.9). Frozen before test.
3. Calibration (P1): fit isotonic or Platt on out-of-fold validation predictions only; report calibrated vs raw Brier. If not done, report raw reliability honestly.
4. Save `artifacts/model.joblib`, `artifacts/oblasts.json`, `artifacts/threshold.json`, `artifacts/best_params.json`, `artifacts/run_manifest.json` (git SHA, library versions, data hash, seeds, `test_start`, `data_end`).

### 8.5 One-shot test evaluation
`scripts/run_test_eval.py` loads the frozen artifacts, builds test features, predicts, scores all baselines and the model, and writes `reports/metrics_test.json` and `reports/metrics_by_oblast.csv`. It refuses to run if `artifacts/run_manifest.json` records `test_start` different from the configured one or if the leakage gate fails.

---

## 9. Evaluation and metrics

All metrics on the **test window**, headline = **calm subset**, secondary = all rows. Report model and B0–B3 side by side.

| Metric | Notes |
|---|---|
| **PR-AUC** (average precision) | Primary. Pooled and macro-averaged across oblasts. |
| Brier score + **Brier skill score** vs B2 | Probability quality. |
| Log loss, ROC-AUC | Secondary. |
| **Calibration** | Reliability diagram data (10 quantile bins) + ECE. |
| Precision / recall / F2 at frozen threshold | Plus the recall=0.8 operating point. |
| **Event-level hit rate** | Fraction of actual alert starts (calm-state) that had ≥1 hourly forecast above threshold in the preceding 3 h. |
| **False alarms** | Threshold exceedances not followed by a start within `H`, per oblast-day. |
| Per-oblast table | PR-AUC, base rate, Brier for each oblast; flag oblasts where the model loses to B2. |
| Uncertainty (P1) | Day-block bootstrap (1000 reps) 95% CI for PR-AUC and for the (model − best baseline) difference. |
| Drift view (P1) | Weekly PR-AUC over the test window. |

Required statement in the report: whether the model beats B2 and B3 on the headline metric, with the numbers, even if it does not.

---

## 10. Visualizations (REQUIRED: 3 presentation-ready figures)

Use **Seaborn**, **Matplotlib**, or **Plotly** (with static PNG export via `kaleido`) to generate static images only. HTML exports are strictly prohibited. Ensure a consistent dark theme or clear white background... Output PNGs to `reports/figures/`.

1. **`fig1_heatmap_hour_dow.{html,png}`: Time-of-Day × Day-of-Week heatmap.**
   - Axes: x = hour of day (`Europe/Kyiv` local, 0–23), y = day of week (Mon–Sun). Cell = number of alert starts (nationwide default; add a dropdown/selector for individual oblasts in the HTML version).
   - **Computed on the pre-test (training) period only**, stated in the subtitle, keeping the project's no-leakage narrative consistent.
   - Sequential colorscale, colorbar labeled, annotated peak cell.

2. **`fig2_feature_importance.{html,png}`: LightGBM feature importance.**
   - Horizontal bar chart of the top 20 features by **gain** (normalized to % of total gain), sorted descending, color-coded by feature group (state/recency/history/neighbor/national/calendar).
   - Subtitle notes: "gain importance is descriptive, not causal".

3. **`fig3_forecast_vs_actual.{html,png}`: Forecast vs. Actual overlay.**
   - One oblast over the **most recent 7 days** of the test window. Default oblast = highest training-period alert count (selected by a rule, not by eyeballing test results); parameterized via config/CLI.
   - Line: `p̂(t)` = predicted probability of a start in `(t, t+3h]`, plotted at issue time `t`.
   - Red shaded bands: actual alert intervals. Triangle markers: actual alert start times. Dashed horizontal line: frozen threshold. Optional second trace: B2 climatology in grey.
   - Annotation: "forecast issued at t covers (t, t+3h]".

Figures MUST be generated by `make figures` from saved artifacts, without retraining.

---

## 11. Repository layout and interfaces

```
alertcast/
├── README.md  PRD.md  Makefile  pyproject.toml  config.yaml  .gitignore
├── data/{raw,interim,processed}/       # gitignored except MANIFEST.json
├── artifacts/                          # model, thresholds, tuning results (gitignored except small JSON)
├── reports/{figures,metrics_test.json,metrics_by_oblast.csv,REPORT.md}
├── docs/{DATA_AUDIT.md,DECISIONS.md,DEVLOG.md}
├── scripts/{fetch_data.py,audit_data.py,run_pipeline.py,run_test_eval.py,make_figures.py}
├── src/alertcast/
│   ├── config.py        # dataclasses / pydantic-free typed config loaded from config.yaml
│   ├── ingest.py        # load + clean (row-local) → alerts.parquet
│   ├── geo.py           # oblast name mapping, adjacency, permanent-siren exclusions
│   ├── as_of.py         # as_of_view
│   ├── labels.py        # y(o,t)  (the ONLY forward-looking module)
│   ├── features/{reference.py,vectorized.py,names.py}
│   ├── splits.py        # purged walk-forward folds, test window
│   ├── baselines.py     # B0–B3
│   ├── training.py      # tuning, final fit, threshold (with fit-scope assertions)
│   ├── evaluation.py    # metrics, bootstrap, event-level analysis
│   └── viz.py           # three figures
└── tests/{leakage/,unit/}
```

**`config.yaml` (initial):**
```yaml
data: {source: volunteer, file: datasets/volunteer_data_en.csv}
horizon_hours: 3
warmup_days: 30
test_days: 90
min_train_starts: 50
cap_minutes: 20160          # 14 days
walk_forward: {n_folds: 5, valid_days: 28}
search: {n_iter: 25, seed: 42}
seed: 42
viz: {oblast: auto, window_days: 7}
```

**`Makefile` targets:** `setup`, `fetch`, `audit`, `test` (all tests), `leakage` (leakage marker only), `train`, `eval`, `figures`, `all` (= setup → fetch → audit → leakage → train → eval → figures).

**Dependencies (pin in `pyproject.toml`):** pandas, numpy, scikit-learn, lightgbm, pyarrow, plotly, kaleido, pyyaml, joblib, pytest. No additional heavy frameworks.

---

## 12. Logging and reporting
- `reports/REPORT.md` auto-generated: dataset summary, split dates, leakage-gate results (pass counts), baseline vs model table, per-oblast table, figure links, limitations.
- `docs/DEVLOG.md`: chronological log of AI-agent actions and human interventions (supports the "AI as primary engineering tool" requirement).

---

## 13. Priorities and acceptance criteria

### P0 (must ship tonight)
- [ ] M0 data audit + source decision recorded.
- [ ] `as_of_view`, reference + vectorized features, labels, purged splits.
- [ ] Leakage tests §6.0–§6.5 and §6.7 passing; runtime gate wired into the pipeline.
- [ ] Baselines B0–B3, LightGBM with small walk-forward search (`n_iter` may be cut to 10).
- [ ] One-shot test evaluation with headline metrics and honest comparison.
- [ ] Three required figures (HTML at minimum).
- [ ] README filled in with real numbers.

### P1 (if time permits)
- [ ] §6.6 static tests, §6.8 alarm, calibration, bootstrap CIs, weekly drift view, 2-hop neighbor features.

### P2 (stretch)
- [ ] Horizon sweep (`H ∈ {1, 6}`), monthly walk-forward retraining in the test window, `info_lag_minutes` sensitivity (hide the last 1–5 minutes of events to mimic feed latency), ablation without neighbor features, per-oblast models, Phase 2 text-signal prototype (only after everything else is done and documented as such).

### Definition of Done
1. `make all` runs from a clean checkout and produces metrics and the 3 figures.
2. `pytest -m leakage` is green, including the canary test that proves leak detection works.
3. The test window is exactly the last 90 days and was evaluated once, after freezing everything.
4. README results table is filled from `reports/metrics_test.json` (no hand-typed numbers).
5. Limitations section is present and truthful.

### Suggested time-boxing (adjust freely)
M0 data audit (20 min) → as-of layer + reference features + tests (60 min) → vectorized features + equivalence (45 min) → baselines/splits/training (45 min) → test eval (20 min) → figures (30 min) → README/report (30 min).

---

## 14. Risks, limitations and disclaimers
- **Data vintage:** retrospectively corrected data; real-time latency and gaps not modeled.
- **Naive end imputation:** `finished_at = started_at + 30 min` for rows with missing end signals is an approximation; affects `alert_min_*`, `mins_since_last_end` and activity state, not labels.
- **Source break risk:** the dataset documentation states that oblast-wide sirens were largely replaced by raion-level alerts since December 2025. The audit gate (§3.2) decides and documents how "oblast alert" is defined.
- **Non-stationarity:** weapon types and tactics evolve; the frozen-model test measures degradation honestly.
- **Distribution of events:** attacks are adversarial; models trained on past patterns can fail on novel tactics.
- **Not for operational use.** This project is a research and portfolio demonstration. It MUST NOT be presented or used as a substitute for the official air raid alert system. This disclaimer MUST appear in the README.

---

## 15. Rules for the AI coding agent
1. Work milestone by milestone (M0 → P0 list). After each milestone, run the relevant tests and append to `docs/DEVLOG.md`.
2. **Write the leakage tests before the feature code** (test-first for `tests/leakage/`).
3. Never edit, relax, skip (`xfail`/`skip`), or delete a leakage test to get green.
4. Never read the test window for any decision before `scripts/run_test_eval.py`.
5. Never add a data source, text/media feature, or network call outside `scripts/fetch_data.py`.
6. Keep functions small and typed; docstrings state the information set used.
7. Seeds everywhere; no wall-clock dependence in `src/alertcast/features/` or `labels.py`.
8. When a decision isn't covered by this PRD, pick the option that is safer against leakage and log it in `docs/DECISIONS.md`.
