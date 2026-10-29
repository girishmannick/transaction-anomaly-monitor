
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



#  Small statistical helpers

def mad(x: np.ndarray) -> float:

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def robust_z(x: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    scale = mad(x)
    if scale == 0:
        scale = np.nanstd(x) if np.nanstd(x) > 0 else 1.0
    return (x - med) / scale



#  Configuration object

@dataclass
class DetectorConfig:

    z_threshold: float = 4.0          # robust z-score alert threshold
    rolling_window_days: int = 14     # window for adaptive decomposition
    roll_threshold: float = 4.0       # threshold on rolling residual score
    iforest_contamination: float = 0.015
    iforest_estimators: int = 300
    auth_drop_z: float = 3.5          # threshold for authorisation collapse
    ensemble_min_votes: int = 2       # votes required to raise a flag
    random_state: int = 42



#  Method 1 - Seasonal robust z-score

def seasonal_robust_zscore(df: pd.DataFrame, value_col: str,
                           cfg: DetectorConfig) -> pd.Series:

    work = df.copy()
    work["dow"] = work["timestamp"].dt.dayofweek
    work["hour"] = work["timestamp"].dt.hour

    scores = pd.Series(index=work.index, dtype=float)
    for (_, _), grp in work.groupby(["dow", "hour"]):
        scores.loc[grp.index] = robust_z(grp[value_col].to_numpy())
    return scores



#  Method 2 - Rolling seasonal decomposition

def rolling_decomposition_score(df: pd.DataFrame, value_col: str,
                                cfg: DetectorConfig) -> pd.Series:

    s = df.sort_values("timestamp").reset_index()
    series = s[value_col].astype(float)

    # --- trend: 25-hour centred rolling median (captures the daily centre) ---
    trend = series.rolling(window=25, center=True, min_periods=6).median()
    detrended = series - trend

    # --- seasonality: median detrended value per (dow, hour) ---
    s["dow"] = s["timestamp"].dt.dayofweek
    s["hour"] = s["timestamp"].dt.hour
    s["_detrended"] = detrended
    seasonal = s.groupby(["dow", "hour"])["_detrended"].transform("median")

    resid = detrended - seasonal

    # --- adaptive standardisation with a trailing window ---
    win = cfg.rolling_window_days * 24
    roll_med = resid.rolling(window=win, min_periods=win // 3).median()
    roll_scale = resid.rolling(window=win, min_periods=win // 3).apply(
        lambda a: mad(a), raw=True
    )
    roll_scale = roll_scale.replace(0, np.nan).fillna(resid.std() or 1.0)

    score = (resid - roll_med) / roll_scale
    # Map back to the original (unsorted) index.
    out = pd.Series(index=df.index, dtype=float)
    out.loc[s["index"].to_numpy()] = score.to_numpy()
    return out



#  Method 3 - Isolation Forest on engineered features

def isolation_forest_score(df: pd.DataFrame, cfg: DetectorConfig) -> pd.Series:

    work = df.copy()
    work["hour"] = work["timestamp"].dt.hour
    work["dow"] = work["timestamp"].dt.dayofweek

    # Deseasonalise volume and value so the model sees deviations, not season.
    vol_z = seasonal_robust_zscore(work, "txn_count", cfg)
    val_z = seasonal_robust_zscore(work, "total_value", cfg)

    feats = pd.DataFrame({
        "vol_z": vol_z,
        "val_z": val_z,
        "auth_rate": work["auth_rate"].fillna(work["auth_rate"].median()),
        "avg_ticket": work["avg_ticket"].fillna(work["avg_ticket"].median()),
        "hour_sin": np.sin(2 * np.pi * work["hour"] / 24),
        "hour_cos": np.cos(2 * np.pi * work["hour"] / 24),
        "dow": work["dow"],
    }).fillna(0.0)

    X = StandardScaler().fit_transform(feats)
    model = IsolationForest(
        n_estimators=cfg.iforest_estimators,
        contamination=cfg.iforest_contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X)
    # decision_function: higher = more normal. Invert and min-max to [0,1].
    raw = -model.decision_function(X)
    rng = raw.max() - raw.min()
    norm = (raw - raw.min()) / rng if rng > 0 else np.zeros_like(raw)
    return pd.Series(norm, index=df.index)



#  Authorisation-rate collapse detector

def auth_collapse_score(df: pd.DataFrame, cfg: DetectorConfig) -> pd.Series:

    z = seasonal_robust_zscore(df, "auth_rate", cfg)
    # We only care about the downside (auth rate falling).
    return (-z).clip(lower=0)



#  Ensemble orchestration

@dataclass
class AnomalyResult:

    scored: pd.DataFrame
    alerts: pd.DataFrame
    config: DetectorConfig = field(default_factory=DetectorConfig)


def detect(df: pd.DataFrame, cfg: DetectorConfig | None = None) -> AnomalyResult:

    cfg = cfg or DetectorConfig()
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    # --- individual scores ------------------------------------------------- #
    df["score_z"] = seasonal_robust_zscore(df, "txn_count", cfg)
    df["score_roll"] = rolling_decomposition_score(df, "txn_count", cfg)
    df["score_iforest"] = isolation_forest_score(df, cfg)
    df["score_auth"] = auth_collapse_score(df, cfg)

    # --- per-method binary votes ------------------------------------------- #
    df["vote_z"] = (df["score_z"].abs() >= cfg.z_threshold).astype(int)
    df["vote_roll"] = (df["score_roll"].abs() >= cfg.roll_threshold).astype(int)
    iforest_cut = df["score_iforest"].quantile(1 - cfg.iforest_contamination)
    df["vote_iforest"] = (df["score_iforest"] >= iforest_cut).astype(int)
    df["vote_auth"] = (df["score_auth"] >= cfg.auth_drop_z).astype(int)

    # --- ensemble decision ------------------------------------------------- #
    df["votes"] = df[["vote_z", "vote_roll", "vote_iforest"]].sum(axis=1)
    df["is_flagged"] = (
        (df["votes"] >= cfg.ensemble_min_votes) | (df["vote_auth"] == 1)
    ).astype(int)

    # --- direction / type classification ----------------------------------- #
    def classify(row):
        if row["vote_auth"] == 1 and row["votes"] < cfg.ensemble_min_votes:
            return "AUTH_DROP"
        if row["is_flagged"]:
            if row["score_z"] > 0 or row["score_roll"] > 0:
                return "SPIKE"
            return "DROP"
        return ""
    df["predicted_type"] = df.apply(classify, axis=1)

    # --- severity score (0-100) for ranking alerts ------------------------- #
    sev = (
        df["score_z"].abs().fillna(0) * 0.35
        + df["score_roll"].abs().fillna(0) * 0.35
        + df["score_iforest"].fillna(0) * 10 * 0.15
        + df["score_auth"].fillna(0) * 0.15
    )
    sev = (sev / sev.max() * 100) if sev.max() > 0 else sev
    df["severity"] = sev.round(1)

    alerts = (
        df[df["is_flagged"] == 1]
        .sort_values("severity", ascending=False)
        [["timestamp", "merchant_id", "txn_count", "total_value",
          "auth_rate", "predicted_type", "severity", "votes",
          "score_z", "score_roll", "score_iforest", "score_auth"]]
        .reset_index(drop=True)
    )

    return AnomalyResult(scored=df, alerts=alerts, config=cfg)


def detect_portfolio(df: pd.DataFrame,
                     cfg: DetectorConfig | None = None) -> AnomalyResult:
    """Convenience wrapper that runs `detect` for every merchant and stitches
    the results back together."""
    cfg = cfg or DetectorConfig()
    scored_frames, alert_frames = [], []
    for mid, grp in df.groupby("merchant_id"):
        res = detect(grp, cfg)
        scored_frames.append(res.scored)
        alert_frames.append(res.alerts)
    scored = pd.concat(scored_frames, ignore_index=True)
    alerts = pd.concat(alert_frames, ignore_index=True).sort_values(
        "severity", ascending=False).reset_index(drop=True)
    return AnomalyResult(scored=scored, alerts=alerts, config=cfg)