
from __future__ import annotations
import numpy as np
import pandas as pd


def point_metrics(scored: pd.DataFrame) -> dict:

    y_true = scored["is_anomaly"].to_numpy().astype(int)
    y_pred = scored["is_flagged"].to_numpy().astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / len(y_true)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 4),
    }


def _label_events(mask: np.ndarray) -> list[tuple[int, int]]:

    events = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            events.append((start, i - 1))
    if in_run:
        events.append((start, len(mask) - 1))
    return events


def event_metrics(scored: pd.DataFrame) -> dict:

    df = scored.sort_values(["merchant_id", "timestamp"]).reset_index(drop=True)
    detected, total = 0, 0
    detection_delays = []

    for _, grp in df.groupby("merchant_id"):
        truth = grp["is_anomaly"].to_numpy().astype(bool)
        pred = grp["is_flagged"].to_numpy().astype(bool)
        for (s, e) in _label_events(truth):
            total += 1
            window_pred = pred[s:e + 1]
            if window_pred.any():
                detected += 1
                # Delay = hours from event start to first detection.
                first = np.argmax(window_pred)
                detection_delays.append(int(first))

    recall_event = detected / total if total else 0.0
    mean_delay = float(np.mean(detection_delays)) if detection_delays else np.nan
    return {
        "events_total": total,
        "events_detected": detected,
        "event_recall": round(recall_event, 3),
        "mean_detection_delay_hours": round(mean_delay, 2),
    }


def per_type_recall(scored: pd.DataFrame) -> pd.DataFrame:
    """Event-level recall broken down by anomaly type."""
    df = scored.sort_values(["merchant_id", "timestamp"]).reset_index(drop=True)
    rows = []
    for atype in ["SPIKE", "DROP", "AUTH_DROP"]:
        total, detected = 0, 0
        for _, grp in df.groupby("merchant_id"):
            truth = (grp["anomaly_type"] == atype).to_numpy()
            pred = grp["is_flagged"].to_numpy().astype(bool)
            for (s, e) in _label_events(truth):
                total += 1
                if pred[s:e + 1].any():
                    detected += 1
        rows.append({
            "anomaly_type": atype,
            "events": total,
            "detected": detected,
            "recall": round(detected / total, 3) if total else np.nan,
        })
    return pd.DataFrame(rows)


def method_agreement(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-method point recall, to show each detector's individual
    contribution to the ensemble."""
    y = scored["is_anomaly"].to_numpy().astype(bool)
    rows = []
    for col, name in [("vote_z", "Robust z-score"),
                      ("vote_roll", "Rolling decomposition"),
                      ("vote_iforest", "Isolation Forest"),
                      ("vote_auth", "Auth-collapse"),
                      ("is_flagged", "Ensemble (final)")]:
        pred = scored[col].to_numpy().astype(bool)
        tp = int((y & pred).sum())
        fp = int((~y & pred).sum())
        fn = int((y & ~pred).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "method": name,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        })
    return pd.DataFrame(rows)