"""Automatic checks that run on GitHub every time the code changes."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import run, DetectorConfig  # noqa: E402
from engine.evaluation import point_metrics, event_metrics, per_type_recall  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(HERE, "sample_data", "synthetic_transactions.csv")
RAW = os.path.join(HERE, "sample_data", "example_raw_payments.csv")


def _sample():
    return run(pd.read_csv(SAMPLE))


def test_project_results_are_reproduced():
    scored, inc, _ = _sample()
    em, pm = event_metrics(scored), point_metrics(scored)
    assert em["events_detected"] == em["events_total"] == 11      # every incident detected
    assert pm["recall"] >= 0.80
    assert pm["precision"] >= 0.30


def test_every_anomaly_type_detected():
    scored, _, _ = _sample()
    assert (per_type_recall(scored)["recall"] == 1.0).all()


def test_incidents_have_cause_and_actions():
    _, inc, _ = _sample()
    serious = inc[inc.level.isin(["Critical", "High"])]
    assert len(serious) >= 10
    assert serious.likely_cause.str.len().gt(10).all()
    assert serious.actions.map(len).ge(4).all()


def test_raw_payment_upload_is_understood():
    scored, inc, rep = run(pd.read_csv(RAW))
    assert rep["shape"] == "raw"
    auth = inc[(inc.type == "AUTH_DROP") & (inc.level == "Critical")]
    assert len(auth) == 1 and "Mastercard" in auth.iloc[0].likely_cause


def test_odd_column_names_work():
    df = pd.read_csv(SAMPLE).rename(columns={"timestamp": "DateTime", "merchant_id": "Merchant",
                                             "txn_count": "Transactions", "auth_count": "Approved",
                                             "total_value": "Amount"})
    scored, inc, rep = run(df.drop(columns=["auth_rate", "avg_ticket"]))
    assert rep["merchants"] == 5 and len(inc) > 0
