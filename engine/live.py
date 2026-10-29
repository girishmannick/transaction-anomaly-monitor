"""
live.py
A simulated live payment feed built on the project's own generator (generate_data.py),
plus helpers to inject incidents on demand for demonstrations.
"""
import numpy as np
import pandas as pd

from . import generate_data as gd

INJECTABLE = {
    "Checkout outage (volume drop)": "DROP",
    "Approval-rate collapse": "AUTH_DROP",
    "Card-testing attack": "CARD_TESTING",
    "Flash-sale promotion (spike)": "SPIKE",
}
MERCHANTS = {m[0]: m[1] for m in gd.MERCHANTS}


def full_stream(seed=7):
    """Four months of hourly data for the 5 merchants (includes the generator's scheduled incidents)."""
    gd.rng = np.random.default_rng(seed)
    return gd.generate_dataset()


def inject(df, merchant_id, kind, start, hours=4, seed=11):
    rng = np.random.default_rng(seed)
    df = df.copy()
    m = (df["merchant_id"] == merchant_id) & (df["timestamp"] >= start) & \
        (df["timestamp"] < start + pd.Timedelta(hours=hours))
    if not m.any():
        return df
    mu, sigma = gd.TICKET_PARAMS.get(merchant_id, (3.5, 0.5))
    base_auth = gd.BASE_AUTH_RATE.get(merchant_id, 0.92)
    for i in np.where(m)[0]:
        c = int(df.iloc[i]["txn_count"])
        if kind == "DROP":
            c = int(round(c * 0.08))
            df.iat[i, df.columns.get_loc("txn_count")] = c
            df.iat[i, df.columns.get_loc("total_value")] = round(rng.lognormal(mu, sigma, c).sum(), 2)
            df.iat[i, df.columns.get_loc("auth_count")] = int(rng.binomial(c, base_auth))
        elif kind == "SPIKE":
            c = int(round(c * 3.5))
            df.iat[i, df.columns.get_loc("txn_count")] = c
            df.iat[i, df.columns.get_loc("total_value")] = round(rng.lognormal(mu, sigma, c).sum(), 2)
            df.iat[i, df.columns.get_loc("auth_count")] = int(rng.binomial(c, base_auth))
        elif kind == "AUTH_DROP":
            df.iat[i, df.columns.get_loc("auth_count")] = int(rng.binomial(c, 0.30))
        elif kind == "CARD_TESTING":
            extra = int(max(c, 20) * 4)
            ok = int(rng.binomial(extra, 0.12))
            df.iat[i, df.columns.get_loc("txn_count")] = c + extra
            df.iat[i, df.columns.get_loc("total_value")] = float(df.iloc[i]["total_value"]) + round(rng.uniform(0.5, 2.0, extra).sum(), 2)
            df.iat[i, df.columns.get_loc("auth_count")] = int(df.iloc[i]["auth_count"]) + ok
    df.loc[m, "auth_rate"] = df.loc[m, "auth_count"] / df.loc[m, "txn_count"].clip(lower=1)
    df.loc[m, "avg_ticket"] = df.loc[m, "total_value"] / df.loc[m, "txn_count"].clip(lower=1)
    df.loc[m, "is_anomaly"] = 1
    df.loc[m, "anomaly_type"] = kind
    return df
