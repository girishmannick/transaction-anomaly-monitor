

import numpy as np
import pandas as pd


#  Reproducibility

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


#  Configuration

START_DATE = "2024-01-01"
N_DAYS = 120                     # ~4 months of hourly data
FREQ = "h"

# Merchant catalogue: (id, display name, baseline hourly volume, sector)
MERCHANTS = [
    ("M001", "Aurora Apparel",      55,  "Retail/Fashion"),
    ("M002", "ByteBox Software",     30,  "Digital/SaaS"),
    ("M003", "Crave Food Delivery",  90,  "Food Delivery"),
    ("M004", "DriftTravel",          22,  "Travel"),
    ("M005", "EverGreen Grocers",    70,  "Grocery"),
]

# Average authorisation (approval) rate per merchant under normal conditions.
BASE_AUTH_RATE = {
    "M001": 0.93, "M002": 0.90, "M003": 0.95, "M004": 0.88, "M005": 0.96
}

# Log-normal parameters (in log space) for the per-transaction value (in EUR).
# mu, sigma chosen so that median ticket sizes look sensible per sector.
TICKET_PARAMS = {
    "M001": (3.7, 0.5),
    "M002": (3.0, 0.7),
    "M003": (2.9, 0.4),
    "M004": (5.5, 0.6),
    "M005": (3.3, 0.4),
}



#  Seasonality profiles

def intraday_profile(hour: int) -> float:

    # A smooth double-bump curve built from two Gaussians (lunch + evening).
    lunch = np.exp(-0.5 * ((hour - 13) / 3.0) ** 2)
    evening = np.exp(-0.5 * ((hour - 20) / 2.5) ** 2)
    night_floor = 0.15
    raw = night_floor + 0.9 * lunch + 1.0 * evening
    return raw


def weekly_profile(dayofweek: int) -> float:

    # Retail-style week: busier Fri-Sun, quieter mid-week.
    profile = {0: 0.95, 1: 0.92, 2: 0.94, 3: 1.00, 4: 1.12, 5: 1.25, 6: 1.18}
    return profile[dayofweek]


# Pre-compute a normalisation so the intraday factors average to 1.0.
_INTRADAY_MEAN = np.mean([intraday_profile(h) for h in range(24)])



#  Core series construction

def build_merchant_series(mid: str, base_volume: float,
                          index: pd.DatetimeIndex) -> pd.DataFrame:

    n = len(index)

    hours = index.hour.to_numpy()
    dows = index.dayofweek.to_numpy()

    intraday = np.array([intraday_profile(h) for h in hours]) / _INTRADAY_MEAN
    weekly = np.array([weekly_profile(d) for d in dows])

    # Slow linear growth: +25% over the whole window.
    t = np.arange(n) / n
    trend = 1.0 + 0.25 * t

    # Expected hourly transaction rate (lambda for the Poisson draw).
    expected = base_volume * intraday * weekly * trend
    expected = np.clip(expected, 0.5, None)

    # Poisson count noise around the expected rate.
    counts = rng.poisson(expected).astype(float)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "merchant_id": mid,
            "expected_volume": expected,
            "txn_count": counts,
        }
    )
    return df


def attach_value_and_auth(df: pd.DataFrame, mid: str) -> pd.DataFrame:

    mu, sigma = TICKET_PARAMS[mid]
    auth_rate = BASE_AUTH_RATE[mid]

    total_value = np.zeros(len(df))
    auth_count = np.zeros(len(df))

    counts = df["txn_count"].to_numpy().astype(int)
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        # Per-transaction monetary value (log-normal), summed for the hour.
        values = rng.lognormal(mean=mu, sigma=sigma, size=c)
        total_value[i] = values.sum()
        # Authorisation outcomes (binomial around the merchant base rate).
        auth_count[i] = rng.binomial(c, auth_rate)

    df = df.copy()
    df["total_value"] = np.round(total_value, 2)
    df["auth_count"] = auth_count.astype(int)
    return df


#  Anomaly injection

def _window_mask(index, start, length_hours):
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(hours=length_hours)
    return (index >= start_ts) & (index < end_ts)


def inject_anomalies(df: pd.DataFrame, mid: str) -> pd.DataFrame:

    df = df.copy()
    df["is_anomaly"] = 0
    df["anomaly_type"] = ""
    index = df["timestamp"]

    # A small per-merchant schedule of events. Times are chosen to fall on
    # different days so they do not overlap.
    schedule = {
        "M001": [
            ("2024-01-20 10:00", 8,  "SPIKE", 4.0),     # flash sale
            ("2024-02-14 00:00", 14, "DROP",  0.15),    # checkout outage
            ("2024-03-05 09:00", 10, "AUTH_DROP", None),# acquirer issue
        ],
        "M002": [
            ("2024-02-02 03:00", 6,  "SPIKE", 6.0),     # card-testing attack
            ("2024-03-18 08:00", 18, "DROP",  0.10),    # API outage
        ],
        "M003": [
            ("2024-01-28 18:00", 5,  "SPIKE", 3.5),     # promo / event
            ("2024-02-25 12:00", 12, "AUTH_DROP", None),# BIN routing fault
        ],
        "M004": [
            ("2024-02-10 06:00", 9,  "DROP",  0.12),    # gateway outage
            ("2024-03-22 11:00", 7,  "SPIKE", 5.0),     # fare sale
        ],
        "M005": [
            ("2024-01-15 16:00", 6,  "SPIKE", 3.0),
            ("2024-03-12 07:00", 16, "DROP",  0.20),
        ],
    }

    for start, length, atype, magnitude in schedule.get(mid, []):
        mask = _window_mask(index, start, length).to_numpy()
        if not mask.any():
            continue

        if atype == "SPIKE":
            df.loc[mask, "txn_count"] = (
                df.loc[mask, "txn_count"] * magnitude
            ).round().astype(int)
            # Re-derive value & auth for the affected hours.
            mu, sigma = TICKET_PARAMS[mid]
            for i in np.where(mask)[0]:
                c = int(df.iloc[i]["txn_count"])
                vals = rng.lognormal(mu, sigma, size=max(c, 0))
                df.iat[i, df.columns.get_loc("total_value")] = round(vals.sum(), 2)
                df.iat[i, df.columns.get_loc("auth_count")] = int(
                    rng.binomial(c, BASE_AUTH_RATE[mid])
                )

        elif atype == "DROP":
            df.loc[mask, "txn_count"] = (
                df.loc[mask, "txn_count"] * magnitude
            ).round().astype(int)
            for i in np.where(mask)[0]:
                c = int(df.iloc[i]["txn_count"])
                mu, sigma = TICKET_PARAMS[mid]
                vals = rng.lognormal(mu, sigma, size=max(c, 0))
                df.iat[i, df.columns.get_loc("total_value")] = round(vals.sum(), 2)
                df.iat[i, df.columns.get_loc("auth_count")] = int(
                    rng.binomial(c, BASE_AUTH_RATE[mid])
                )

        elif atype == "AUTH_DROP":
            # Volume looks normal; authorisation rate collapses to ~30%.
            for i in np.where(mask)[0]:
                c = int(df.iloc[i]["txn_count"])
                df.iat[i, df.columns.get_loc("auth_count")] = int(
                    rng.binomial(c, 0.30)
                )

        df.loc[mask, "is_anomaly"] = 1
        df.loc[mask, "anomaly_type"] = atype

    return df



#  Public entry point

def generate_dataset() -> pd.DataFrame:
    """Generate the full multi-merchant dataset and return a tidy dataframe."""
    index = pd.date_range(start=START_DATE, periods=N_DAYS * 24, freq=FREQ)

    frames = []
    for mid, name, base, sector in MERCHANTS:
        s = build_merchant_series(mid, base, index)
        s = attach_value_and_auth(s, mid)
        s = inject_anomalies(s, mid)
        s["merchant_name"] = name
        s["sector"] = sector
        frames.append(s)

    data = pd.concat(frames, ignore_index=True)

    # Derived fields useful downstream.
    data["auth_rate"] = np.where(
        data["txn_count"] > 0, data["auth_count"] / data["txn_count"], np.nan
    )
    data["avg_ticket"] = np.where(
        data["txn_count"] > 0, data["total_value"] / data["txn_count"], np.nan
    )

    # Tidy column order.
    cols = [
        "timestamp", "merchant_id", "merchant_name", "sector",
        "txn_count", "total_value", "auth_count", "auth_rate", "avg_ticket",
        "is_anomaly", "anomaly_type",
    ]
    data = data[cols].sort_values(["merchant_id", "timestamp"]).reset_index(drop=True)
    return data


if __name__ == "__main__":
    df = generate_dataset()
    out = "synthetic_transactions.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} rows for {df['merchant_id'].nunique()} merchants.")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Injected anomaly rows: {int(df['is_anomaly'].sum()):,} "
          f"({100*df['is_anomaly'].mean():.2f}% of data)")
    print(f"Saved to {out}")
    print(df.head())