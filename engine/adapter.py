"""
adapter.py
Turns any uploaded / live data into the exact table the detector expects:

    timestamp, merchant_id, merchant_name, txn_count, total_value,
    auth_count, auth_rate, avg_ticket   (+ is_anomaly, anomaly_type if present)

Two input shapes are accepted:
  * HOURLY  : one row per merchant per hour with a transaction-count column
  * RAW     : one row per payment (timestamp, merchant, amount, status ...)
              -> aggregated to hourly automatically.
Column names are matched case-insensitively against common aliases.
"""
import re

import numpy as np
import pandas as pd

ALIASES = {
    "timestamp": ["timestamp", "timestamp_utc", "datetime", "date_time", "time", "hour", "created_at", "date"],
    "merchant_id": ["merchant_id", "merchant", "mid", "merchant_code", "client_id", "entity_id"],
    "merchant_name": ["merchant_name", "name", "merchant_label"],
    "sector": ["sector", "vertical", "industry", "category"],
    "txn_count": ["txn_count", "transactions", "transaction_count", "count", "n_txns", "volume", "txns", "attempts"],
    "auth_count": ["auth_count", "authorised", "authorized", "approved", "approvals", "successful", "success_count"],
    "declined": ["declined", "declines", "decline_count", "failed", "failures"],
    "total_value": ["total_value", "value_eur", "value", "amount", "gmv", "total_amount", "processed_value"],
    "is_anomaly": ["is_anomaly", "anomaly", "label_flag"],
    "anomaly_type": ["anomaly_type", "label", "incident_type"],
    # raw-transaction fields
    "status": ["status", "outcome", "result", "response", "approved_flag", "is_approved", "auth_status"],
    "scheme": ["scheme", "card_scheme", "network", "brand", "card_brand"],
    "decline_code": ["decline_code", "response_code", "reason_code", "decline_reason", "error_code"],
}

DECLINE_NAMES = {"05": "do_not_honour", "51": "insufficient_funds", "14": "invalid_card", "54": "expired_card",
                 "91": "issuer_unavailable", "96": "system_malfunction", "57": "not_permitted", "61": "exceeds_limit",
                 "41": "lost_card", "43": "stolen_card", "59": "suspected_fraud", "65": "exceeds_frequency",
                 "62": "restricted_card", "13": "invalid_amount", "03": "invalid_merchant"}
SCHEMES = ["visa", "mastercard", "amex", "other"]


def _find(cols, key):
    low = {str(c).lower().strip(): c for c in cols}
    for a in ALIASES[key]:
        if a in low:
            return low[a]
    return None


def _code(x):
    m = re.search(r"\d{1,2}", str(x))
    return m.group(0).zfill(2) if m else None


def _to_naive_utc(s):
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t.dt.tz_localize(None)


def _aggregate_raw(df, m):
    out = pd.DataFrame({"timestamp": _to_naive_utc(df[m["timestamp"]]),
                        "merchant_id": df[m["merchant_id"]].astype(str)})
    st = df[m["status"]]
    vals = set(pd.Series(st).dropna().unique())
    if st.dtype == bool or vals <= {0, 1, True, False}:
        ok = st.astype(bool)
    else:
        ok = st.astype(str).str.lower().str.contains(r"approv|authori|success|captur|paid|^00$|^ok", regex=True)
    ok = ok.to_numpy()
    out["auth_count"] = ok.astype(int)
    out["txn_count"] = 1
    out["total_value"] = pd.to_numeric(df[m["total_value"]], errors="coerce").fillna(0).to_numpy() if m.get("total_value") else 0.0
    if m.get("merchant_name"):
        out["merchant_name"] = df[m["merchant_name"]].astype(str).to_numpy()
    if m.get("scheme"):
        sc = df[m["scheme"]].astype(str).str.lower()
        sc = np.where(sc.str.contains("visa"), "visa", np.where(sc.str.contains("master|mc"), "mastercard",
                      np.where(sc.str.contains("amex|american"), "amex", "other")))
        for s in SCHEMES:
            out[f"{s}_txns"] = (sc == s).astype(int)
            out[f"{s}_auth"] = ((sc == s) & ok).astype(int)
    if m.get("decline_code"):
        codes = df[m["decline_code"]].map(_code).where(~ok)
        for c in sorted(codes.dropna().unique()):
            out[f"d{c}_{DECLINE_NAMES.get(c, 'other')}"] = (codes == c).astype(int).to_numpy()
    out["timestamp"] = out["timestamp"].dt.floor("h")
    keys = ["timestamp", "merchant_id"] + (["merchant_name"] if "merchant_name" in out else [])
    return out.dropna(subset=["timestamp"]).groupby(keys, as_index=False).sum(numeric_only=True)


def prepare(df_in):
    """Return (hourly_df, report)."""
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    m = {k: _find(df.columns, k) for k in ALIASES}
    warnings = []
    if not m["timestamp"] or not m["merchant_id"]:
        raise ValueError("Could not find a timestamp column and a merchant column "
                         "(expected names such as 'timestamp' and 'merchant_id').")
    if m["txn_count"]:
        shape = "hourly"
    elif m["status"]:
        shape = "raw"
    else:
        raise ValueError("Could not find a transaction-count column (hourly data) or a "
                         "status/outcome column (one row per payment).")

    if shape == "raw":
        h = _aggregate_raw(df, m)
    else:
        keep = {v: k for k, v in m.items() if v and k not in ("status", "scheme", "decline_code")}
        extra = [c for c in df.columns if re.match(r"^(visa|mastercard|amex|other)_(txns|auth)$", c, re.I)
                 or re.match(r"^d\d{2}_", c)]
        h = df[list(keep) + [c for c in extra if c not in keep]].rename(columns=keep)
        h["timestamp"] = _to_naive_utc(h["timestamp"])
        h = h.dropna(subset=["timestamp"])
        h["timestamp"] = h["timestamp"].dt.floor("h")
        h["merchant_id"] = h["merchant_id"].astype(str)
        h["txn_count"] = pd.to_numeric(h["txn_count"], errors="coerce").fillna(0)
        if "auth_count" not in h:
            if "declined" in h:
                h["auth_count"] = h["txn_count"] - pd.to_numeric(h["declined"], errors="coerce").fillna(0)
            elif _find(df.columns, "auth_count") is None and "auth_rate" in [c.lower() for c in df.columns]:
                col = [c for c in df.columns if c.lower() == "auth_rate"][0]
                h["auth_count"] = (pd.to_numeric(df.loc[h.index, col], errors="coerce").fillna(1) * h["txn_count"]).round()
            else:
                h["auth_count"] = h["txn_count"]
                warnings.append("No approval/decline column found, so authorisation-collapse detection is switched off.")
        h = h.drop(columns=[c for c in ["declined"] if c in h])
        if "total_value" not in h:
            h["total_value"] = 0.0
            warnings.append("No amount/value column found, so ticket-size checks are limited.")
        for c in ("anomaly_type",):
            if c in h:
                h[c] = h[c].fillna("").astype(str).replace({"nan": "", "NORMAL": ""})
        num = [c for c in h.columns if c not in ("timestamp", "merchant_id") and pd.api.types.is_numeric_dtype(h[c])]
        txt = [c for c in ("merchant_name", "sector", "anomaly_type") if c in h]
        agg = {c: "sum" for c in num}
        if "is_anomaly" in agg:
            agg["is_anomaly"] = "max"
        agg.update({c: "first" for c in txt})
        h = h.groupby(["timestamp", "merchant_id"], as_index=False).agg(agg)

    # Complete the hourly grid: a missing hour is a real zero-volume hour (a possible outage)
    parts = []
    for mid, g in h.groupby("merchant_id"):
        idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        g = g.set_index("timestamp").reindex(idx)
        g.index.name = "timestamp"
        g["merchant_id"] = mid
        for c in g.columns:
            if c == "merchant_id":
                continue
            if pd.api.types.is_numeric_dtype(g[c]):
                g[c] = g[c].fillna(0)
            elif c in ("merchant_name", "sector"):
                g[c] = g[c].ffill().bfill().fillna("")
            else:
                g[c] = g[c].fillna("")
        parts.append(g.reset_index())
    h = pd.concat(parts, ignore_index=True)

    if "merchant_name" not in h:
        h["merchant_name"] = h["merchant_id"]
    h["merchant_name"] = h["merchant_name"].replace("", np.nan).fillna(h["merchant_id"])
    h["auth_count"] = np.minimum(h["auth_count"], h["txn_count"])
    h["auth_rate"] = np.where(h["txn_count"] > 0, h["auth_count"] / h["txn_count"].clip(lower=1), np.nan)
    h["avg_ticket"] = np.where(h["txn_count"] > 0, h["total_value"] / h["txn_count"].clip(lower=1), np.nan)
    if "is_anomaly" in h:
        h["is_anomaly"] = h["is_anomaly"].astype(int)
    if "anomaly_type" in h:
        h["anomaly_type"] = h["anomaly_type"].astype(str)

    span = h.groupby("merchant_id")["timestamp"].agg(lambda s: (s.max() - s.min()).days + 1)
    short = span[span < 14]
    if len(short):
        warnings.append(f"{len(short)} merchant(s) have less than 14 days of history "
                        f"({', '.join(map(str, short.index[:5]))}); their seasonal baselines will be unreliable.")
    report = dict(shape=shape, mapped={k: v for k, v in m.items() if v}, warnings=warnings,
                  merchants=h["merchant_id"].nunique(), hours=len(h),
                  start=h["timestamp"].min(), end=h["timestamp"].max())
    return h.sort_values(["merchant_id", "timestamp"]).reset_index(drop=True), report
