"""
explain.py
Turns the detector's flagged hours (is_flagged / predicted_type) into incidents an analyst
can act on: what happened, the most likely causes (using evidence in the data) and the
recommended response.
"""
import numpy as np
import pandas as pd

TITLES = {
    "CARD_TESTING": "Suspected card-testing attack",
    "SPIKE": "Transaction volume spike",
    "DROP": "Transaction volume drop / possible outage",
    "AUTH_DROP": "Authorisation (approval-rate) collapse",
}

DECLINE_MEANING = {
    "05": "Do not honour: the issuer refused without a specific reason (often issuer risk rules)",
    "51": "Insufficient funds: customers' cards lack funds (usually benign)",
    "14": "Invalid card number: typical of card testing or bad card data",
    "54": "Expired card: typical of card testing with old card lists",
    "91": "Issuer or switch unavailable: the bank/scheme could not be reached (technical)",
    "96": "System malfunction: technical failure at the issuer, scheme or acquirer",
    "57": "Transaction not permitted: card/merchant category restriction",
    "59": "Suspected fraud: issuer fraud systems are blocking",
    "65": "Exceeds frequency limit: velocity rules triggered",
    "03": "Invalid merchant: configuration problem",
}

PLAYBOOK = {
    "CARD_TESTING": dict(team="Fraud & Risk monitoring", actions=[
        "Apply velocity limits per card, IP address, device and email for this merchant immediately.",
        "Force 3-D Secure (or add a CAPTCHA at checkout) on low-value payments until the attack stops.",
        "Block the offending card BIN ranges, IP addresses or user agents seen in the raw transactions.",
        "Refund or void small approved test payments to limit chargebacks and scheme fines.",
        "Tell the merchant's account manager and technical contact about the attack.",
        "Watch fraud and chargeback ratios for this merchant over the next 30-60 days."]),
    "SPIKE": dict(team="Account Management / Risk monitoring", actions=[
        "Ask the merchant's account manager whether a promotion, launch or campaign is running.",
        "If confirmed, record it in the commercial calendar so similar spikes are expected next time.",
        "Check capacity and rate limits so the merchant's traffic is not throttled.",
        "Keep an eye on fraud, refund and chargeback rates for the next few days.",
        "If nobody can explain it (especially at night), treat it as possible fraud."]),
    "DROP": dict(team="Merchant Support / Technical Account Management", actions=[
        "Check the merchant's API error rates (4xx/5xx) and response times.",
        "Check the platform status page and whether other merchants dropped at the same time.",
        "Contact the merchant's technical team: a broken checkout page, expired API key or failed release is likely.",
        "Verify the merchant's API keys, certificates and IP allow-lists are valid and unchanged.",
        "If several merchants are affected together, escalate it as a platform incident.",
        "After recovery, confirm volume is back to normal and record the lost sales."]),
    "AUTH_DROP": dict(team="Payments Operations / Acquiring & Routing", actions=[
        "Check acquirer, card-scheme and issuer status pages and internal incident channels.",
        "If one scheme or bank is affected, re-route traffic through a backup acquirer or connection.",
        "Turn on smart retries for soft declines (codes 91, 96).",
        "Review recent configuration changes for this merchant (MCC, descriptor, 3-D Secure, routing rules).",
        "Tell the merchant their customers' payments are being declined and when it should be fixed.",
        "Escalate to the scheme / acquirer relationship team if it lasts more than one hour."]),
}


def _code_of(col):
    return col[1:3] if isinstance(col, str) and col.startswith("d") and col[1:3].isdigit() else None


def add_baselines(scored):
    """Seasonal baselines (median per merchant, weekday, hour), the same reference the
    robust z-score uses. Used for charts and explanations."""
    s = scored.copy()
    s["hour"] = s["timestamp"].dt.hour
    s["dow"] = s["timestamp"].dt.dayofweek
    g = s.groupby(["merchant_id", "dow", "hour"])
    s["baseline_volume"] = g["txn_count"].transform("median")
    s["baseline_auth"] = g["auth_rate"].transform("median")
    return s


def _level(max_abs_z, auth_score, hours):
    m = max(max_abs_z, auth_score)
    if m >= 10 or (hours >= 6 and m >= 6):
        return "Critical"
    if m >= 6:
        return "High"
    if m >= 4:
        return "Medium"
    return "Low"


def build_incidents(scored, gap_hours=1):
    s = scored if "baseline_volume" in scored else add_baselines(scored)
    dec_cols = [c for c in s.columns if _code_of(c)]
    schemes = [x for x in ["visa", "mastercard", "amex"] if f"{x}_txns" in s.columns]
    groups = []
    for mid, g in s.groupby("merchant_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        hits = g.index[g["is_flagged"] == 1].tolist()
        if not hits:
            continue
        cur = [hits[0]]
        for i in hits[1:]:
            if i - cur[-1] <= gap_hours + 1:
                cur.append(i)
            else:
                groups.append((mid, g, cur)); cur = [i]
        groups.append((mid, g, cur))

    windows = []
    for mid, g, idx in groups:
        types = [t for t in g.loc[idx, "predicted_type"] if t]
        windows.append((mid, g.loc[idx[0], "timestamp"], g.loc[idx[-1], "timestamp"], types))

    rows = []
    for k, (mid, g, idx) in enumerate(groups):
        inc = g.loc[idx[0]:idx[-1]]
        flagged = inc[inc["is_flagged"] == 1]
        types = [t for t in flagged["predicted_type"] if t]
        kind = max(set(types), key=types.count) if types else "SPIKE"
        normal = g[g["is_flagged"] == 0]
        base = normal[normal["hour"].isin(inc["hour"]) & normal["dow"].isin(inc["dow"])]
        if len(base) < 3:
            base = normal
        vol_now = inc["txn_count"].mean()
        vol_base = max(inc["baseline_volume"].mean(), 1e-9)
        vol_chg = vol_now / vol_base - 1
        auth_now = inc["auth_count"].sum() / max(inc["txn_count"].sum(), 1)
        auth_base = base["auth_count"].sum() / max(base["txn_count"].sum(), 1)
        tk_now = inc["total_value"].sum() / max(inc["txn_count"].sum(), 1)
        tk_base = base["total_value"].sum() / max(base["txn_count"].sum(), 1)
        night = bool(len(flagged)) and (flagged["hour"].iloc[0] < 6 or flagged["hour"].iloc[0] >= 23)
        n_pay = int(inc["txn_count"].sum())

        # scheme + decline evidence (only when those optional columns exist)
        worst, gap, srate, orate = "", None, None, None
        if len(schemes) >= 2:
            r = {x: inc[f"{x}_auth"].sum() / max(inc[f"{x}_txns"].sum(), 1) for x in schemes if inc[f"{x}_txns"].sum() >= 5}
            if len(r) >= 2:
                worst = min(r, key=r.get)
                oth = [x for x in r if x != worst]
                orate = sum(inc[f"{x}_auth"].sum() for x in oth) / max(sum(inc[f"{x}_txns"].sum() for x in oth), 1)
                srate, gap = r[worst], orate - r[worst]
        top_code = None
        if dec_cols and inc[dec_cols].sum().sum() > 0:
            now = inc[dec_cols].sum() / inc[dec_cols].sum().sum()
            bas = base[dec_cols].sum() / max(base[dec_cols].sum().sum(), 1)
            top_code = _code_of((now - bas).idxmax())

        # Double-check the direction for volume alarms using the actual volume vs the seasonal baseline
        if kind in ("SPIKE", "DROP"):
            kind = "DROP" if vol_chg < 0 else "SPIKE"

        # Card-testing refinement of a SPIKE: small tickets + falling approvals
        if kind == "SPIKE" and tk_base > 0 and tk_now < 0.6 * tk_base and auth_now < auth_base - 0.10:
            kind = "CARD_TESTING"

        t0, t1 = inc["timestamp"].min(), inc["timestamp"].max()
        corr = []
        if kind == "DROP":
            corr = sorted({w[0] for j, w in enumerate(windows) if j != k and w[0] != mid and "DROP" in w[3]
                           and w[1] <= t1 + pd.Timedelta(hours=1) and w[2] >= t0 - pd.Timedelta(hours=1)})

        causes = []
        if kind == "CARD_TESTING":
            what = (f"Volume rose {abs(vol_chg):.0%} above normal while the average payment fell to "
                    f"€{tk_now:.2f} (normally €{tk_base:.2f}) and approvals dropped to {auth_now:.0%} (normally {auth_base:.0%}).")
            causes.append(("Card-testing attack: criminals checking which stolen cards work with many small payments", "High"))
            if night:
                causes.append(("It happened at night, which is typical of automated bot attacks", "Supporting"))
            if top_code in ("14", "54", "05"):
                causes.append((f"Decline code {top_code} jumped: {DECLINE_MEANING[top_code]}", "Supporting"))
            causes.append(("A broken merchant integration retrying payments in a loop", "Low"))
        elif kind == "SPIKE":
            what = (f"Volume rose {abs(vol_chg):.0%} above normal ({vol_now:.0f} vs ~{vol_base:.0f} payments per hour). "
                    f"Approvals {auth_now:.0%} (normally {auth_base:.0%}), average payment €{tk_now:.2f} (normally €{tk_base:.2f}).")
            healthy = auth_now >= auth_base - 0.05 and (tk_base == 0 or tk_now >= 0.6 * tk_base)
            if night:
                causes.append(("Unexplained night-time spike: possible bot traffic or early card testing, verify before treating as normal", "Medium"))
                causes.append(("Legitimate business event (e.g. a sale launched at midnight or in another time zone)", "Medium"))
            elif healthy:
                causes.append(("Legitimate business event: promotion, flash sale, product launch or media coverage", "High"))
            else:
                causes.append(("Approvals or payment sizes look unusual: possible fraud or bot traffic", "Medium"))
            causes.append(("The merchant moved traffic to us from another payment provider", "Low"))
        elif kind == "DROP":
            what = (f"Volume fell {abs(vol_chg):.0%} below normal ({vol_now:.0f} vs ~{vol_base:.0f} payments per hour) "
                    f"for {len(inc)} hour(s).")
            if corr:
                causes.append((f"Platform or gateway incident: {len(corr)} other merchant(s) dropped at the same time "
                               f"({', '.join(corr[:4])})", "High"))
            if vol_chg <= -0.7:
                causes.append(("Merchant checkout or integration outage (broken page, expired API key, failed release)",
                               "High" if not corr else "Medium"))
            else:
                causes.append(("Partial outage affecting one payment method, country, device or browser", "Medium"))
            if not corr:
                causes.append(("Only this merchant is affected, so the problem is most likely on the merchant's side", "Supporting"))
            if auth_now < auth_base - 0.10:
                causes.append(("Approvals also fell: payments that do arrive are failing too (acquirer problem)", "Medium"))
            causes.append(("Traffic loss on the merchant's side (campaign stopped, website or DNS problem)", "Low"))
        else:  # AUTH_DROP
            what = (f"The approval rate fell to {auth_now:.0%} (normally {auth_base:.0%}) while volume stayed "
                    f"{'normal' if abs(vol_chg) < 0.3 else f'{vol_chg:+.0%} versus normal'}: customers are trying to pay but being declined.")
            if worst and gap is not None and gap > 0.25:
                causes.append((f"Problem with one card network: {worst.capitalize()} approvals {srate:.0%} vs {orate:.0%} for others "
                               f"(issuer/scheme connection or routing fault)", "High"))
            elif worst:
                causes.append(("All card networks affected: acquirer-side failure or merchant configuration change", "High"))
            else:
                causes.append(("Acquirer, card-network or bank (issuer) connection or routing failure", "High"))
            if top_code in ("91", "96"):
                causes.append((f"Decline code {top_code} jumped: {DECLINE_MEANING[top_code]}", "High"))
            elif top_code in ("05", "59"):
                causes.append((f"Decline code {top_code} jumped: bank fraud rules are rejecting this merchant's payments", "Medium"))
            elif top_code == "51":
                causes.append(("Decline code 51 jumped: customers lack funds (often end-of-month timing), probably harmless", "Medium"))
            causes.append(("Recent change in the merchant's settings (category code, descriptor, 3-D Secure, routing)", "Medium"))
            causes.append(("Bank fraud systems blocking this merchant after suspicious activity", "Low"))

        max_z = float(flagged["score_z"].abs().max()) if len(flagged) else 0.0
        max_auth = float(flagged["score_auth"].max()) if len(flagged) else 0.0
        if kind == "AUTH_DROP":
            drop = auth_base - auth_now
            level = ("Critical" if drop >= 0.40 and len(inc) >= 3 else "High" if drop >= 0.25 else
                     "Medium" if drop >= 0.10 else "Low")
            if n_pay < 30:
                level = "Low"
                causes.insert(0, (f"Only {n_pay} payments in this period, so a few declines move the rate a lot: "
                                  "probably random noise, no action unless it repeats", "Medium"))
        else:
            level = _level(max_z, 0.0, len(inc))
        if kind == "CARD_TESTING" and level in ("Low", "Medium"):
            level = "High"
        if kind == "DROP":
            impact = max(0.0, vol_base - vol_now) * len(inc) * tk_base
        elif kind == "AUTH_DROP":
            impact = max(0.0, auth_base - auth_now) * inc["txn_count"].sum() * tk_base
        else:
            impact = 0.0
        rows.append(dict(
            incident_id=f"INC-{k + 1:04d}", merchant_id=mid, merchant_name=str(inc["merchant_name"].iloc[0]),
            start=t0, end=t1 + pd.Timedelta(hours=1), hours=len(inc), flagged_hours=len(flagged),
            type=kind, detector_type=", ".join(sorted(set(types))), title=TITLES[kind], level=level,
            severity=float(flagged["severity"].max()) if len(flagged) else 0.0, max_abs_z=round(max_z, 1),
            what=what, causes=causes, likely_cause=None, actions=PLAYBOOK[kind]["actions"],
            team=PLAYBOOK[kind]["team"], vol_change=vol_chg, auth_now=auth_now, auth_base=auth_base,
            est_impact_eur=round(impact, 0), payments=n_pay, correlated=corr, top_decline=top_code or "", worst_scheme=worst))
    out = pd.DataFrame(rows)
    if len(out):
        out["likely_cause"] = out["causes"].map(lambda c: c[0][0])
        order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        out = out.assign(_o=out["level"].map(order)).sort_values(["_o", "max_abs_z"], ascending=[True, False]) \
                 .drop(columns="_o").reset_index(drop=True)
    return out
