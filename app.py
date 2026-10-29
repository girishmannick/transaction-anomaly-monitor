"""
Transaction Anomaly Monitor: website for the work-placement anomaly-detection project.

Upload a file or connect a live feed. The site runs the project's detector
(engine/anomaly_detection.py) and explains each incident: what happened,
the likely cause and what to do.

Run locally:   streamlit run app.py
"""
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from engine import run, DetectorConfig
from engine.evaluation import point_metrics, event_metrics, per_type_recall
from engine.explain import DECLINE_MEANING
from engine.live import full_stream, inject, INJECTABLE, MERCHANTS

st.set_page_config(page_title="Transaction Anomaly Monitor", page_icon="📈", layout="wide")

INK, BLUE, ORANGE, RED, GREY = "#1f2933", "#2a78d6", "#eb6834", "#e34948", "#9aa3ad"
LEVEL_COLOR = {"Critical": "#b42318", "High": "#d9480f", "Medium": "#b7791f", "Low": "#4b5563"}
LEVEL_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "⚪"}
SAMPLE = "sample_data/synthetic_transactions.csv"
RAW_EXAMPLE = "sample_data/example_raw_payments.csv"

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; color:white;}
.card {border:1px solid #e5e7eb; border-radius:12px; padding:14px 18px; background:#fff; margin-bottom:10px;}
.muted {color:#6b7280; font-size:0.86rem;}
.cause {padding:6px 0; border-bottom:1px solid #f1f1f1;}
.conf {font-size:0.72rem; font-weight:700; padding:1px 8px; border-radius:6px; margin-right:8px; border:1px solid #d1d5db; color:#374151;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data + analysis (cached)
@st.cache_data(show_spinner=False)
def read_upload(name, data):
    n = name.lower()
    if n.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if n.endswith(".json"):
        return pd.read_json(io.BytesIO(data))
    if n.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def analyse(raw, cfg_json):
    return run(raw, DetectorConfig(**json.loads(cfg_json)))


def settings():
    with st.sidebar.expander("⚙️ Detection settings", expanded=False):
        z = st.slider("Alert threshold (robust z-score)", 2.5, 6.0, 4.0, 0.25,
                      help="Lower = more sensitive (more alerts). 4.0 is the value used in the project.")
        votes = st.radio("Detectors that must agree", [1, 2, 3], index=1, horizontal=True)
        auth = st.slider("Approval-collapse threshold", 2.0, 6.0, 3.5, 0.25)
    return dict(z_threshold=z, roll_threshold=z, ensemble_min_votes=votes, auth_drop_z=auth)


# ------------------------------------------------------------------ charts
def _style(fig, h):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="#eef0f2"); fig.update_yaxes(gridcolor="#eef0f2")
    return fig


def incident_chart(sc, r, pad=48):
    g = sc[sc.merchant_id == r.merchant_id]
    w = g[(g.timestamp >= r.start - pd.Timedelta(hours=pad)) & (g.timestamp <= r.end + pd.Timedelta(hours=pad))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.6, 0.4],
                        subplot_titles=("Payments per hour", "Approval rate"))
    fig.add_vrect(x0=r.start, x1=r.end, fillcolor=ORANGE, opacity=0.13, line_width=0, row="all", col=1)
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.baseline_volume, name="Normal (seasonal baseline)",
                             line=dict(color=BLUE, width=1.5, dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.txn_count, name="Actual", line=dict(color=INK, width=2)), 1, 1)
    a = w[w.is_flagged == 1]
    fig.add_trace(go.Scatter(x=a.timestamp, y=a.txn_count, mode="markers", name="Alarm",
                             marker=dict(color=RED, size=9, line=dict(color="white", width=1.5))), 1, 1)
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.baseline_auth, showlegend=False,
                             line=dict(color=BLUE, width=1.5, dash="dash")), 2, 1)
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.auth_rate, showlegend=False, line=dict(color=INK, width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=a.timestamp, y=a.auth_rate, mode="markers", showlegend=False,
                             marker=dict(color=RED, size=9, line=dict(color="white", width=1.5))), 2, 1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _style(fig, 430)


def evidence_charts(sc, r):
    g = sc[sc.merchant_id == r.merchant_id]
    win = g[(g.timestamp >= r.start) & (g.timestamp < r.end)]
    base = g[g.is_flagged == 0]
    figs = []
    tk_now = win.total_value.sum() / max(win.txn_count.sum(), 1)
    tk_base = base.total_value.sum() / max(base.txn_count.sum(), 1)
    au_now = win.auth_count.sum() / max(win.txn_count.sum(), 1)
    au_base = base.auth_count.sum() / max(base.txn_count.sum(), 1)
    f = make_subplots(rows=1, cols=2, subplot_titles=("Average payment (€)", "Approval rate"))
    f.add_trace(go.Bar(x=["Normal", "Incident"], y=[tk_base, tk_now], marker_color=[GREY, RED], showlegend=False), 1, 1)
    f.add_trace(go.Bar(x=["Normal", "Incident"], y=[au_base, au_now], marker_color=[GREY, RED], showlegend=False), 1, 2)
    f.update_yaxes(tickformat=".0%", row=1, col=2)
    figs.append(_style(f, 280))
    sch = [s for s in ["visa", "mastercard", "amex"] if f"{s}_txns" in g]
    if len(sch) >= 2:
        now = [win[f"{s}_auth"].sum() / max(win[f"{s}_txns"].sum(), 1) for s in sch]
        nrm = [base[f"{s}_auth"].sum() / max(base[f"{s}_txns"].sum(), 1) for s in sch]
        f = go.Figure([go.Bar(name="Normal", x=[s.capitalize() for s in sch], y=nrm, marker_color=GREY),
                       go.Bar(name="Incident", x=[s.capitalize() for s in sch], y=now, marker_color=RED)])
        f.update_layout(title="Approval rate by card network", barmode="group", yaxis_tickformat=".0%")
        figs.append(_style(f, 280))
    dec = [c for c in g.columns if c.startswith("d") and c[1:3].isdigit()]
    if dec and win[dec].sum().sum() > 0:
        n = win[dec].sum() / win[dec].sum().sum(); b = base[dec].sum() / max(base[dec].sum().sum(), 1)
        lab = [f"{c[1:3]} {c[4:].replace('_', ' ')}" for c in dec]
        f = go.Figure([go.Bar(name="Normal", y=lab, x=b.values, orientation="h", marker_color=GREY),
                       go.Bar(name="Incident", y=lab, x=n.values, orientation="h", marker_color=RED)])
        f.update_layout(title="Decline reasons", barmode="group", xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"))
        figs.append(_style(f, 280))
    return figs


def merchant_chart(sc, mid):
    g = sc[sc.merchant_id == mid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.timestamp, y=g.baseline_volume, name="Normal", line=dict(color=BLUE, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=g.timestamp, y=g.txn_count, name="Actual", line=dict(color=INK, width=1.2)))
    a = g[g.is_flagged == 1]
    fig.add_trace(go.Scatter(x=a.timestamp, y=a.txn_count, mode="markers", name="Alarm", text=a.predicted_type,
                             marker=dict(color=RED, size=7, line=dict(color="white", width=1)),
                             hovertemplate="%{x}<br>%{y} payments<br>%{text}<extra></extra>"))
    fig.update_yaxes(title="Payments / hour")
    return _style(fig, 340)


# ------------------------------------------------------------------ incident views
def badge(level):
    return f"<span class='badge' style='background:{LEVEL_COLOR[level]}'>{level}</span>"


def incident_table(inc):
    t = inc.assign(Level=inc.level.map(lambda l: f"{LEVEL_ICON[l]} {l}"),
                   Start=inc.start.dt.strftime("%d %b %Y %H:%M"), Duration=inc.hours.astype(str) + " h",
                   Change=inc.vol_change.map(lambda x: f"{x:+.0%}"),
                   Approval=[f"{a:.0%} (normal {b:.0%})" for a, b in zip(inc.auth_now, inc.auth_base)],
                   Impact=inc.est_impact_eur.map(lambda x: f"€{x:,.0f}" if x > 0 else "–"))
    return t[["incident_id", "Level", "merchant_name", "title", "Start", "Duration", "Change", "Approval",
              "likely_cause", "Impact"]].rename(columns={"incident_id": "ID", "merchant_name": "Merchant",
                                                         "title": "What", "likely_cause": "Most likely cause",
                                                         "Impact": "Value affected"})


def incident_detail(sc, r, key=""):
    st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap'>"
                f"<div><b style='font-size:1.15rem'>{r.title}</b> &nbsp;{badge(r.level)}<br>"
                f"<span class='muted'>{r.incident_id} · {r.merchant_name} ({r.merchant_id}) · {r.start:%d %b %Y %H:%M} → "
                f"{r.end:%d %b %H:%M} · {r.hours} h · {r.payments:,} payments · detector label: {r.detector_type}</span></div>"
                f"<div style='text-align:right'><span class='muted'>Notify</span><br><b>{r.team}</b></div></div></div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown("**What happened**")
        st.write(r.what)
        if r.est_impact_eur > 0:
            st.markdown(f"**Estimated payment value affected:** €{r.est_impact_eur:,.0f}")
        st.markdown("**Most likely causes**")
        st.markdown("".join(f"<div class='cause'><span class='conf'>{c}</span>{t}</div>" for t, c in r.causes),
                    unsafe_allow_html=True)
    with c2:
        st.markdown("**What to do (recommended actions)**")
        st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(r.actions, 1)))
    st.plotly_chart(incident_chart(sc, r), width="stretch", key=f"ic_{key}_{r.incident_id}")
    figs = evidence_charts(sc, r)
    cols = st.columns(len(figs))
    for j, (c, f) in enumerate(zip(cols, figs)):
        c.plotly_chart(f, width="stretch", key=f"ev_{key}_{r.incident_id}_{j}")
    if r.top_decline in DECLINE_MEANING:
        st.caption(f"Most over-represented decline code: **{r.top_decline}**: {DECLINE_MEANING[r.top_decline]}")


def export(inc):
    out = inc.copy()
    out["causes"] = out.causes.map(lambda cs: " | ".join(f"[{c}] {t}" for t, c in cs))
    out["actions"] = out.actions.map(lambda a: " | ".join(a))
    out["correlated"] = out.correlated.map(", ".join)
    return out.to_csv(index=False)


# ------------------------------------------------------------------ batch view (upload / sample)
def batch_view(raw, label):
    cfg = settings()
    with st.spinner("Running the anomaly detector…"):
        try:
            sc, inc, rep = analyse(raw, json.dumps(cfg))
        except ValueError as e:
            st.error(str(e)); format_help(); return
    st.caption(f"{label} · {rep['shape']} data · {rep['merchants']} merchants · "
               f"{rep['start']:%d %b %Y} – {rep['end']:%d %b %Y} · {rep['hours']:,} merchant-hours")
    for w in rep["warnings"]:
        st.warning(w)
    levels = st.sidebar.multiselect("Show incident levels", list(LEVEL_COLOR), ["Critical", "High"])
    view = inc[inc.level.isin(levels)] if len(inc) else inc

    k = st.columns(5)
    k[0].metric("Merchants", rep["merchants"])
    k[1].metric("Hours flagged", int(sc.is_flagged.sum()))
    k[2].metric("Incidents shown", len(view))
    k[3].metric("Critical / High", int(inc.level.isin(["Critical", "High"]).sum()) if len(inc) else 0)
    k[4].metric("Value affected (est.)", f"€{inc.est_impact_eur.sum():,.0f}" if len(inc) else "€0")

    tabs = st.tabs(["🚨 Incidents", "🏪 Merchant explorer", "📊 Evaluation", "📄 Data & export"])
    with tabs[0]:
        if not len(view):
            st.success("No incidents at the selected levels.")
        else:
            f1, f2 = st.columns(2)
            types = f1.multiselect("Type", sorted(view.title.unique()), sorted(view.title.unique()))
            merch = f2.multiselect("Merchant", sorted(view.merchant_name.unique()), [])
            v = view[view.title.isin(types)]
            if merch:
                v = v[v.merchant_name.isin(merch)]
            st.dataframe(incident_table(v), width="stretch", hide_index=True, height=min(430, 38 + 35 * len(v)))
            if len(v):
                pick = st.selectbox("Open an incident to see the cause and what to do",
                                    v.incident_id + " · " + v.merchant_name + " · " + v.title + " · " +
                                    v.start.dt.strftime("%d %b %H:%M"))
                incident_detail(sc, v[v.incident_id == pick.split(" · ")[0]].iloc[0], "b")
    with tabs[1]:
        names = sc.drop_duplicates("merchant_id").set_index("merchant_id")["merchant_name"]
        mid = st.selectbox("Merchant", names.index, format_func=lambda m: f"{names[m]} ({m})")
        g = sc[sc.merchant_id == mid]
        c = st.columns(4)
        c[0].metric("Payments", f"{int(g.txn_count.sum()):,}")
        c[1].metric("Approval rate", f"{g.auth_count.sum() / max(g.txn_count.sum(), 1):.1%}")
        c[2].metric("Hours flagged", int(g.is_flagged.sum()))
        c[3].metric("Incidents", int((inc.merchant_id == mid).sum()) if len(inc) else 0)
        st.plotly_chart(merchant_chart(sc, mid), width="stretch")
    with tabs[2]:
        if "is_anomaly" not in sc or sc.is_anomaly.sum() == 0:
            st.info("Add an `is_anomaly` column (1 = known incident hour) and optionally `anomaly_type` "
                    "to measure how accurate the detector is on your data.")
        else:
            pm, em = point_metrics(sc), event_metrics(sc)
            c = st.columns(5)
            c[0].metric("Incidents detected", f"{em['events_detected']} / {em['events_total']}")
            c[1].metric("Mean delay", f"{em['mean_detection_delay_hours']:.2f} h")
            c[2].metric("Recall", f"{pm['recall']:.3f}")
            c[3].metric("Precision", f"{pm['precision']:.3f}")
            c[4].metric("F1", f"{pm['f1']:.3f}")
            st.markdown(f"Confusion matrix: TN **{pm['TN']:,}** · FP **{pm['FP']}** · FN **{pm['FN']}** · TP **{pm['TP']}**")
            if "anomaly_type" in sc and (sc.anomaly_type.isin(["SPIKE", "DROP", "AUTH_DROP"])).any():
                st.markdown("**Recall by anomaly type**")
                st.dataframe(per_type_recall(sc), hide_index=True)
    with tabs[3]:
        c1, c2 = st.columns(2)
        if len(inc):
            c1.download_button("⬇️ Incidents with causes and actions (CSV)", export(inc), "incidents.csv", "text/csv", width="stretch")
        cols = ["timestamp", "merchant_id", "merchant_name", "txn_count", "auth_rate", "baseline_volume", "score_z",
                "score_roll", "score_iforest", "score_auth", "votes", "is_flagged", "predicted_type", "severity"]
        c2.download_button("⬇️ Hour-by-hour detector scores (CSV)", sc[cols].to_csv(index=False), "scores.csv", "text/csv", width="stretch")
        st.markdown("**How your columns were read**")
        st.json(rep["mapped"])
        st.dataframe(raw.head(200), width="stretch")


def format_help():
    with st.expander("📘 What file can I upload?", expanded=True):
        st.markdown("""
**Option A: hourly data (one row per merchant per hour), the same format as the project**

| column | needed? | example |
|---|---|---|
| `timestamp` | ✅ | 2024-02-14 09:00 |
| `merchant_id` | ✅ | M001 |
| `txn_count` | ✅ | 412 |
| `auth_count` (approved) or `declined` | recommended | 389 |
| `total_value` (amount) | recommended | 10450.20 |
| `merchant_name` | optional | Aurora Apparel |
| `is_anomaly`, `anomaly_type` | optional, turns on the Evaluation tab | 1, SPIKE |
| `visa_txns`, `visa_auth`, `mastercard_txns` … | optional, adds card-network diagnosis | |
| `d05_…`, `d51_…`, `d91_…` decline-code counts | optional, adds decline-reason diagnosis | |

**Option B: raw payments (one row per payment)**: `timestamp`, `merchant_id`, `amount`, `status`
(approved / declined) and optionally `scheme` and `response_code`. They are grouped into hours automatically.

Column names don't have to match exactly (e.g. `Created_At`, `Merchant`, `Amount`, `Status` also work).
Use **at least 3–4 weeks** of history so the detector can learn what "normal" looks like.
""")


# ------------------------------------------------------------------ live view
def live_view(mode):
    ss = st.session_state
    sb = st.sidebar
    cfg = DetectorConfig(iforest_estimators=150)
    if mode == "sim":
        if "stream" not in ss:
            ss.stream = full_stream()
            ss.cursor = pd.Timestamp("2024-02-12 08:00")
            ss.playing = False
            ss.log, ss.notified = [], []
        sb.markdown("**Live feed controls**")
        c = sb.columns(2)
        def _toggle(): ss.playing = not ss.playing
        def _step(): ss.cursor += pd.Timedelta(hours=1)
        c[0].button("⏸ Pause" if ss.playing else "▶️ Play", width="stretch", on_click=_toggle)
        c[1].button("⏭ +1 hour", width="stretch", on_click=_step)
        speed = sb.select_slider("Speed (hours per tick)", [1, 2, 3, 6], 1)
        sb.markdown("**Create an incident now (demo)**")
        im = sb.selectbox("Merchant", list(MERCHANTS), format_func=lambda m: MERCHANTS[m])
        ik = sb.selectbox("Incident", list(INJECTABLE))
        ih = sb.slider("Duration (hours)", 2, 8, 4)
        if sb.button("💥 Inject", width="stretch"):
            ss.stream = inject(ss.stream, im, INJECTABLE[ik], ss.cursor + pd.Timedelta(hours=1), ih)
            sb.success(f"{ik} starts for {MERCHANTS[im]} in the next hour. Press +1 hour or Play.")
        if sb.button("↺ Reset feed", width="stretch"):
            for k in ("stream", "cursor", "playing", "log", "notified"):
                ss.pop(k, None)
            st.rerun()
        refresh = 4
    else:
        url = sb.text_input("Feed URL (returns CSV or JSON)", ss.get("feed_url", ""),
                            placeholder="https://example.com/merchant-hourly.csv")
        ss.feed_url = url
        refresh = sb.slider("Check every (seconds)", 15, 600, 60)
        ss.playing = bool(url)
        ss.setdefault("log", []); ss.setdefault("notified", [])
        if not url:
            st.info("Paste the web address of any CSV/JSON that is updated with the latest hourly payments "
                    "(for example a Google Sheet published as CSV, or your company's API). "
                    "The site re-reads it on the interval you choose and analyses the most recent 4 weeks.")
            format_help()
            return
    ss.webhook = sb.text_input("Send alerts to Slack/Teams webhook (optional)", ss.get("webhook", ""), type="password")

    @st.fragment(run_every=refresh if ss.get("playing") else None)
    def tick():
        if mode == "sim":
            if ss.playing:
                ss.cursor += pd.Timedelta(hours=speed)
            s = ss.stream
            window = s[(s.timestamp > ss.cursor - pd.Timedelta(days=28)) & (s.timestamp <= ss.cursor)]
        else:
            try:
                u = ss.feed_url
                window = pd.read_json(u) if u.lower().split("?")[0].endswith(".json") else pd.read_csv(u)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the feed: {e}"); return
        try:
            sc, inc, _ = run(window, cfg)
        except ValueError as e:
            st.error(str(e)); return
        last = sc.timestamp.max()
        active = inc[inc.end > last] if len(inc) else inc
        active = active[active.level.isin(["Critical", "High", "Medium"])] if len(active) else active
        recent = inc[(inc.end > last - pd.Timedelta(hours=72)) & inc.level.isin(["Critical", "High"])] if len(inc) else inc

        h = st.columns([1.6, 1, 1, 1])
        h[0].markdown(f"#### {'🟢 LIVE' if ss.get('playing') else '⏸ Paused'}<br>"
                      f"<span style='font-weight:500'>{last:%a %d %b %Y · %H:00}</span>", unsafe_allow_html=True)
        h[1].metric("Active incidents", len(active))
        h[2].metric("Critical/High, last 72 h", len(recent))
        h[3].metric("Payments last hour", f"{int(sc[sc.timestamp == last].txn_count.sum()):,}")

        for _, r in active.iterrows():
            key = [r.merchant_id, r.type, str(r.start)]
            if key not in ss.notified:
                ss.notified.append(key)
                ss.log.insert(0, dict(time=f"{last:%d %b %H:%M}", merchant=r.merchant_name, incident=r.title,
                                      level=r.level, likely_cause=r.likely_cause))
                if r.level in ("Critical", "High"):
                    st.toast(f"{LEVEL_ICON[r.level]} {r.title}: {r.merchant_name}")
                if ss.get("webhook"):
                    try:
                        import requests
                        requests.post(ss.webhook, timeout=5, json={"text": f"{LEVEL_ICON[r.level]} {r.title}: {r.merchant_name}\n"
                                                                          f"{r.what}\nLikely cause: {r.likely_cause}\nNotify: {r.team}"})
                    except Exception:  # noqa: BLE001
                        pass
        if len(active):
            for i, (_, r) in enumerate(active.iterrows()):
                with st.expander(f"{LEVEL_ICON[r.level]} {r.title}: {r.merchant_name} (since {r.start:%H:%M})", expanded=i == 0):
                    incident_detail(sc, r, f"live{i}")
        else:
            st.success("All merchants are within their normal range.")

        st.markdown("#### Merchant health (last 48 hours)")
        cols = st.columns(min(4, sc.merchant_id.nunique()))
        for j, mid in enumerate(sc.merchant_id.unique()):
            g = sc[(sc.merchant_id == mid) & (sc.timestamp > last - pd.Timedelta(hours=48))]
            bad = len(active) and (active.merchant_id == mid).any()
            fig = go.Figure([go.Scatter(x=g.timestamp, y=g.baseline_volume, line=dict(color=BLUE, width=1, dash="dash"), hoverinfo="skip"),
                             go.Scatter(x=g.timestamp, y=g.txn_count, line=dict(color=INK, width=1.6)),
                             go.Scatter(x=g[g.is_flagged == 1].timestamp, y=g[g.is_flagged == 1].txn_count, mode="markers",
                                        marker=dict(color=RED, size=6))])
            fig.update_layout(height=150, margin=dict(l=0, r=0, t=26, b=0), showlegend=False, plot_bgcolor="white",
                              title=dict(text=f"{'🔴' if bad else '🟢'} {g.merchant_name.iloc[0]}", font=dict(size=13)),
                              xaxis=dict(showticklabels=False, showgrid=False), yaxis=dict(gridcolor="#eef0f2"))
            cols[j % len(cols)].plotly_chart(fig, width="stretch", key=f"hg_{mid}", config={"displayModeBar": False})
        if ss.log:
            st.markdown("#### Alert log")
            st.dataframe(pd.DataFrame(ss.log), hide_index=True, width="stretch")
    tick()


# ------------------------------------------------------------------ page
st.title("📈 Transaction Anomaly Monitor")
src = st.sidebar.radio("Data source", ["📁 Upload a file", "🧪 Project sample data", "📡 Live feed (simulated)", "🌐 Live feed (from a URL)"])
if src == "📁 Upload a file":
    up = st.file_uploader("Upload payment data (CSV, Excel, JSON or Parquet)", type=["csv", "xlsx", "xls", "json", "parquet"])
    if up is None:
        format_help()
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Example: hourly file", open(SAMPLE, "rb").read(), "example_hourly.csv", "text/csv")
        c2.download_button("⬇️ Example: one row per payment", open(RAW_EXAMPLE, "rb").read(), "example_raw_payments.csv", "text/csv")
    else:
        try:
            raw = read_upload(up.name, up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
        else:
            batch_view(raw, f"File: {up.name}")
elif src == "🧪 Project sample data":
    batch_view(read_upload("s.csv", open(SAMPLE, "rb").read()),
               "Project dataset (5 fictitious merchants, Jan–Apr 2024, 11 labelled incidents)")
elif src == "📡 Live feed (simulated)":
    live_view("sim")
else:
    live_view("url")

st.sidebar.markdown("---")
st.sidebar.caption("Work-placement project · Girish Mannick · BSc (Hons) Data Science & AI, Open University of Mauritius. "
                   "Demo uses fictitious data only.")
