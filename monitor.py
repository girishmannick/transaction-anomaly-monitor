"""
monitor.py: run the detector from the command line (used by the GitHub Action).

    python monitor.py --source sample_data/synthetic_transactions.csv
    python monitor.py --source https://example.com/feed.csv --recent-hours 2

Writes reports/incidents.csv and reports/summary.md. Exit code 0 always; the file
reports/new_alerts.md exists only if there are Critical/High incidents in the recent window.
"""
import argparse
import os

import pandas as pd

from engine import run

LEVEL_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "⚪"}


def read(src):
    s = src.lower().split("?")[0]
    if s.endswith(".json"):
        return pd.read_json(src)
    if s.endswith((".xlsx", ".xls")):
        return pd.read_excel(src)
    return pd.read_csv(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="CSV/JSON/Excel file path or URL")
    ap.add_argument("--recent-hours", type=int, default=0,
                    help="Only alert on incidents still active in the last N hours (0 = report everything)")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scored, inc, rep = run(read(a.source))
    last = scored["timestamp"].max()
    serious = inc[inc["level"].isin(["Critical", "High"])] if len(inc) else inc
    if a.recent_hours and len(serious):
        serious = serious[serious["end"] > last - pd.Timedelta(hours=a.recent_hours)]

    flat = inc.copy()
    if len(flat):
        flat["causes"] = flat["causes"].map(lambda cs: " | ".join(f"[{c}] {t}" for t, c in cs))
        flat["actions"] = flat["actions"].map(" | ".join)
    flat.to_csv(os.path.join(a.out, "incidents.csv"), index=False)

    lines = [f"## Transaction anomaly report",
             f"Data: **{rep['merchants']} merchants**, {rep['start']:%d %b %Y %H:%M} → {last:%d %b %Y %H:%M} "
             f"({rep['hours']:,} merchant-hours). Hours flagged: **{int(scored['is_flagged'].sum())}**. "
             f"Incidents: **{len(inc)}** ({int(inc['level'].isin(['Critical', 'High']).sum()) if len(inc) else 0} Critical/High).",
             ""]
    if len(serious):
        lines += ["| | Merchant | What | Start | Hours | Most likely cause | Notify |", "|---|---|---|---|---|---|---|"]
        for _, r in serious.iterrows():
            lines.append(f"| {LEVEL_ICON[r.level]} {r.level} | {r.merchant_name} | {r.title} | {r.start:%d %b %H:%M} | "
                         f"{r.hours} | {r.likely_cause} | {r.team} |")
        lines += ["", "### What to do"]
        for _, r in serious.head(5).iterrows():
            lines += [f"**{r.incident_id}: {r.title}, {r.merchant_name}**: {r.what}", ""]
            lines += [f"{i}. {x}" for i, x in enumerate(r.actions, 1)] + [""]
    else:
        lines.append("✅ No Critical/High incidents in the selected window.")
    md = "\n".join(lines)
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    if len(serious):
        open(os.path.join(a.out, "new_alerts.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
