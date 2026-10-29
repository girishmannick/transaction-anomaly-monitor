"""
pipeline.py
One function the website, the command-line monitor and the tests all use:
raw data  ->  adapter.prepare  ->  YOUR detect_portfolio  ->  incidents with causes and actions.
"""
from .adapter import prepare
from .anomaly_detection import detect_portfolio, DetectorConfig
from .explain import add_baselines, build_incidents


def run(raw_df, cfg=None):
    cfg = cfg or DetectorConfig()
    hourly, report = prepare(raw_df)
    result = detect_portfolio(hourly, cfg)
    scored = add_baselines(result.scored)
    incidents = build_incidents(scored)
    return scored, incidents, report
