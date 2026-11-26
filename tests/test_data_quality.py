"""Automated 12-metric data quality & D4 anti-leakage regression gate."""

from __future__ import annotations

from training.data_audit import run_data_audit


def test_data_quality_scorecard_and_zero_d4_leakage():
    report = run_data_audit(proc_count_per_family=20, base_seed=5000)
    assert report["stats"]["d4_collisions"] == [], f"D4 leakage detected: {report['stats']['d4_collisions']}"
    for name, score in report["scores"].items():
        assert score >= 9.5, f"Metric {name!r} scored {score}/10 (< 9.5): stats={report['stats']}"
    assert report["overall_score"] >= 9.5, f"Overall score {report['overall_score']}/10 (< 9.5)"
