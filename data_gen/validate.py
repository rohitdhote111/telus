#!/usr/bin/env python3
"""Compare episode_qc's output against the fixture ground truth written by
inject_faults.py, and report precision/recall per fault type plus the
healthy-episode false-flag rate. This script is validation-only tooling: the
analysis tool itself (episode_qc) never reads the ground truth fixture.

Usage:
    python data_gen/validate.py --reports outputs/build --fixture data/fixtures/build_ground_truth.json --out outputs/build/validation_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Which check-prefix (and, where relevant, which specific evidence field/value)
# is expected to fire for each injected fault type. This mapping is used only
# by this validation script -- it is intentionally NOT consulted by the
# analysis tool, which has no notion of "which fault type is present."
EXPECTED = {
    "extrinsic_swap": {"prefix": "calibration.", "entity_field": None},
    "timestamp_drift": {"prefix": "sync.", "entity_field": "stream", "entity_from_details": "camera"},
    "frame_drop": {"prefix": "continuity.", "entity_field": "serial", "entity_from_details": "camera"},
    "pose_teleport": {"prefix": "trajectory.", "entity_field": None},
    "metadata_mismatch": {"prefix": "metadata.", "entity_field": None},
    "imu_desync": {"prefix": "sync.", "entity_field": "stream", "entity_from_details": None, "entity_literal": "imu"},
}


def load_report(reports_dir: Path, episode_id: str) -> dict:
    path = reports_dir / "episodes" / f"{episode_id}.json"
    with open(path) as f:
        return json.load(f)


def matches_expected(findings: list[dict], fault_type: str, details: dict) -> bool:
    spec = EXPECTED[fault_type]
    prefix = spec["prefix"]
    candidates = [f for f in findings if f["check"].startswith(prefix)]
    if not candidates:
        return False
    entity_field = spec.get("entity_field")
    if entity_field is None:
        return True
    expected_entity = spec.get("entity_literal") or details.get(spec.get("entity_from_details"))
    if expected_entity is None:
        return True
    return any(str(c["evidence"].get(entity_field)) == str(expected_entity) for c in candidates)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", required=True, type=Path, help="the --out directory passed to episode_qc.cli")
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.fixture) as f:
        ground_truth = json.load(f)

    rows = []
    for episode_id, injected in ground_truth.items():
        report = load_report(args.reports, episode_id)
        findings = report["findings"]
        recommendation = report["recommendation"]
        if injected is None:
            rows.append({
                "episode_id": episode_id, "fault_type": "healthy", "severity": "",
                "recommendation": recommendation, "flagged_any": recommendation != "PASS",
                "detected_specific_check": None,
            })
        else:
            detected = matches_expected(findings, injected["type"], injected["details"])
            rows.append({
                "episode_id": episode_id, "fault_type": injected["type"], "severity": injected["severity"],
                "recommendation": recommendation, "flagged_any": recommendation != "PASS",
                "detected_specific_check": detected,
            })

    df = pd.DataFrame(rows)

    healthy = df[df.fault_type == "healthy"]
    faulted = df[df.fault_type != "healthy"]

    lines = ["# Validation report", "", f"Source reports: `{args.reports}`  |  Fixture: `{args.fixture}`", ""]

    lines += [
        "## Healthy-episode false-flag rate",
        "",
        f"{healthy.flagged_any.sum()}/{len(healthy)} healthy (unfaulted) episodes were flagged REVIEW or FAIL "
        f"({healthy.flagged_any.mean()*100:.1f}%). At a z>=3 REVIEW threshold this is expected to be nonzero -- "
        "see docs/ANALYSIS.md for the statistical reasoning -- but should stay well under, say, 25%.",
        "",
    ]

    lines += ["## Per-fault-type detection (recall)", "", "| fault_type | severity | n | flagged_any | correct_check_fired |", "|---|---|---|---|---|"]
    for (ft, sev), grp in faulted.groupby(["fault_type", "severity"]):
        lines.append(
            f"| {ft} | {sev} | {len(grp)} | {grp.flagged_any.mean()*100:.0f}% | {grp.detected_specific_check.mean()*100:.0f}% |"
        )

    overall_recall_any = faulted.flagged_any.mean()
    overall_recall_specific = faulted.detected_specific_check.mean()
    lines += [
        "",
        f"Overall: {overall_recall_any*100:.1f}% of faulted episodes were flagged REVIEW/FAIL at all; "
        f"{overall_recall_specific*100:.1f}% were flagged by the *specific* check module expected for that fault type "
        "(i.e. not just 'something looked wrong' but 'the right diagnosis').",
        "",
    ]

    lines += ["## Raw per-episode results", "", df.to_markdown(index=False), ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"wrote {args.out}")
    print(f"healthy false-flag rate: {healthy.flagged_any.mean()*100:.1f}%  |  "
          f"faulted recall(any): {overall_recall_any*100:.1f}%  |  faulted recall(specific): {overall_recall_specific*100:.1f}%")


if __name__ == "__main__":
    main()
