"""Turn a dict of EpisodeReport into the files the assignment asks for:
a per-episode JSON (full evidence), and a corpus-level CSV + markdown
summary (health assessment, evidence, severity, recommendation for every
episode)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .scoring import EpisodeReport, severity_rank


def write_episode_json(report: EpisodeReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.episode_id}.json"
    payload = {
        "episode_id": report.episode_id,
        "recommendation": report.recommendation,
        "overall_severity": report.overall.value,
        "findings": [
            {
                "check": f.check,
                "severity": f.severity.value,
                "message": f.message,
                "evidence": f.evidence,
                "downstream_consequence": f.downstream_consequence,
            }
            for f in report.findings
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def summary_dataframe(reports: dict[str, EpisodeReport]) -> pd.DataFrame:
    rows = []
    for episode_id, report in sorted(reports.items()):
        flagged = report.flagged_findings()
        rows.append(
            {
                "episode_id": episode_id,
                "recommendation": report.recommendation,
                "n_findings": len(flagged),
                "checks_flagged": ";".join(sorted({f.check for f in flagged})),
                "worst_check": max(flagged, key=lambda f: severity_rank(f.severity)).check if flagged else "",
            }
        )
    return pd.DataFrame(rows)


def write_summary(reports: dict[str, EpisodeReport], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = summary_dataframe(reports)
    df.to_csv(out_dir / "summary.csv", index=False)

    counts = df["recommendation"].value_counts().to_dict()
    lines = [
        "# Episode QC summary",
        "",
        f"Episodes analyzed: {len(df)}",
        f"PASS: {counts.get('PASS', 0)}  REVIEW: {counts.get('REVIEW', 0)}  FAIL: {counts.get('FAIL', 0)}",
        "",
        "| episode_id | recommendation | n_findings | checks_flagged |",
        "|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(f"| {row['episode_id']} | {row['recommendation']} | {row['n_findings']} | {row['checks_flagged']} |")
    with open(out_dir / "summary.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
