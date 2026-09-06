"""Command-line entry point.

    python -m episode_qc.cli --calibration data/real/calibration --episodes data/synthetic/holdout --out outputs/holdout

No episode ids or per-clip thresholds are hardcoded anywhere in this module
or the checks it calls -- every threshold is derived from whatever corpus is
passed via --episodes at run time (see runner.py / scoring.py docstrings).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checks.calibration_checks import CalibrationChecks
from .checks.continuity_checks import ContinuityChecks
from .checks.metadata_checks import MetadataChecks
from .checks.sync_checks import SyncChecks
from .checks.trajectory_checks import TrajectoryChecks
from .io import load_calibration, load_episodes
from .report import write_episode_json, write_summary
from .runner import run_corpus

ALL_MODULES = [
    CalibrationChecks(),
    SyncChecks(),
    ContinuityChecks(),
    TrajectoryChecks(),
    MetadataChecks(),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True, type=Path, help="path to data/real/calibration")
    ap.add_argument("--episodes", required=True, type=Path, help="directory of episode_*.json files")
    ap.add_argument("--out", required=True, type=Path, help="output directory for reports")
    args = ap.parse_args(argv)

    calibration = load_calibration(args.calibration)
    episodes = load_episodes(args.episodes)
    if not episodes:
        print(f"no episodes found in {args.episodes}", file=sys.stderr)
        return 1

    reports = run_corpus(episodes, calibration, ALL_MODULES)

    per_episode_dir = args.out / "episodes"
    for report in reports.values():
        write_episode_json(report, per_episode_dir)
    write_summary(reports, args.out)

    n_fail = sum(1 for r in reports.values() if r.recommendation == "FAIL")
    n_review = sum(1 for r in reports.values() if r.recommendation == "REVIEW")
    n_pass = len(reports) - n_fail - n_review
    print(f"analyzed {len(reports)} episodes -> PASS={n_pass} REVIEW={n_review} FAIL={n_fail}")
    print(f"reports written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
