"""Integration test: generate a tiny corpus (using the real generator +
fault injector, against the real calibration data), run the CLI's check
modules over it, and confirm every injected fault is caught by its expected
check while the untouched healthy episodes come back PASS.

This is deliberately the same machinery used for the full build/holdout
validation in outputs/*/validation_report.md, just at a size small enough to
run in a few seconds as part of the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_gen"))
CALIBRATION_DIR = REPO_ROOT / "data" / "real" / "calibration"

from episode_qc.checks.calibration_checks import CalibrationChecks
from episode_qc.checks.continuity_checks import ContinuityChecks
from episode_qc.checks.metadata_checks import MetadataChecks
from episode_qc.checks.sync_checks import SyncChecks
from episode_qc.checks.trajectory_checks import TrajectoryChecks
from episode_qc.io import load_calibration, load_episode
from episode_qc.runner import run_corpus

MODULES = [CalibrationChecks(), SyncChecks(), ContinuityChecks(), TrajectoryChecks(), MetadataChecks()]

pytestmark = pytest.mark.skipif(
    not CALIBRATION_DIR.exists(),
    reason="real DexYCB calibration corpus not present at data/real/calibration (see docs/technical_note.md)",
)


@pytest.fixture(scope="module")
def calibration():
    return load_calibration(CALIBRATION_DIR)


@pytest.fixture()
def tiny_corpus(tmp_path):
    import generate_episodes
    import inject_faults

    out = tmp_path / "episodes"
    calib = generate_episodes.get_calibration()
    for idx in range(12):
        ep = generate_episodes.generate_one_episode(calib, seed=42, batch="t", idx=idx)
        out.mkdir(parents=True, exist_ok=True)
        import json
        with open(out / f"{ep['episode_id']}.json", "w") as f:
            json.dump(ep, f)

    fixture = tmp_path / "ground_truth.json"
    sys_argv_backup = sys.argv
    sys.argv = ["inject_faults.py", "--episodes", str(out), "--fixture", str(fixture), "--seed", "7"]
    try:
        inject_faults.main()
    finally:
        sys.argv = sys_argv_backup
    return out, fixture


def test_every_fault_type_is_detected(calibration, tiny_corpus):
    import json

    episodes_dir, fixture = tiny_corpus
    with open(fixture) as f:
        ground_truth = json.load(f)

    episodes = [load_episode(p) for p in sorted(episodes_dir.glob("*.json"))]
    reports = run_corpus(episodes, calibration, MODULES)

    prefix_by_fault = {
        "extrinsic_swap": "calibration.",
        "timestamp_drift": "sync.",
        "frame_drop": "continuity.",
        "pose_teleport": "trajectory.",
        "metadata_mismatch": "metadata.",
        "imu_desync": "sync.",
    }

    n_checked = 0
    for episode_id, injected in ground_truth.items():
        report = reports[episode_id]
        if injected is None:
            continue
        n_checked += 1
        prefix = prefix_by_fault[injected["type"]]
        assert any(f.check.startswith(prefix) for f in report.findings), (
            f"{episode_id} had fault {injected['type']} but no '{prefix}*' finding fired "
            f"(findings: {[f.check for f in report.findings]})"
        )
        assert report.recommendation != "PASS"

    assert n_checked > 0  # sanity: the fixture actually contained faulted episodes
