"""Shared constants + helpers used by both the generator and the fault
injector, so the two stay in lock-step about the episode schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from episode_qc.io import Calibration, load_calibration  # noqa: E402

CALIBRATION_DIR = REPO_ROOT / "data" / "real" / "calibration"

FPS_NOMINAL = 30.0
IMU_RATE_HZ = 200.0
TASK_LABELS = ["pick_and_place", "pour", "insert_peg", "stack_blocks", "open_drawer"]

# Plausible tabletop workspace in front of the (identity) reference camera,
# metres, in the world/reference-camera frame used by the real extrinsics.
WORKSPACE_X = (-0.30, 0.30)
WORKSPACE_Y = (-0.30, 0.30)
WORKSPACE_Z = (0.40, 0.90)


def get_calibration() -> Calibration:
    return load_calibration(CALIBRATION_DIR)


def rng_for(seed: int, *parts) -> np.random.Generator:
    """Deterministic per-episode RNG: same seed + same episode index always
    reproduces the same episode, independent of generation order."""
    ss = np.random.SeedSequence([seed, *[abs(hash(p)) % (2**32) for p in parts]])
    return np.random.default_rng(ss)
