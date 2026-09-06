#!/usr/bin/env python3
"""Deliberately corrupt a fraction of an already-generated episode batch, one
fault type + magnitude per chosen episode, and write the ground truth to a
fixture file. The analysis tool (episode_qc) never reads this fixture -- it
exists purely so we can measure precision/recall afterward.

Usage:
    python data_gen/inject_faults.py --episodes data/synthetic/build --fixture data/fixtures/build_ground_truth.json --seed 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import rng_for

FAULT_TYPES = [
    "extrinsic_swap",
    "timestamp_drift",
    "frame_drop",
    "pose_teleport",
    "metadata_mismatch",
    "imu_desync",
]
SEVERITIES = ["mild", "severe"]

FRACTION_FAULTED = 0.5  # rest of the batch stays as healthy controls


def _rotation_about_random_axis(rng: np.random.Generator, angle_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def fault_extrinsic_swap(d: dict, rng: np.random.Generator, severity: str) -> dict:
    serial = rng.choice(d["camera_serials"])
    angle_deg, trans_m = {"mild": (2.0, 0.01), "severe": (15.0, 0.08)}[severity]
    M = np.asarray(d["extrinsics_used"][serial], dtype=float).reshape(3, 4)
    R_perturb = _rotation_about_random_axis(rng, angle_deg)
    R_new = R_perturb @ M[:, :3]
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    t_new = M[:, 3] + direction * trans_m
    d["extrinsics_used"][serial] = np.concatenate([R_new, t_new[:, None]], axis=1).tolist()
    return {"camera": serial, "angle_deg": angle_deg, "trans_m": trans_m}


def fault_timestamp_drift(d: dict, rng: np.random.Generator, severity: str) -> dict:
    serial = rng.choice(d["camera_serials"])
    drift_end_s = {"mild": 0.03, "severe": 0.15}[severity]
    stream = d["streams"][serial]
    ts = np.asarray(stream["color_timestamps_s"], dtype=float)
    n = len(ts)
    drift = drift_end_s * (np.arange(n) / max(n - 1, 1))
    stream["color_timestamps_s"] = (ts + drift).tolist()
    depth_ts = np.asarray(stream["depth_timestamps_s"], dtype=float)
    stream["depth_timestamps_s"] = (depth_ts + drift).tolist()
    return {"camera": serial, "drift_end_s": drift_end_s}


def fault_frame_drop(d: dict, rng: np.random.Generator, severity: str) -> dict:
    serial = rng.choice(d["camera_serials"])
    stream = d["streams"][serial]
    n = len(stream["frame_present"])
    run_len = {"mild": 3, "severe": 15}[severity]
    run_len = min(run_len, max(n - 4, 1))
    start = int(rng.integers(2, max(n - run_len - 2, 3)))

    present = np.array(stream["frame_present"], dtype=bool)
    present[start:start + run_len] = False
    stream["frame_present"] = present.tolist()

    keep = present  # keep only present frames in the raw timestamp arrays
    ts = np.asarray(stream["color_timestamps_s"], dtype=float)
    depth_ts = np.asarray(stream["depth_timestamps_s"], dtype=float)
    stream["color_timestamps_s"] = ts[keep].tolist()
    stream["depth_timestamps_s"] = depth_ts[keep].tolist()

    # The per-camera 2D hand detections are a downstream product of the raw
    # frames -- a missing raw frame means a missing detection at that same
    # nominal frame index, but every OTHER camera/frame stays fully indexed.
    cam2d = d["hand_pose_2d_by_camera"][serial]["uv_px"]
    for i in range(start, min(start + run_len, len(cam2d))):
        cam2d[i] = None

    return {"camera": serial, "start_frame": start, "run_len": run_len}


def fault_pose_teleport(d: dict, rng: np.random.Generator, severity: str) -> dict:
    xyz = np.asarray(d["hand_pose_world"]["keypoints_m"], dtype=float)
    n = len(xyz)
    frame_idx = int(rng.integers(max(n // 4, 1), max(3 * n // 4, 2)))
    jump_m = {"mild": 0.15, "severe": 0.6}[severity]
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    xyz[frame_idx] = xyz[frame_idx] + direction * jump_m
    d["hand_pose_world"]["keypoints_m"] = xyz.tolist()
    return {"frame_index": frame_idx, "jump_m": jump_m}


def fault_metadata_mismatch(d: dict, rng: np.random.Generator, severity: str) -> dict:
    rel_err = {"mild": 0.15, "severe": 0.60}[severity]
    sign = 1 if rng.random() < 0.5 else -1
    d["declared_duration_s"] = d["declared_duration_s"] * (1 + sign * rel_err)
    return {"relative_error": rel_err, "sign": sign}


def fault_imu_desync(d: dict, rng: np.random.Generator, severity: str) -> dict:
    offset_s = {"mild": 0.05, "severe": 0.25}[severity]
    sign = 1 if rng.random() < 0.5 else -1
    ts = np.asarray(d["imu"]["timestamps_s"], dtype=float)
    d["imu"]["timestamps_s"] = (ts + sign * offset_s).tolist()
    return {"offset_s": sign * offset_s}


FAULT_FNS = {
    "extrinsic_swap": fault_extrinsic_swap,
    "timestamp_drift": fault_timestamp_drift,
    "frame_drop": fault_frame_drop,
    "pose_teleport": fault_pose_teleport,
    "metadata_mismatch": fault_metadata_mismatch,
    "imu_desync": fault_imu_desync,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", required=True, type=Path)
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    files = sorted(args.episodes.glob("*.json"))
    top_rng = np.random.default_rng(args.seed)
    n_faulted = int(round(len(files) * FRACTION_FAULTED))
    faulted_files = top_rng.choice(files, size=n_faulted, replace=False)
    faulted_set = {f.name for f in faulted_files}

    # Evenly cycle through (fault_type, severity) combinations rather than
    # sampling them, so a small batch still exercises every combination.
    combos = [(ft, sv) for ft in FAULT_TYPES for sv in SEVERITIES]

    ground_truth = {}
    combo_i = 0
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if f.name in faulted_set:
            fault_type, severity = combos[combo_i % len(combos)]
            combo_i += 1
            rng = rng_for(args.seed, f.name, fault_type, severity)
            details = FAULT_FNS[fault_type](d, rng, severity)
            d["fault_injected"] = {"type": fault_type, "severity": severity, "details": details}
            ground_truth[d["episode_id"]] = d["fault_injected"]
        else:
            ground_truth[d["episode_id"]] = None
        with open(f, "w") as fh:
            json.dump(d, fh)

    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    with open(args.fixture, "w") as fh:
        json.dump(ground_truth, fh, indent=2)

    n_healthy = len(files) - len(faulted_set)
    print(f"injected faults into {len(faulted_set)}/{len(files)} episodes ({n_healthy} healthy controls)")
    print(f"ground truth written to {args.fixture}")


if __name__ == "__main__":
    main()
