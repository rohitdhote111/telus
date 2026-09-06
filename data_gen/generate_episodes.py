#!/usr/bin/env python3
"""Generate a batch of schema-faithful, mostly-healthy episodes.

Every episode references *real* DexYCB calibration (a real extrinsics
session, real per-camera intrinsics, a real subject's MANO shape id) fetched
from data/real/calibration. What's synthetic is the per-episode timing,
3D hand trajectory, per-camera 2D detections and IMU stream — the raw
capture that would normally supply these is 12-119GB and impractical to pull
into this exercise. See docs/technical_note.md for the full disclosure.

Usage:
    python data_gen/generate_episodes.py --out data/synthetic/build --n 40 --seed 1
    python data_gen/generate_episodes.py --out data/synthetic/holdout --n 20 --seed 999
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from common import (
    FPS_NOMINAL,
    IMU_RATE_HZ,
    TASK_LABELS,
    WORKSPACE_X,
    WORKSPACE_Y,
    WORKSPACE_Z,
    get_calibration,
    rng_for,
)


def sample_smooth_trajectory(rng: np.random.Generator, frame_times: np.ndarray) -> np.ndarray:
    """A smooth reach/transport hand path through 3-5 random waypoints in the
    shared tabletop workspace, sampled at the given frame times."""
    n_waypoints = rng.integers(3, 6)
    wp_t = np.linspace(frame_times[0], frame_times[-1], n_waypoints)
    wp_xyz = np.stack(
        [
            rng.uniform(*WORKSPACE_X, size=n_waypoints),
            rng.uniform(*WORKSPACE_Y, size=n_waypoints),
            rng.uniform(*WORKSPACE_Z, size=n_waypoints),
        ],
        axis=1,
    )
    spline = CubicSpline(wp_t, wp_xyz, bc_type="clamped", axis=0)
    return spline(frame_times)


def project(K: np.ndarray, extrinsic_3x4: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    R, t = extrinsic_3x4[:, :3], extrinsic_3x4[:, 3]
    cam = points_world @ R.T + t
    z = np.clip(cam[:, 2], 1e-6, None)
    uv1 = (cam / z[:, None]) @ K.T
    return uv1[:, :2]


def generate_one_episode(calibration, seed: int, batch: str, idx: int) -> dict:
    rng = rng_for(seed, batch, idx)

    subject_dirs = sorted(calibration.mano_shapes.keys())
    session_ids = sorted(calibration.sessions.keys())
    camera_serials = calibration.camera_serials()

    mano_calib_id = subject_dirs[rng.integers(len(subject_dirs))]
    subject_id = mano_calib_id.split("_subject-")[1].split("_")[0]
    subject_id = f"subject-{subject_id}"
    extrinsics_session_id = session_ids[rng.integers(len(session_ids))]
    session = calibration.sessions[extrinsics_session_id]
    task_label = TASK_LABELS[rng.integers(len(TASK_LABELS))]

    duration_s = float(rng.uniform(2.5, 5.0))
    frame_count = int(round(duration_s * FPS_NOMINAL))
    frame_times = np.arange(frame_count) / FPS_NOMINAL

    world_xyz = sample_smooth_trajectory(rng, frame_times)
    world_xyz_annotated = world_xyz + rng.normal(scale=0.002, size=world_xyz.shape)  # 2mm annotation noise

    streams = {}
    hand_pose_2d_by_camera = {}
    for serial in camera_serials:
        color_ts = frame_times + rng.normal(scale=0.0015, size=frame_count)  # ~1.5ms driver jitter
        streams[serial] = {
            "color_timestamps_s": color_ts.tolist(),
            "depth_timestamps_s": (color_ts + rng.normal(scale=0.0003, size=frame_count)).tolist(),
            "frame_present": [True] * frame_count,
        }
        K = calibration.intrinsics[serial].color_K()
        extr = session.camera_from_world[serial]
        uv = project(K, extr, world_xyz) + rng.normal(scale=1.5, size=(frame_count, 2))  # ~1.5px detector noise
        hand_pose_2d_by_camera[serial] = {
            "timestamps_s": color_ts.tolist(),
            "uv_px": uv.tolist(),
        }

    n_imu = int(round(duration_s * IMU_RATE_HZ))
    imu_ts = np.arange(n_imu) / IMU_RATE_HZ + rng.normal(scale=0.0005, size=n_imu)
    imu_ts.sort()  # jitter must not violate monotonicity in the healthy case
    world_at_imu = CubicSpline(frame_times, world_xyz, bc_type="clamped", axis=0)(np.clip(imu_ts, frame_times[0], frame_times[-1]))
    accel = np.gradient(np.gradient(world_at_imu, imu_ts, axis=0), imu_ts, axis=0)
    accel += np.array([0.0, 0.0, 9.81])  # illustrative gravity/bias term
    accel += rng.normal(scale=0.05, size=accel.shape)

    extrinsics_used = {s: session.camera_from_world[s].tolist() for s in camera_serials}

    episode = {
        "episode_id": f"{batch}_{idx:04d}",
        "subject_id": subject_id,
        "mano_calib_id": mano_calib_id,
        "mano_side": "right",
        "extrinsics_session_id": extrinsics_session_id,
        "camera_serials": camera_serials,
        "fps_nominal": FPS_NOMINAL,
        "task_label": task_label,
        "declared_frame_count": frame_count,
        "declared_duration_s": duration_s,
        "streams": streams,
        "hand_pose_world": {
            "timestamps_s": frame_times.tolist(),
            "keypoints_m": world_xyz_annotated.tolist(),
        },
        "hand_pose_2d_by_camera": hand_pose_2d_by_camera,
        "imu": {
            "nominal_rate_hz": IMU_RATE_HZ,
            "timestamps_s": imu_ts.tolist(),
            "accel_mss": accel.tolist(),
        },
        "extrinsics_used": extrinsics_used,
        "fault_injected": None,  # filled in later by inject_faults.py for the faulted subset
    }
    return episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=40, help="number of episodes to generate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", default=None, help="batch name embedded in episode_id (default: out dir name)")
    args = ap.parse_args()

    batch = args.batch or args.out.name
    args.out.mkdir(parents=True, exist_ok=True)
    calibration = get_calibration()

    for idx in range(args.n):
        ep = generate_one_episode(calibration, args.seed, batch, idx)
        with open(args.out / f"{ep['episode_id']}.json", "w") as f:
            json.dump(ep, f)

    print(f"wrote {args.n} episodes to {args.out} (batch='{batch}', seed={args.seed})")


if __name__ == "__main__":
    main()
