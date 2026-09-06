"""Loaders for the real DexYCB calibration corpus and the (schema-faithful,
partly synthetic) episode files produced by ``data_gen/generate_episodes.py``.

Nothing here is specific to any one episode or calibration session — every
function takes a directory and discovers what is inside it. That is what lets
``episode_qc.cli`` run unmodified on a corpus it has never seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------
# Calibration corpus
# --------------------------------------------------------------------------


@dataclass
class CameraIntrinsics:
    serial: str
    width: int
    height: int
    color_fx: float
    color_fy: float
    color_ppx: float
    color_ppy: float
    depth_fx: float
    depth_fy: float
    depth_ppx: float
    depth_ppy: float
    color_from_depth: np.ndarray  # 3x4, depth-frame -> color-frame (per-unit factory calib)

    def color_K(self) -> np.ndarray:
        return np.array(
            [
                [self.color_fx, 0.0, self.color_ppx],
                [0.0, self.color_fy, self.color_ppy],
                [0.0, 0.0, 1.0],
            ]
        )


@dataclass
class ExtrinsicsSession:
    session_id: str
    # serial -> 3x4 [R|t] mapping *world* (= reference camera frame) -> this camera
    camera_from_world: dict[str, np.ndarray]
    reference_serial: str


@dataclass
class Calibration:
    """The full calibration corpus: one intrinsics set per physical camera,
    many extrinsics *sessions* (the rig was re-mounted/re-calibrated between
    DexYCB capture days), and per-subject MANO hand-shape parameters."""

    intrinsics: dict[str, CameraIntrinsics]
    sessions: dict[str, ExtrinsicsSession]
    mano_shapes: dict[str, np.ndarray]  # calib_id -> 10 betas

    def camera_serials(self) -> list[str]:
        return sorted(self.intrinsics.keys())


def _parse_3x4(values) -> np.ndarray:
    return np.asarray(list(values), dtype=float).reshape(3, 4)


def load_intrinsics(path: Path) -> CameraIntrinsics:
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.UnsafeLoader)
    serial, resolution = path.stem.split("_")
    width, height = (int(v) for v in resolution.split("x"))
    color, depth = y["color"], y["depth"]
    return CameraIntrinsics(
        serial=serial,
        width=width, height=height,
        color_fx=color["fx"], color_fy=color["fy"],
        color_ppx=color["ppx"], color_ppy=color["ppy"],
        depth_fx=depth["fx"], depth_fy=depth["fy"],
        depth_ppx=depth["ppx"], depth_ppy=depth["ppy"],
        color_from_depth=_parse_3x4(y["extrinsics"]),
    )


def load_extrinsics_session(path: Path) -> ExtrinsicsSession:
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.UnsafeLoader)
    raw = y["extrinsics"]
    cams = {k: _parse_3x4(v) for k, v in raw.items() if k != "apriltag"}
    # The reference camera is whichever one is the identity transform; DexYCB
    # is consistent about this but we verify rather than assume.
    ref = None
    for serial, m in cams.items():
        R, t = m[:, :3], m[:, 3]
        if np.allclose(R, np.eye(3), atol=1e-5) and np.allclose(t, 0.0, atol=1e-5):
            ref = serial
            break
    session_id = path.parent.name
    return ExtrinsicsSession(session_id=session_id, camera_from_world=cams, reference_serial=ref)


def load_mano_betas(path: Path) -> np.ndarray:
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.UnsafeLoader)
    return np.asarray(y["betas"], dtype=float)


def load_calibration(calibration_dir: Path) -> Calibration:
    calibration_dir = Path(calibration_dir)
    intrinsics = {}
    for p in sorted((calibration_dir / "intrinsics").glob("*.yml")):
        ci = load_intrinsics(p)
        intrinsics[ci.serial] = ci

    sessions = {}
    for d in sorted(calibration_dir.glob("extrinsics_*")):
        es = load_extrinsics_session(d / "extrinsics.yml")
        sessions[es.session_id] = es

    mano_shapes = {}
    for d in sorted(calibration_dir.glob("mano_*")):
        mano_shapes[d.name] = load_mano_betas(d / "mano.yml")

    return Calibration(intrinsics=intrinsics, sessions=sessions, mano_shapes=mano_shapes)


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------


@dataclass
class Episode:
    episode_id: str
    subject_id: str
    mano_calib_id: str
    extrinsics_session_id: str
    camera_serials: list[str]
    fps_nominal: float
    task_label: str
    declared_frame_count: int
    declared_duration_s: float
    streams: dict  # serial -> {"color_timestamps_s": [...], "frame_present": [...]}
    hand_pose_world: dict  # {"timestamps_s": [...], "keypoints_m": [[x,y,z], ...]}
    hand_pose_2d_by_camera: dict  # serial -> {"timestamps_s": [...], "uv_px": [[u,v], ...]}
    imu: dict  # {"timestamps_s": [...], "accel_mss": [[ax,ay,az], ...]}
    extrinsics_used: dict  # serial -> 3x4 actually attached to this episode
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def hand_keypoints(self) -> np.ndarray:
        return np.asarray(self.hand_pose_world["keypoints_m"], dtype=float)

    @property
    def hand_timestamps(self) -> np.ndarray:
        return np.asarray(self.hand_pose_world["timestamps_s"], dtype=float)


def load_episode(path: Path) -> Episode:
    with open(path) as f:
        d = json.load(f)
    extrinsics_used = {k: np.asarray(v, dtype=float).reshape(3, 4) for k, v in d["extrinsics_used"].items()}
    return Episode(
        episode_id=d["episode_id"],
        subject_id=d["subject_id"],
        mano_calib_id=d["mano_calib_id"],
        extrinsics_session_id=d["extrinsics_session_id"],
        camera_serials=d["camera_serials"],
        fps_nominal=d["fps_nominal"],
        task_label=d["task_label"],
        declared_frame_count=d["declared_frame_count"],
        declared_duration_s=d["declared_duration_s"],
        streams=d["streams"],
        hand_pose_world=d["hand_pose_world"],
        hand_pose_2d_by_camera=d.get("hand_pose_2d_by_camera", {}),
        imu=d["imu"],
        extrinsics_used=extrinsics_used,
        raw=d,
    )


def load_episodes(episodes_dir: Path) -> list[Episode]:
    episodes_dir = Path(episodes_dir)
    out = []
    for p in sorted(episodes_dir.glob("*.json")):
        out.append(load_episode(p))
    return out
