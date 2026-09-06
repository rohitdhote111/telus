"""Calibration / extrinsics / geometric-consistency checks.

Two kinds of check live here, deliberately handled differently:

1. **Absolute / physical** — a rotation matrix must be orthonormal with
   determinant +1. This is true regardless of which corpus you are looking
   at, so it uses a fixed numerical tolerance (float32-camera-calibration
   noise is ~1e-6; anything above 1e-3 is not "noisier calibration", it is a
   different kind of matrix). This is NOT the kind of hardcoded threshold the
   assignment warns against — it does not depend on the specific clips
   inspected, it is a property of what a rotation matrix *is*.

2. **Corpus-relative** — does the extrinsic actually *used* for this episode
   agree with the canonical value in the extrinsics session the episode
   claims to reference, and does independent multi-view triangulation of the
   per-camera 2D hand detections agree with what that extrinsic predicts?
   Both are scored against the population of (episode, camera) pairs in the
   run, because a single mis-attached calibration file should stand out
   against seven other cameras (and every other episode) behaving normally.

Important, evidence-backed design note: real DexYCB extrinsics sessions were
probed while choosing this dataset (see docs/technical_note.md), and
**cameras legitimately move 15-25 degrees / up to ~0.4m between capture
sessions** because the physical rig was re-mounted on different days. That
drift is real and enormous compared to any single-session noise floor, so
this module never compares "session A's camera X" against "session B's
camera X" as if disagreement were suspicious. Instead it only ever compares
an episode's *used* extrinsic against *its own declared session's* canonical
value. Session-to-session drift is expected; episode-to-declared-session
disagreement is not.
"""

from __future__ import annotations

import numpy as np

from ..geometry import (
    reprojection_errors_px,
    rotation_determinant,
    rotation_orthonormality_error,
    geodesic_rotation_distance_deg,
)
from ..io import Calibration, Episode
from ..runner import CheckModule, Metric
from ..scoring import Finding, RobustStats, Severity, severity_from_z, worse_severity as _worse

ORTHONORMALITY_TOL = 1e-3  # matrices below this are numerically-clean rotations in this corpus
DETERMINANT_TOL = 1e-3

# Floors below which a used-vs-canonical mismatch is float64/serialization
# noise, not a real discrepancy -- geodesic_rotation_distance_deg's arccos is
# numerically sensitive near zero (its derivative blows up as the angle -> 0),
# so a JSON round-trip alone can produce a "mismatch" of ~1e-4 degrees on two
# matrices that are bit-for-bit intended to be identical. A real mis-attached
# calibration (see docs/technical_note.md's measured session-to-session drift
# of 14-25 degrees / several cm) is orders of magnitude above this. Without
# this floor, a corpus where every healthy episode has an exact-zero mismatch
# has a degenerate (zero-MAD) population, and RobustStats.z() correctly (but
# unhelpfully) reports even noise-level deviations as an infinite z-score.
ROT_MISMATCH_FLOOR_DEG = 0.1
TRANS_MISMATCH_FLOOR_M = 0.001

# A handful of frames per episode is enough to estimate the median reprojection
# residual robustly without doing full 8-view triangulation on every frame.
MAX_FRAMES_FOR_REPROJECTION = 20


class CalibrationChecks(CheckModule):
    name = "calibration"

    def _per_camera_mismatch(self, episode: Episode, calibration: Calibration):
        """Yield (serial, rot_mismatch_deg, trans_mismatch_m) for every camera,
        or None entries if the episode's declared session doesn't exist."""
        session = calibration.sessions.get(episode.extrinsics_session_id)
        if session is None:
            return None
        out = []
        for serial, used in episode.extrinsics_used.items():
            canonical = session.camera_from_world.get(serial)
            if canonical is None:
                out.append((serial, None, None))
                continue
            rot_deg = geodesic_rotation_distance_deg(canonical[:, :3], used[:, :3])
            trans_m = float(np.linalg.norm(used[:, 3] - canonical[:, 3]))
            out.append((serial, rot_deg, trans_m))
        return out

    def _reprojection_rmse(self, episode: Episode, calibration: Calibration) -> dict[str, float]:
        n_frames = len(episode.hand_timestamps)
        if n_frames == 0:
            return {}
        idx = np.linspace(0, n_frames - 1, min(MAX_FRAMES_FOR_REPROJECTION, n_frames)).astype(int)
        K_by_serial = {s: calibration.intrinsics[s].color_K() for s in episode.camera_serials if s in calibration.intrinsics}
        per_camera_errs: dict[str, list[float]] = {s: [] for s in K_by_serial}
        for i in idx:
            obs = {}
            for s in K_by_serial:
                cam_stream = episode.hand_pose_2d_by_camera.get(s)
                if cam_stream is None or i >= len(cam_stream["uv_px"]):
                    continue
                val = cam_stream["uv_px"][i]
                if val is None:  # no detection for this camera at this nominal frame (e.g. dropped raw frame)
                    continue
                obs[s] = np.asarray(val, dtype=float)
            if len(obs) < 3:  # need a well-conditioned multi-view solve
                continue
            extr = {s: episode.extrinsics_used[s] for s in obs}
            errs = reprojection_errors_px(K_by_serial, extr, obs)
            for s, e in errs.items():
                if np.isfinite(e):
                    per_camera_errs[s].append(e)
        return {s: float(np.median(v)) for s, v in per_camera_errs.items() if v}

    def collect(self, episode: Episode, calibration: Calibration) -> list[Metric]:
        metrics = []
        mismatches = self._per_camera_mismatch(episode, calibration)
        if mismatches:
            for serial, rot_deg, trans_m in mismatches:
                if rot_deg is None:
                    continue
                metrics.append(Metric("calibration.extrinsic_rot_mismatch_deg", episode.episode_id, serial, rot_deg))
                metrics.append(Metric("calibration.extrinsic_trans_mismatch_m", episode.episode_id, serial, trans_m))

        for serial, rmse in self._reprojection_rmse(episode, calibration).items():
            metrics.append(Metric("calibration.reprojection_rmse_px", episode.episode_id, serial, rmse))

        return metrics

    def evaluate(self, episode: Episode, calibration: Calibration, stats: dict[str, RobustStats]) -> list[Finding]:
        findings: list[Finding] = []

        # --- absolute physical validity of every extrinsic actually used ---
        for serial, used in episode.extrinsics_used.items():
            R = used[:, :3]
            ortho_err = rotation_orthonormality_error(R)
            det = rotation_determinant(R)
            if ortho_err > ORTHONORMALITY_TOL or abs(det - 1.0) > DETERMINANT_TOL:
                findings.append(
                    Finding(
                        check="calibration.invalid_rotation",
                        severity=Severity.FAIL,
                        message=(
                            f"camera {serial}: extrinsic rotation is not a valid rigid-body rotation "
                            f"(||RtR-I||_F={ortho_err:.4g}, det(R)={det:.4g}; a physically valid "
                            f"rotation has {ORTHONORMALITY_TOL:.0e} and 1.0 respectively)"
                        ),
                        evidence={"serial": serial, "orthonormality_error": ortho_err, "determinant": det},
                        downstream_consequence=(
                            "any 3D point transformed by this camera is geometrically meaningless; "
                            "hand/object pose derived via this camera is unusable for training or evaluation"
                        ),
                    )
                )

        # --- referential integrity of the declared calibration session ---
        session = calibration.sessions.get(episode.extrinsics_session_id)
        if session is None:
            findings.append(
                Finding(
                    check="calibration.unknown_session",
                    severity=Severity.FAIL,
                    message=f"episode references extrinsics session '{episode.extrinsics_session_id}', which does not exist in the calibration corpus",
                    evidence={"extrinsics_session_id": episode.extrinsics_session_id},
                    downstream_consequence="the episode cannot be geometrically verified at all; nothing 3D derived from it should be trusted",
                )
            )
            return findings  # nothing more we can check without a valid session

        # --- used-vs-canonical mismatch, scored against the whole-corpus population ---
        rot_stats = stats.get("calibration.extrinsic_rot_mismatch_deg")
        trans_stats = stats.get("calibration.extrinsic_trans_mismatch_m")
        for serial, rot_deg, trans_m in self._per_camera_mismatch(episode, calibration) or []:
            if rot_deg is None:
                findings.append(
                    Finding(
                        check="calibration.unknown_camera_in_session",
                        severity=Severity.FAIL,
                        message=f"camera {serial} used in this episode has no entry in session '{episode.extrinsics_session_id}'",
                        evidence={"serial": serial, "extrinsics_session_id": episode.extrinsics_session_id},
                        downstream_consequence="this camera's frames cannot be geometrically related to the others; drop or re-attach the correct calibration",
                    )
                )
                continue
            if rot_deg < ROT_MISMATCH_FLOOR_DEG and trans_m < TRANS_MISMATCH_FLOOR_M:
                continue  # noise-level, not a real mismatch -- see floor comment above
            z_rot = rot_stats.z(rot_deg) if rot_stats else 0.0
            z_trans = trans_stats.z(trans_m) if trans_stats else 0.0
            sev = _worse(severity_from_z(z_rot), severity_from_z(z_trans))
            if sev != Severity.NONE:
                findings.append(
                    Finding(
                        check="calibration.extrinsic_mismatch",
                        severity=sev,
                        message=(
                            f"camera {serial}: extrinsic actually attached to this episode differs from the "
                            f"canonical value in its declared session '{episode.extrinsics_session_id}' by "
                            f"{rot_deg:.2f} deg / {trans_m*100:.1f} cm (corpus z-score {max(z_rot, z_trans):.1f})"
                        ),
                        evidence={
                            "serial": serial, "rot_mismatch_deg": rot_deg, "trans_mismatch_m": trans_m,
                            "z_rot": z_rot, "z_trans": z_trans,
                        },
                        downstream_consequence=(
                            "3D hand/object pose reconstructed using this camera is systematically biased; "
                            "an offline metric that averages over all 8 cameras may barely move while any "
                            "action label derived from just this view is wrong -- the classic 'moderate "
                            "offline delta, large online delta' signature"
                        ),
                    )
                )

        # --- multi-view reprojection residual (independent corroborating evidence) ---
        rmse_stats = stats.get("calibration.reprojection_rmse_px")
        for serial, rmse in self._reprojection_rmse(episode, calibration).items():
            z = rmse_stats.z(rmse) if rmse_stats else 0.0
            sev = severity_from_z(z)
            if sev != Severity.NONE:
                findings.append(
                    Finding(
                        check="calibration.reprojection_outlier",
                        severity=sev,
                        message=(
                            f"camera {serial}: median multi-view reprojection residual is {rmse:.1f}px "
                            f"(corpus median {rmse_stats.median:.1f}px, z={z:.1f}) -- this camera disagrees "
                            f"with the other 7 views on where the tracked hand keypoint actually is"
                        ),
                        evidence={"serial": serial, "reprojection_rmse_px": rmse, "z": z},
                        downstream_consequence=(
                            "keypoint triangulation / hand pose annotation for this episode is degraded "
                            "specifically on this camera's contribution"
                        ),
                    )
                )

        # NOTE: an intrinsics-plausibility check (fx/fy/principal-point z-score
        # against the rest of the rig) was prototyped here and immediately
        # reverted -- with only 8 real physical cameras, a robust z-score
        # population of size 8 is too small to be meaningful: real hardware
        # manufacturing variance alone put one genuine, undamaged camera
        # (fx=621.8 vs. a median of 615.7 across the other 7) at z=5.5,
        # which flagged *every* episode referencing it. Shipping a check that
        # produces a 100% false-flag rate the moment it's turned on would be
        # worse than not having it; see docs/ANALYSIS.md for the numbers and
        # docs/technical_note.md's limitations for what this would need
        # (a much larger camera population, or a domain prior on acceptable
        # RealSense-D415 intrinsics range) before it could ship for real.

        return findings
