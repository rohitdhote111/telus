"""Hand-pose trajectory quality checks.

Manipulation demonstrations move smoothly (a human hand cannot teleport).
For each episode we compute the frame-to-frame speed of the annotated 3D
hand keypoint and look for a frame whose speed is an outlier *relative to
that same episode's own speed distribution* -- deliberately self-relative
rather than corpus-relative, because "does this jump look normal for this
episode's own motion" is the right question, not "does this episode move at
the same overall pace as every other episode" (different tasks have
genuinely different speed profiles; that difference is not a defect -- see
the "don't assume every suspicious signal is a defect" instruction in the
assignment). The peak z-score is *additionally* pooled across the corpus as
a secondary, corroborating signal reported alongside the primary finding.
"""

from __future__ import annotations

import numpy as np

from ..io import Calibration, Episode
from ..runner import CheckModule, Metric
from ..scoring import Finding, RobustStats, Severity, robust_stats, severity_from_z


def _frame_speeds(episode: Episode) -> np.ndarray:
    xyz = episode.hand_keypoints
    ts = episode.hand_timestamps
    if len(xyz) < 3:
        return np.zeros(0)
    dt = np.diff(ts)
    dt = np.where(dt <= 1e-6, np.nan, dt)
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
    return speed


class TrajectoryChecks(CheckModule):
    name = "trajectory"

    def collect(self, episode: Episode, calibration: Calibration) -> list[Metric]:
        speeds = _frame_speeds(episode)
        if len(speeds) < 5:
            return []
        within_ep = robust_stats(list(speeds))
        peak_z = max(abs(within_ep.z(s)) for s in speeds if np.isfinite(s))
        return [Metric("trajectory.episode_peak_speed_z", episode.episode_id, "hand", float(peak_z))]

    def evaluate(self, episode: Episode, calibration: Calibration, stats: dict[str, RobustStats]) -> list[Finding]:
        speeds = _frame_speeds(episode)
        if len(speeds) < 5:
            return []
        ts = episode.hand_timestamps
        within_ep = robust_stats(list(speeds))

        findings = []
        peak_z_pool = stats.get("trajectory.episode_peak_speed_z")
        for i, s in enumerate(speeds):
            if not np.isfinite(s):
                continue
            z = within_ep.z(s)
            sev = severity_from_z(z, review_z=4.0, fail_z=8.0)  # self-relative pop is small (frames in 1 episode); use a wider band
            if sev != Severity.NONE:
                corpus_note = ""
                if peak_z_pool and peak_z_pool.n > 3:
                    corpus_z = peak_z_pool.z(z)
                    corpus_note = f"; this episode's peak jump is itself a corpus z-score of {corpus_z:.1f} vs other episodes' peak jumps"
                findings.append(
                    Finding(
                        check="trajectory.implausible_jump",
                        severity=sev,
                        message=(
                            f"hand keypoint speed at t={ts[i+1]:.2f}s is {s:.2f} m/s vs this episode's own "
                            f"median speed {within_ep.median:.2f} m/s (within-episode z={z:.1f}){corpus_note}"
                        ),
                        evidence={
                            "frame_index": i + 1, "timestamp_s": float(ts[i + 1]),
                            "speed_mps": float(s), "episode_median_speed_mps": within_ep.median,
                            "within_episode_z": z,
                        },
                        downstream_consequence=(
                            "a physically implausible jump in the labeled hand trajectory produces an "
                            "action-label spike that a VLA will try to imitate; at inference this is either "
                            "unreachable (policy attempts an impossible velocity) or the model learns a "
                            "smoothed/averaged version that undershoots the true motion"
                        ),
                    )
                )
        return findings
