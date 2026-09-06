"""Cross-stream synchronization checks.

Every stream in an episode (each camera's color feed, the IMU, and the 3D
hand-pose/annotation stream) *claims* to sample a shared nominal clock. For
each stream we fit its actual timestamps against the frame/sample index it
claims to be reporting (accounting for any frames already known to be
missing, via ``frame_present`` -- so this check is independent of the
continuity check even though both can be triggered by a mangled capture) and
extract two numbers: a start-of-episode **offset** (seconds) and a **rate
error** (parts-per-million deviation from the nominal sample rate).

Both numbers are pooled and scored with a robust z-score -- but *separately
per stream role* (a color camera vs. the IMU vs. the 3D hand-pose annotation
stream), not into one global population. This matters because the three
roles have genuinely different noise floors even when nothing is wrong: a
~30-frame-per-second camera stream fits a much noisier slope from the same
per-sample jitter than a ~200Hz IMU does, simply because it has fewer, more
widely-spaced samples over the same duration. Pooling everything together
lets the noisier role's spread dominate the population MAD and produces
either too many false reviews on the noisy role or too little sensitivity on
the quiet one; a naive fix of just raising the z-threshold trades one problem
for the other instead of fixing it. Stratifying by role is the standard
answer, is still learned entirely from this run's own corpus (not a
hardcoded constant), and was adopted after the first end-to-end validation
run measured a 35-42% false-review rate on healthy episodes, all traced to
this exact pooling mistake -- see docs/ANALYSIS.md for the before/after.
"""

from __future__ import annotations

import numpy as np

from ..io import Calibration, Episode
from ..runner import CheckModule, Metric
from ..scoring import Finding, RobustStats, Severity, severity_from_z


def _fit_offset_and_rate(expected_s: np.ndarray, actual_s: np.ndarray) -> tuple[float, float]:
    if len(expected_s) < 2:
        return 0.0, 0.0
    m, c = np.polyfit(expected_s, actual_s, 1)
    rate_ppm = (m - 1.0) * 1e6
    return float(c), float(rate_ppm)


# Below these, a fitted offset/rate is indistinguishable from float64/least-
# squares solver noise (see calibration_checks.py's identical floor pattern
# for the same underlying reason: a degenerate, near-zero-MAD population
# turns any nonzero noise into an infinite z-score). Both floors sit roughly
# two orders of magnitude below the smallest injected fault magnitude (50ms
# offset, ~10,000ppm drift -- see data_gen/inject_faults.py), so genuine
# faults are never at risk of being floored away.
OFFSET_FLOOR_S = 0.001
RATE_FLOOR_PPM = 200.0


def _role(stream_name: str, episode: Episode) -> str:
    """Which noise-floor population this stream belongs to."""
    if stream_name in episode.camera_serials:
        return "camera"
    return stream_name  # "imu" / "hand_pose_annotation" are each their own role


class SyncChecks(CheckModule):
    name = "sync"

    def _streams(self, episode: Episode):
        """Yield (stream_name, expected_time_s, actual_time_s) for every
        stream in the episode."""
        for serial in episode.camera_serials:
            stream = episode.streams.get(serial)
            if not stream:
                continue
            present = stream.get("frame_present", [True] * len(stream["color_timestamps_s"]))
            true_idx = np.flatnonzero(np.asarray(present, dtype=bool))
            actual = np.asarray(stream["color_timestamps_s"], dtype=float)
            if len(true_idx) != len(actual):
                # inconsistent bookkeeping -- continuity_checks reports this;
                # fall back to a plain index so sync can still say *something*.
                true_idx = np.arange(len(actual))
            expected = true_idx / episode.fps_nominal
            yield serial, expected, actual

        imu_ts = np.asarray(episode.imu["timestamps_s"], dtype=float)
        if len(imu_ts) > 1:
            # Use the sensor's *declared* nominal rate, not one re-estimated
            # from these same timestamps -- comparing against a rate derived
            # from the data under test would partially absorb any real drift
            # into the "expected" grid and mask it.
            imu_rate = float(episode.imu.get("nominal_rate_hz", 1.0 / float(np.median(np.diff(imu_ts)))))
            expected = np.arange(len(imu_ts)) / imu_rate
            yield "imu", expected, imu_ts

        pose_ts = episode.hand_timestamps
        if len(pose_ts) > 1:
            expected = np.arange(len(pose_ts)) / episode.fps_nominal
            yield "hand_pose_annotation", expected, pose_ts

    def collect(self, episode: Episode, calibration: Calibration) -> list[Metric]:
        metrics = []
        for stream_name, expected, actual in self._streams(episode):
            role = _role(stream_name, episode)
            offset_s, rate_ppm = _fit_offset_and_rate(expected, actual)
            metrics.append(Metric(f"sync.stream_offset_s::{role}", episode.episode_id, stream_name, offset_s))
            metrics.append(Metric(f"sync.stream_rate_ppm::{role}", episode.episode_id, stream_name, rate_ppm))
        return metrics

    def evaluate(self, episode: Episode, calibration: Calibration, stats: dict[str, RobustStats]) -> list[Finding]:
        findings = []

        for stream_name, expected, actual in self._streams(episode):
            if len(expected) < 2:
                continue
            # Monotonicity is absolute: a timestamp stream must not go backwards.
            if np.any(np.diff(actual) <= 0):
                findings.append(
                    Finding(
                        check="sync.non_monotonic_timestamps",
                        severity=Severity.FAIL,
                        message=f"stream '{stream_name}' has non-monotonic or duplicate timestamps",
                        evidence={"stream": stream_name},
                        downstream_consequence="frame ordering for this stream cannot be trusted; any temporal alignment against other sensors is undefined",
                    )
                )

            role = _role(stream_name, episode)
            offset_stats = stats.get(f"sync.stream_offset_s::{role}")
            rate_stats = stats.get(f"sync.stream_rate_ppm::{role}")
            offset_s, rate_ppm = _fit_offset_and_rate(expected, actual)
            if abs(offset_s) < OFFSET_FLOOR_S and abs(rate_ppm) < RATE_FLOOR_PPM:
                continue  # noise-level, not a real mismatch -- see floor comment above
            z_offset = offset_stats.z(offset_s) if offset_stats else 0.0
            z_rate = rate_stats.z(rate_ppm) if rate_stats else 0.0
            sev = severity_from_z(max(abs(z_offset), abs(z_rate)))
            if sev != Severity.NONE:
                findings.append(
                    Finding(
                        check="sync.stream_clock_mismatch",
                        severity=sev,
                        message=(
                            f"stream '{stream_name}': fitted clock offset {offset_s*1000:.1f}ms "
                            f"(corpus z={z_offset:.1f}), rate error {rate_ppm:.0f}ppm (corpus z={z_rate:.1f}) "
                            f"relative to the episode's nominal clock"
                        ),
                        evidence={
                            "stream": stream_name, "offset_s": offset_s, "rate_ppm": rate_ppm,
                            "z_offset": z_offset, "z_rate": z_rate,
                        },
                        downstream_consequence=(
                            "any cross-modal fusion (e.g. matching an action to the visual frame it was "
                            "actually observed at) is misaligned by roughly the offset/drift magnitude -- "
                            "for a VLA this shows up as the model conditioning on the wrong observation for "
                            "a given action label"
                        ),
                    )
                )
        return findings
