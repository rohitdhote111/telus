"""Episode/task metadata sanity checks.

Two things are checked:
1. Referential integrity of the subject / MANO-shape id against the real
   calibration corpus (a hard, corpus-independent fact -- an id either
   exists or it doesn't).
2. Whether the episode's declared duration is arithmetically consistent
   with its declared frame count and nominal frame rate. Real-world
   metadata is never bit-exact here (duration is usually recorded from a
   wall clock, frame count from a counter, so a small rounding-level
   mismatch is normal) -- so this is scored against the corpus population
   of relative errors, not a fixed epsilon.
"""

from __future__ import annotations

from ..io import Calibration, Episode
from ..runner import CheckModule, Metric
from ..scoring import Finding, RobustStats, Severity, severity_from_z


class MetadataChecks(CheckModule):
    name = "metadata"

    def collect(self, episode: Episode, calibration: Calibration) -> list[Metric]:
        expected_duration = episode.declared_frame_count / episode.fps_nominal
        rel_err = abs(episode.declared_duration_s - expected_duration) / max(expected_duration, 1e-6)
        return [Metric("metadata.duration_relative_error", episode.episode_id, "episode", rel_err)]

    def evaluate(self, episode: Episode, calibration: Calibration, stats: dict[str, RobustStats]) -> list[Finding]:
        findings = []

        if episode.mano_calib_id not in calibration.mano_shapes:
            findings.append(
                Finding(
                    check="metadata.unknown_mano_calib",
                    severity=Severity.FAIL,
                    message=f"episode references MANO calibration id '{episode.mano_calib_id}', which does not exist in the calibration corpus",
                    evidence={"mano_calib_id": episode.mano_calib_id},
                    downstream_consequence="hand shape for this subject cannot be recovered; any hand-mesh/contact reasoning built on MANO is invalid",
                )
            )

        unknown_cams = [s for s in episode.camera_serials if s not in calibration.intrinsics]
        if unknown_cams:
            findings.append(
                Finding(
                    check="metadata.unknown_camera_serial",
                    severity=Severity.FAIL,
                    message=f"episode references camera serial(s) with no intrinsics in the calibration corpus: {unknown_cams}",
                    evidence={"unknown_serials": unknown_cams},
                    downstream_consequence="frames from this camera cannot be undistorted or projected at all",
                )
            )

        expected_duration = episode.declared_frame_count / episode.fps_nominal
        rel_err = abs(episode.declared_duration_s - expected_duration) / max(expected_duration, 1e-6)
        dur_stats = stats.get("metadata.duration_relative_error")
        z = dur_stats.z(rel_err) if dur_stats else 0.0
        sev = severity_from_z(z)
        if sev != Severity.NONE:
            findings.append(
                Finding(
                    check="metadata.duration_frame_count_mismatch",
                    severity=sev,
                    message=(
                        f"declared duration {episode.declared_duration_s:.2f}s is inconsistent with "
                        f"declared frame count / fps ({expected_duration:.2f}s expected); relative error "
                        f"{rel_err*100:.1f}% (corpus z={z:.1f})"
                    ),
                    evidence={
                        "declared_duration_s": episode.declared_duration_s,
                        "expected_duration_s": expected_duration,
                        "relative_error": rel_err, "z": z,
                    },
                    downstream_consequence=(
                        "any code that slices this episode by declared duration (rather than actual frame "
                        "timestamps) will misalign with the real data; suggests the metadata was generated "
                        "or edited independently of the actual capture"
                    ),
                )
            )
        return findings
