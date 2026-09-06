"""Frame continuity / completeness checks.

For every camera stream in an episode we compare the number of frames it
actually has against the episode's declared frame count, and look for the
longest run of consecutive missing frames. Healthy episodes in this corpus
have zero dropped frames on every camera, so the "population" for this
metric is degenerate at zero -- which is exactly why *any* dropped frame is
scored as a hard FAIL rather than graded by a z-score (see
``scoring.RobustStats.z``: a nonzero value against an all-zero population is
an infinite z-score, i.e. "this does not happen in normal operation"). The
exact fraction and run-length are still reported as quantified evidence.
"""

from __future__ import annotations

import numpy as np

from ..io import Calibration, Episode
from ..runner import CheckModule, Metric
from ..scoring import Finding, RobustStats, Severity, severity_from_z


def _longest_false_run(present: np.ndarray) -> int:
    if present.all():
        return 0
    # index of each False; longest run of consecutive indices
    missing = np.flatnonzero(~present)
    if len(missing) == 0:
        return 0
    breaks = np.flatnonzero(np.diff(missing) > 1)
    run_starts = np.concatenate(([0], breaks + 1))
    run_ends = np.concatenate((breaks, [len(missing) - 1]))
    return int(np.max(run_ends - run_starts + 1))


class ContinuityChecks(CheckModule):
    name = "continuity"

    def collect(self, episode: Episode, calibration: Calibration) -> list[Metric]:
        metrics = []
        for serial in episode.camera_serials:
            stream = episode.streams.get(serial)
            if not stream:
                continue
            present = np.asarray(stream.get("frame_present", [True] * episode.declared_frame_count), dtype=bool)
            dropped_fraction = 1.0 - (present.sum() / max(len(present), 1))
            metrics.append(Metric("continuity.dropped_frame_fraction", episode.episode_id, serial, dropped_fraction))
        return metrics

    def evaluate(self, episode: Episode, calibration: Calibration, stats: dict[str, RobustStats]) -> list[Finding]:
        findings = []
        frac_stats = stats.get("continuity.dropped_frame_fraction")

        for serial in episode.camera_serials:
            stream = episode.streams.get(serial)
            if not stream:
                continue
            present = np.asarray(stream.get("frame_present", [True] * episode.declared_frame_count), dtype=bool)
            actual_len = len(stream["color_timestamps_s"])

            if actual_len != int(present.sum()):
                findings.append(
                    Finding(
                        check="continuity.bookkeeping_mismatch",
                        severity=Severity.FAIL,
                        message=(
                            f"camera {serial}: {actual_len} timestamps recorded but frame_present marks "
                            f"{int(present.sum())} frames present -- the two disagree"
                        ),
                        evidence={"serial": serial, "n_timestamps": actual_len, "n_present": int(present.sum())},
                        downstream_consequence="frame indices for this camera cannot be safely mapped to timestamps at all",
                    )
                )

            dropped_fraction = 1.0 - (present.sum() / max(len(present), 1))
            longest_run = _longest_false_run(present)
            if dropped_fraction > 0:
                z = frac_stats.z(dropped_fraction) if frac_stats else float("inf")
                sev = severity_from_z(z) if np.isfinite(z) else Severity.FAIL
                findings.append(
                    Finding(
                        check="continuity.dropped_frames",
                        severity=sev,
                        message=(
                            f"camera {serial}: {dropped_fraction*100:.1f}% of declared frames missing "
                            f"(longest contiguous gap: {longest_run} frames / "
                            f"{longest_run/episode.fps_nominal*1000:.0f}ms)"
                        ),
                        evidence={
                            "serial": serial, "dropped_frame_fraction": dropped_fraction,
                            "longest_dropped_run_frames": longest_run,
                        },
                        downstream_consequence=(
                            "any downstream consumer that assumes a fixed frame rate (e.g. an action "
                            "chunking window in a VLA policy) sees a silently corrupted temporal window "
                            "for this camera during the gap"
                        ),
                    )
                )

        # Cross-camera agreement: do all cameras in this episode at least agree
        # on how many frames they captured? (independent of the population above,
        # this catches an episode where every camera is short by the same amount
        # -- e.g. a global metadata error -- vs one camera alone being short.)
        counts = {s: len(episode.streams[s]["color_timestamps_s"]) for s in episode.camera_serials if s in episode.streams}
        if len(set(counts.values())) > 1:
            majority = max(set(counts.values()), key=list(counts.values()).count)
            outliers = {s: c for s, c in counts.items() if c != majority}
            findings.append(
                Finding(
                    check="continuity.cross_camera_frame_count_disagreement",
                    severity=Severity.REVIEW,
                    message=(
                        f"cameras disagree on frame count within this episode: {outliers} vs majority {majority}"
                    ),
                    evidence={"counts": counts},
                    downstream_consequence="multi-view fusion for the disagreeing camera(s) will be misaligned in time relative to the rest of the rig",
                )
            )
        return findings
