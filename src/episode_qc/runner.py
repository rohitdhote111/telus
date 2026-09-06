"""Two-pass corpus runner.

Pass 1 ("collect"): every check module computes raw numeric metrics for every
episode (and, within an episode, every camera/stream it applies to) without
making any PASS/REVIEW/FAIL judgement yet.

Between passes: metrics are pooled *by key* across the whole corpus that was
handed to this run, and a robust (median/MAD) population is built per key.

Pass 2 ("evaluate"): each check module re-visits every episode and turns its
raw metrics into Findings, comparing against the population built a moment
ago. Because the population comes from whatever corpus was passed in, the
same code produces different numeric thresholds on a 5-episode toy corpus
than on a 5,000-episode real one, without a single constant being edited.

A handful of checks are physically absolute (a rotation matrix's determinant
must be +1) and skip the population step entirely — see the module docstring
in calibration_checks.py for which ones and why.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .io import Calibration, Episode
from .scoring import EpisodeReport, Finding, RobustStats, robust_stats


@dataclass
class Metric:
    key: str
    episode_id: str
    entity: str
    value: float


class CheckModule:
    name: str = "unnamed"

    def collect(self, episode: Episode, calibration: Calibration) -> list[Metric]:
        raise NotImplementedError

    def evaluate(
        self, episode: Episode, calibration: Calibration, stats: dict[str, RobustStats]
    ) -> list[Finding]:
        raise NotImplementedError


def build_population(metrics: list[Metric]) -> dict[str, RobustStats]:
    by_key: dict[str, list[float]] = defaultdict(list)
    for m in metrics:
        by_key[m.key].append(m.value)
    return {k: robust_stats(v) for k, v in by_key.items()}


def run_corpus(
    episodes: list[Episode], calibration: Calibration, modules: list[CheckModule]
) -> dict[str, EpisodeReport]:
    all_metrics: list[Metric] = []
    for ep in episodes:
        for mod in modules:
            all_metrics.extend(mod.collect(ep, calibration))

    stats = build_population(all_metrics)

    reports: dict[str, EpisodeReport] = {}
    for ep in episodes:
        findings: list[Finding] = []
        for mod in modules:
            findings.extend(mod.evaluate(ep, calibration, stats))
        reports[ep.episode_id] = EpisodeReport(episode_id=ep.episode_id, findings=findings)
    return reports
