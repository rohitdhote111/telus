"""episode_qc — episode-level data-quality analysis for a manipulation dataset.

The package is deliberately dependency-light (numpy/scipy/pandas/pyyaml) and
schema-driven: nothing in here refers to a specific episode id, camera serial,
or calibration session by name. Everything the checks need (which cameras
exist, what the "normal" range of a signal looks like) is computed at run
time from whatever corpus of episodes + calibration is passed on the command
line. See docs/ANALYSIS.md for the full design rationale.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("episode_qc")
except Exception:  # pragma: no cover - not installed as a package
    __version__ = "0.0.0-dev"
