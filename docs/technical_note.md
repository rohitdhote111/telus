# Technical note — Problem 1 (episode-analysis / data-quality tool)

## Dataset choice

**DexYCB** (Chao et al., CVPR 2021) was selected as the anchor dataset, but only its
**calibration corpus** is used as real data — see "What's missing" below for why the full
capture was not pulled in, and how the gap is filled.

DexYCB records tabletop grasping with an 8-camera Intel RealSense D415 rig (fixed RGB-D
cameras, not egocentric), MANO hand-pose annotation, and 6D object pose, which covers most of
the modalities the case study lists (camera/depth data, calibration, coordinate transforms, pose
trajectories) and is exactly the kind of corpus a VLA/manipulation pipeline like the one in the case
study would be built on. It was chosen over the alternatives on the assignment's suggested list
for a concrete, checked reason, not just familiarity:

| Dataset | Why not used as the primary source here |
|---|---|
| HOI4D | Only available via OneDrive/Baidu share links — not reliably scriptable, no way to fetch a small slice without manual browser interaction |
| HO-3D | Download is gated behind a Nextcloud share/click-through form |
| Ego4D / Ego-Exo4D | Requires a signed license agreement — not completable in an unattended session |
| EPIC-KITCHENS | RGB-only; no depth, calibration, or IMU, which is a weak fit for the geometry/sync axis this tool focuses on |
| DexYCB | Public, direct Google Drive links, no click-through license gate — **but the full dataset is 119GB (12GB per subject)**, too large to responsibly pull into this exercise |

`data/real/calibration/` was fetched directly from DexYCB's own separate `calibration.tar.gz`
(16KB — a real, distinct download from the 12GB-per-subject captures) and used completely
as-is, unmodified. It contains, for the real physical rig:
- **8 camera intrinsics files** (color + depth focal length/principal point, plus the factory
  depth-to-color extrinsic for that physical unit).
- **10 extrinsics sessions**, one per real recalibration event/capture day, each giving all 8
  cameras' rotation+translation relative to a fixed reference camera.
- **10 MANO hand-shape ("beta") calibrations**, one per real subject.

This alone was investigated before writing any check (see "what the real calibration data
actually shows" below) — it is not treated as ground truth by assumption.

## What it does not contain that a real engagement would need

- **Real RGB/depth image bytes.** The 12–119GB raw capture was not pulled in; this tool never
  looks at pixel content, only at timestamps, calibration, and per-frame pose/keypoint numbers.
- **Real per-frame hand/object annotations at the scale of a full capture day.** DexYCB's real
  `joint_3d`/`joint_2d`/`pose_y` labels were not used; a schema-faithful *synthetic* stand-in was
  generated instead (smooth 3D hand trajectory, per-camera 2D re-projections with detector-level
  noise) — see "Synthetic layer" below.
- **A real IMU stream.** DexYCB does not ship IMU data at all. The IMU channel in this tool's
  episodes is entirely illustrative/synthetic (derived by double-differentiating the synthetic hand
  trajectory), included only so the sync-check reasoning the assignment asks for ("timestamps...
  IMU") has something concrete to operate on. Any real engagement would need to redo this check
  against an actual IMU log with its own real noise/bias characteristics.
- **Real annotator/QC history.** No record of who labeled what, inter-annotator agreement, or a
  prelabel-vs-human-reviewed split (that is Problem 2's territory, not built here).
- **A record of *why* the rig was recalibrated 10 times.** The real extrinsics-session dates
  imply the physical rig was re-mounted repeatedly; DexYCB does not document why, so this tool
  cannot distinguish "routine recalibration" from "something moved after a bump" except by the
  magnitude of the change (see below).

## What the real calibration data actually shows (investigated, not assumed)

Before designing any check, the real extrinsics/intrinsics were probed directly:

- All 10 real extrinsics sessions list the **same 8 camera serials** plus an `apriltag` entry (the
  board defining the tabletop frame); camera `840412060917` is the identity transform in **every**
  session, confirming it is the fixed reference/world frame by convention, not by assumption.
- Every real extrinsic rotation matrix is numerically clean: `||RᵀR − I||_F` never exceeds `5×10⁻⁷`
  across all 10 sessions × 8 cameras — i.e. the real corpus has *zero* orthonormality violations,
  which set the "this should never happen" tolerance (`ORTHONORMALITY_TOL`) used later in
  `calibration_checks.py`'s absolute rotation-validity check.
- **The same physical camera's extrinsic drifts by 14–25° (rotation) and up to ~0.37m
  (translation) between capture sessions.** This was an explicit design trap avoided: it would be
  easy to write a "does this camera's calibration match its value in other sessions" check and have
  it fire constantly on completely normal data, because the rig was evidently physically
  re-mounted between DexYCB's capture days. Every check in this tool that compares an extrinsic
  only ever compares an episode's *used* value against its own *declared* session's canonical
  value — never against a different session — for exactly this reason.
- Per-camera color/depth intrinsics are tightly clustered across all 8 units (`fx` median 615.7,
  std 2.7; `ppx` median 314.3, std 6.5) and each unit's internal color→depth baseline is ~1.50cm
  (std ~0.01cm) — a real, usable population for a data-driven plausibility check, though this
  corpus's fault taxonomy does not currently exercise it (see limitations).

## Synthetic layer (what's generated, and why)

Per-episode timestamps, the 3D hand trajectory, per-camera 2D detections, task/episode metadata,
and the illustrative IMU stream are generated by `data_gen/generate_episodes.py`. Every episode:
- draws a **real** extrinsics-session id, a **real** subject/MANO-calibration id, and uses the
  **real** 8 camera serials and intrinsics — so every geometry check runs against genuine
  calibration numbers, never invented ones;
- gets a smooth 3D hand path (a clamped cubic spline through 3–5 random waypoints in a plausible
  tabletop workspace) sampled at a nominal 30fps, annotated with 2mm noise;
- gets each camera's 2D observation of that same path by projecting it through that camera's real
  intrinsics/extrinsics plus ~1.5px detector noise (RealSense-plausible USB timestamp jitter of
  ~1.5ms is added to color/depth timestamps independently per camera);
- gets an illustrative 200Hz IMU stream from double-differentiating the same path plus a bias/
  noise term.

This is disclosed in full, not hidden: the geometry, calibration-consistency, and timing checks
are exercised against real sensor numbers; the specific hand motion, its exact timing, and the IMU
channel are synthetic.

## Faults deliberately injected (and how)

`data_gen/inject_faults.py` corrupts exactly one property of a chosen fraction of episodes
(50% in this run — the rest are healthy controls), seeded and reproducible. Each fault has a
**mild** and **severe** magnitude:

| Fault | What is corrupted | Mild | Severe | Meant to simulate |
|---|---|---|---|---|
| `extrinsic_swap` | one camera's `extrinsics_used` is rotated/translated away from its episode's declared session | 2°/1cm | 15°/8cm | wrong/mis-attached calibration file for one camera |
| `timestamp_drift` | one camera's color+depth timestamps drift linearly over the episode | 30ms by episode end | 150ms | a camera's clock skewing relative to the rig |
| `frame_drop` | a contiguous run of one camera's frames is removed (raw stream + its 2D detections) | 3 frames | 15 frames | dropped frames during capture/transfer |
| `pose_teleport` | one frame of the 3D hand trajectory is displaced | 0.15m in 1/30s | 0.6m in 1/30s | a tracking glitch/mislink in the annotation pipeline |
| `metadata_mismatch` | declared episode duration is scaled away from frame_count/fps | ±15% | ±60% | metadata generated/edited independently of the actual capture |
| `imu_desync` | all IMU timestamps are shifted by a fixed offset | ±50ms | ±250ms | an unsynchronized IMU clock |

Two independently-seeded batches were generated from this pipeline: a **build** batch
(`data/synthetic/build`, seed 1, 40 episodes) used while developing and tuning the tool, and a
**holdout** batch (`data/synthetic/holdout`, seed 999, 24 episodes) generated and analyzed only
after the tool's thresholds were frozen, to check the "must generalize to unseen data" requirement
honestly rather than by assertion. Ground truth for both lives in `data/fixtures/*_ground_truth.json`
and is read only by `data_gen/validate.py` — the analysis tool itself never sees it.

Full validation results (precision/recall per fault type, and the healthy-episode false-flag rate)
are in `outputs/build/validation_report.md` and `outputs/holdout/validation_report.md`.

## Assumptions

- A "healthy" episode is defined as one with no deliberately injected fault; natural sensor/driver
  jitter (timestamp noise, detector noise) is present in every episode, healthy or not, and checks
  are tuned (via the REVIEW/FAIL z-thresholds in `scoring.py`) to tolerate it.
- The reference/world frame is the identity camera per session, matching the real data's own
  convention (verified, not assumed — see above).
- Episodes share a fixed 8-camera rig and 30fps nominal rate; the tool does not currently handle a
  variable camera count per episode (documented limitation).

## Known limitations

- Linear (DLT) multi-view triangulation is a joint least-squares fit: an extreme, workspace-scale
  extrinsic error on one view can pull every camera's reprojection residual up together rather than
  isolating the bad one (demonstrated in `tests/test_geometry.py`). At the realistic fault
  magnitudes actually injected (≤15°/8cm against an 8-camera rig), this was not observed to be a
  problem in either validated batch, but a corpus with much larger calibration errors would need a
  robust estimator (e.g. RANSAC over camera subsets) instead of plain DLT.
- An intrinsics-plausibility check (fx/fy/principal-point z-score against the rest of the rig,
  motivated by "what the real data shows" above) was prototyped and then reverted before shipping:
  with only 8 real physical cameras, a robust z-score population of size 8 is too small — real
  hardware manufacturing variance alone put one genuine, undamaged camera (fx=621.8 vs. a median
  of 615.7 across the other 7) at z=5.5, which flagged every episode referencing it — a 100%
  false-flag rate the moment it was turned on. It was removed rather than shipped broken. A real
  fix needs either a much larger camera population
  or a domain prior (e.g. the documented RealSense D415 factory tolerance) instead of a
  corpus-derived threshold.
- IMU reasoning is validated only against a synthetic, illustrative IMU signal — see "what's
  missing" above.
- The tool assumes single-hand tracking (one 3D keypoint trajectory); it does not model bimanual
  episodes or multi-object contact state.
