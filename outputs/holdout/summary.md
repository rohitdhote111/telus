# Episode QC summary

Episodes analyzed: 24
PASS: 11  REVIEW: 1  FAIL: 12

| episode_id | recommendation | n_findings | checks_flagged |
|---|---|---|---|
| holdout_0000 | PASS | 0 |  |
| holdout_0001 | FAIL | 4 | calibration.extrinsic_mismatch;calibration.reprojection_outlier |
| holdout_0002 | PASS | 0 |  |
| holdout_0003 | PASS | 0 |  |
| holdout_0004 | FAIL | 10 | calibration.extrinsic_mismatch;calibration.reprojection_outlier;sync.stream_clock_mismatch |
| holdout_0005 | FAIL | 1 | sync.stream_clock_mismatch |
| holdout_0006 | FAIL | 1 | sync.stream_clock_mismatch |
| holdout_0007 | FAIL | 3 | continuity.cross_camera_frame_count_disagreement;continuity.dropped_frames;sync.stream_clock_mismatch |
| holdout_0008 | FAIL | 2 | continuity.cross_camera_frame_count_disagreement;continuity.dropped_frames |
| holdout_0009 | PASS | 0 |  |
| holdout_0010 | FAIL | 2 | trajectory.implausible_jump |
| holdout_0011 | FAIL | 2 | trajectory.implausible_jump |
| holdout_0012 | REVIEW | 1 | trajectory.implausible_jump |
| holdout_0013 | PASS | 0 |  |
| holdout_0014 | PASS | 0 |  |
| holdout_0015 | PASS | 0 |  |
| holdout_0016 | PASS | 0 |  |
| holdout_0017 | PASS | 0 |  |
| holdout_0018 | FAIL | 1 | metadata.duration_frame_count_mismatch |
| holdout_0019 | FAIL | 1 | metadata.duration_frame_count_mismatch |
| holdout_0020 | FAIL | 1 | sync.stream_clock_mismatch |
| holdout_0021 | PASS | 0 |  |
| holdout_0022 | PASS | 0 |  |
| holdout_0023 | FAIL | 1 | sync.stream_clock_mismatch |
