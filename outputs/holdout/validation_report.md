# Validation report

Source reports: `outputs/holdout`  |  Fixture: `data/fixtures/holdout_ground_truth.json`

## Healthy-episode false-flag rate

1/12 healthy (unfaulted) episodes were flagged REVIEW or FAIL (8.3%). At a z>=3 REVIEW threshold this is expected to be nonzero -- see docs/ANALYSIS.md for the statistical reasoning -- but should stay well under, say, 25%.

## Per-fault-type detection (recall)

| fault_type | severity | n | flagged_any | correct_check_fired |
|---|---|---|---|---|
| extrinsic_swap | mild | 1 | 100% | 100% |
| extrinsic_swap | severe | 1 | 100% | 100% |
| frame_drop | mild | 1 | 100% | 100% |
| frame_drop | severe | 1 | 100% | 100% |
| imu_desync | mild | 1 | 100% | 100% |
| imu_desync | severe | 1 | 100% | 100% |
| metadata_mismatch | mild | 1 | 100% | 100% |
| metadata_mismatch | severe | 1 | 100% | 100% |
| pose_teleport | mild | 1 | 100% | 100% |
| pose_teleport | severe | 1 | 100% | 100% |
| timestamp_drift | mild | 1 | 100% | 100% |
| timestamp_drift | severe | 1 | 100% | 100% |

Overall: 100.0% of faulted episodes were flagged REVIEW/FAIL at all; 100.0% were flagged by the *specific* check module expected for that fault type (i.e. not just 'something looked wrong' but 'the right diagnosis').

## Raw per-episode results

| episode_id   | fault_type        | severity   | recommendation   | flagged_any   | detected_specific_check   |
|:-------------|:------------------|:-----------|:-----------------|:--------------|:--------------------------|
| holdout_0000 | healthy           |            | PASS             | False         |                           |
| holdout_0001 | extrinsic_swap    | mild       | FAIL             | True          | True                      |
| holdout_0002 | healthy           |            | PASS             | False         |                           |
| holdout_0003 | healthy           |            | PASS             | False         |                           |
| holdout_0004 | extrinsic_swap    | severe     | FAIL             | True          | True                      |
| holdout_0005 | timestamp_drift   | mild       | FAIL             | True          | True                      |
| holdout_0006 | timestamp_drift   | severe     | FAIL             | True          | True                      |
| holdout_0007 | frame_drop        | mild       | FAIL             | True          | True                      |
| holdout_0008 | frame_drop        | severe     | FAIL             | True          | True                      |
| holdout_0009 | healthy           |            | PASS             | False         |                           |
| holdout_0010 | pose_teleport     | mild       | FAIL             | True          | True                      |
| holdout_0011 | pose_teleport     | severe     | FAIL             | True          | True                      |
| holdout_0012 | healthy           |            | REVIEW           | True          |                           |
| holdout_0013 | healthy           |            | PASS             | False         |                           |
| holdout_0014 | healthy           |            | PASS             | False         |                           |
| holdout_0015 | healthy           |            | PASS             | False         |                           |
| holdout_0016 | healthy           |            | PASS             | False         |                           |
| holdout_0017 | healthy           |            | PASS             | False         |                           |
| holdout_0018 | metadata_mismatch | mild       | FAIL             | True          | True                      |
| holdout_0019 | metadata_mismatch | severe     | FAIL             | True          | True                      |
| holdout_0020 | imu_desync        | mild       | FAIL             | True          | True                      |
| holdout_0021 | healthy           |            | PASS             | False         |                           |
| holdout_0022 | healthy           |            | PASS             | False         |                           |
| holdout_0023 | imu_desync        | severe     | FAIL             | True          | True                      |

