# Validation report

Source reports: `outputs/build`  |  Fixture: `data/fixtures/build_ground_truth.json`

## Healthy-episode false-flag rate

0/20 healthy (unfaulted) episodes were flagged REVIEW or FAIL (0.0%). At a z>=3 REVIEW threshold this is expected to be nonzero -- see docs/ANALYSIS.md for the statistical reasoning -- but should stay well under, say, 25%.

## Per-fault-type detection (recall)

| fault_type | severity | n | flagged_any | correct_check_fired |
|---|---|---|---|---|
| extrinsic_swap | mild | 2 | 100% | 100% |
| extrinsic_swap | severe | 2 | 100% | 100% |
| frame_drop | mild | 2 | 100% | 100% |
| frame_drop | severe | 2 | 100% | 100% |
| imu_desync | mild | 1 | 100% | 100% |
| imu_desync | severe | 1 | 100% | 100% |
| metadata_mismatch | mild | 1 | 100% | 100% |
| metadata_mismatch | severe | 1 | 100% | 100% |
| pose_teleport | mild | 2 | 100% | 100% |
| pose_teleport | severe | 2 | 100% | 100% |
| timestamp_drift | mild | 2 | 100% | 100% |
| timestamp_drift | severe | 2 | 100% | 100% |

Overall: 100.0% of faulted episodes were flagged REVIEW/FAIL at all; 100.0% were flagged by the *specific* check module expected for that fault type (i.e. not just 'something looked wrong' but 'the right diagnosis').

## Raw per-episode results

| episode_id   | fault_type        | severity   | recommendation   | flagged_any   | detected_specific_check   |
|:-------------|:------------------|:-----------|:-----------------|:--------------|:--------------------------|
| build_0000   | healthy           |            | PASS             | False         |                           |
| build_0001   | extrinsic_swap    | mild       | FAIL             | True          | True                      |
| build_0002   | extrinsic_swap    | severe     | FAIL             | True          | True                      |
| build_0003   | healthy           |            | PASS             | False         |                           |
| build_0004   | healthy           |            | PASS             | False         |                           |
| build_0005   | timestamp_drift   | mild       | FAIL             | True          | True                      |
| build_0006   | timestamp_drift   | severe     | FAIL             | True          | True                      |
| build_0007   | frame_drop        | mild       | FAIL             | True          | True                      |
| build_0008   | healthy           |            | PASS             | False         |                           |
| build_0009   | frame_drop        | severe     | FAIL             | True          | True                      |
| build_0010   | pose_teleport     | mild       | FAIL             | True          | True                      |
| build_0011   | healthy           |            | PASS             | False         |                           |
| build_0012   | pose_teleport     | severe     | FAIL             | True          | True                      |
| build_0013   | healthy           |            | PASS             | False         |                           |
| build_0014   | healthy           |            | PASS             | False         |                           |
| build_0015   | healthy           |            | PASS             | False         |                           |
| build_0016   | healthy           |            | PASS             | False         |                           |
| build_0017   | metadata_mismatch | mild       | FAIL             | True          | True                      |
| build_0018   | metadata_mismatch | severe     | FAIL             | True          | True                      |
| build_0019   | healthy           |            | PASS             | False         |                           |
| build_0020   | imu_desync        | mild       | FAIL             | True          | True                      |
| build_0021   | imu_desync        | severe     | FAIL             | True          | True                      |
| build_0022   | healthy           |            | PASS             | False         |                           |
| build_0023   | extrinsic_swap    | mild       | FAIL             | True          | True                      |
| build_0024   | healthy           |            | PASS             | False         |                           |
| build_0025   | extrinsic_swap    | severe     | FAIL             | True          | True                      |
| build_0026   | timestamp_drift   | mild       | FAIL             | True          | True                      |
| build_0027   | timestamp_drift   | severe     | FAIL             | True          | True                      |
| build_0028   | healthy           |            | PASS             | False         |                           |
| build_0029   | healthy           |            | PASS             | False         |                           |
| build_0030   | frame_drop        | mild       | FAIL             | True          | True                      |
| build_0031   | healthy           |            | PASS             | False         |                           |
| build_0032   | frame_drop        | severe     | FAIL             | True          | True                      |
| build_0033   | healthy           |            | PASS             | False         |                           |
| build_0034   | healthy           |            | PASS             | False         |                           |
| build_0035   | healthy           |            | PASS             | False         |                           |
| build_0036   | healthy           |            | PASS             | False         |                           |
| build_0037   | pose_teleport     | mild       | FAIL             | True          | True                      |
| build_0038   | pose_teleport     | severe     | FAIL             | True          | True                      |
| build_0039   | healthy           |            | PASS             | False         |                           |

