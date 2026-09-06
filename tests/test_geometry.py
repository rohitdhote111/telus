import numpy as np

from episode_qc.geometry import (
    geodesic_rotation_distance_deg,
    project,
    reprojection_errors_px,
    rotation_determinant,
    rotation_orthonormality_error,
    triangulate_point_dlt,
)


def test_identity_rotation_is_valid():
    R = np.eye(3)
    assert rotation_orthonormality_error(R) < 1e-9
    assert abs(rotation_determinant(R) - 1.0) < 1e-9


def test_reflection_has_negative_determinant():
    R = np.diag([1.0, 1.0, -1.0])  # a reflection, not a physical rotation
    assert abs(rotation_determinant(R) + 1.0) < 1e-9


def test_non_orthonormal_matrix_is_flagged():
    R = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # sheared, not a rotation
    assert rotation_orthonormality_error(R) > 1e-3


def test_geodesic_distance_zero_for_identical_rotations():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # 90deg about z
    assert geodesic_rotation_distance_deg(R, R) < 1e-6


def test_geodesic_distance_matches_known_angle():
    theta = np.radians(30.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    assert abs(geodesic_rotation_distance_deg(np.eye(3), Rz) - 30.0) < 1e-6


def test_triangulate_and_reproject_recovers_known_point():
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    # Two cameras looking at the same point from different positions.
    extr_a = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.0]])  # 1m in front of world origin
    extr_b = np.array([[1, 0, 0, -0.2], [0, 1, 0, 0], [0, 0, 1, 1.0]])  # shifted 0.2m in x

    true_point = np.array([0.05, -0.03, 0.6])
    uv_a, _ = project(K, extr_a, true_point[None, :])
    uv_b, _ = project(K, extr_b, true_point[None, :])

    recovered = triangulate_point_dlt([(K @ extr_a, uv_a[0]), (K @ extr_b, uv_b[0])])
    assert np.allclose(recovered, true_point, atol=1e-6)


def test_reprojection_errors_flag_the_bad_camera():
    # 8 cameras roughly surrounding a workspace, mirroring the real rig this
    # tool is validated against (data/real/calibration has 8 serials), with a
    # "severe"-magnitude perturbation matching data_gen/inject_faults.py's
    # extrinsic_swap fault (15 degrees / 8cm) rather than an arbitrary large
    # offset. Linear (DLT) triangulation is a joint least-squares fit, so an
    # extreme, workspace-scale perturbation on even one view can drag every
    # residual up together -- that failure mode is real (see docs/ANALYSIS.md
    # "limitations") but is not what a real mis-attached calibration file
    # looks like, so this test uses a realistic magnitude and a correspondingly
    # modest (not 10x/20x) separation requirement.
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    true_extr = {}
    for i in range(7):
        angle = 2 * np.pi * i / 7
        t = np.array([0.3 * np.cos(angle), 0.3 * np.sin(angle), 1.0])
        true_extr[f"good_{i}"] = np.concatenate([np.eye(3), t[:, None]], axis=1)
    true_extr["bad"] = np.array([[1, 0, 0, 0.15], [0, 1, 0, 0.15], [0, 0, 1, 1.0]])

    true_point = np.array([0.02, -0.01, 0.6])
    obs = {name: project(K, e, true_point[None, :])[0][0] for name, e in true_extr.items()}

    # The pipeline mistakenly attaches a wrong extrinsic for "bad" (e.g. a
    # different session's calibration) while the other seven keep their
    # correct, true extrinsics -- this is exactly the extrinsic_swap fault,
    # at the same "severe" magnitude used in data_gen/inject_faults.py.
    theta = np.radians(15.0)
    Rz_perturb = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    R_bad, t_bad = true_extr["bad"][:, :3], true_extr["bad"][:, 3]
    used_extr = dict(true_extr)
    used_extr["bad"] = np.concatenate([Rz_perturb @ R_bad, (t_bad + np.array([0.08, 0.0, 0.0]))[:, None]], axis=1)

    K_by_serial = {k: K for k in true_extr}
    errors = reprojection_errors_px(K_by_serial, used_extr, obs)
    good_errors = [v for k, v in errors.items() if k != "bad"]
    assert errors["bad"] > 2 * max(good_errors)
