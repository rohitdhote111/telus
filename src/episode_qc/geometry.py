"""Pure-geometry helpers: rotation validity, projection, multi-view
triangulation and reprojection error. No episode/check-specific logic lives
here so it can be unit-tested against hand-built matrices independent of the
data schema.
"""

from __future__ import annotations

import numpy as np


def rotation_orthonormality_error(R: np.ndarray) -> float:
    """||R^T R - I||_F. Zero for a perfect rotation matrix."""
    return float(np.linalg.norm(R.T @ R - np.eye(3)))


def rotation_determinant(R: np.ndarray) -> float:
    """+1 for a proper rotation, -1 for a reflection (physically impossible
    for a rigid camera mount -> always a defect, never natural noise)."""
    return float(np.linalg.det(R))


def geodesic_rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle (degrees) of the rotation that takes R_a to R_b."""
    dR = R_a.T @ R_b
    trace = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def project(K: np.ndarray, extrinsic_3x4: np.ndarray, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points through [R|t] then K.

    Returns (pixels Nx2, depth N) — depth is the camera-frame z, so callers
    can discard points behind the camera (depth <= 0) before trusting the
    pixel coordinates.
    """
    R, t = extrinsic_3x4[:, :3], extrinsic_3x4[:, 3]
    cam = points_world @ R.T + t
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv1 = (cam / safe_z[:, None]) @ K.T
    return uv1[:, :2], z


def triangulate_point_dlt(cams: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Linear (DLT) triangulation of one 3D point from >=2 views.

    ``cams`` is a list of (P, uv) where P is the 3x4 camera projection matrix
    (K @ [R|t]) and uv is the observed 2D pixel for that view.
    """
    if len(cams) < 2:
        raise ValueError("triangulation needs at least 2 views")
    A = []
    for P, (u, v) in cams:
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]
    return X


def reprojection_errors_px(
    K_by_serial: dict[str, np.ndarray],
    extrinsic_by_serial: dict[str, np.ndarray],
    observed_uv_by_serial: dict[str, np.ndarray],
) -> dict[str, float]:
    """Triangulate a 3D point from all provided views, then report each
    view's own reprojection residual (pixels) against that reconstruction.

    A camera whose *own* extrinsic is wrong will show up with high residual
    here even though the point was reconstructed using *all* views (including
    the bad one) — the bad view still pulls the DLT solution towards it, but
    with 3+ well-conditioned views the good cameras dominate the least-squares
    fit and the outlier camera's residual remains the largest by a wide
    margin. This is the standard way a multi-camera rig catches a single
    mis-calibrated unit.
    """
    Ps = {s: K_by_serial[s] @ extrinsic_by_serial[s] for s in observed_uv_by_serial}
    X = triangulate_point_dlt([(Ps[s], observed_uv_by_serial[s]) for s in Ps])
    errors = {}
    for s, uv in observed_uv_by_serial.items():
        pred, z = project(K_by_serial[s], extrinsic_by_serial[s], X[None, :])
        errors[s] = float(np.linalg.norm(pred[0] - uv)) if z[0] > 0 else float("nan")
    return errors
