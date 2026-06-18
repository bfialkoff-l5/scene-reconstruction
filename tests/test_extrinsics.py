import numpy as np

from scene_recon.geometry.extrinsics import CameraPose, world_rays


class StubCamera:
    def pixels_to_rays(self, pixels):
        pixels = np.asarray(pixels, dtype=float)
        n = len(pixels)
        return np.tile([0.0, 0.0, 1.0], (n, 1))


def _hit_ground(pose, h):
    origins, dirs = world_rays(StubCamera(), pose, np.array([[0.0, 0.0]]))
    o, d = origins[0], dirs[0]
    assert d[2] < 0
    t = (0.0 - o[2]) / d[2]
    return o + t * d


def test_center_ray_matches_legacy_footprint():
    h, e0, n0 = 100.0, 500.0, 700.0
    for yaw, pitch in [(0.0, -np.pi / 4), (np.pi / 2, -np.pi / 4), (1.0, -np.pi / 3)]:
        pose = CameraPose(e0, n0, h, 0.0, pitch, yaw)
        nadir = np.pi / 2 + pitch
        fwd = h * np.tan(nadir)
        cx_e = e0 + np.sin(yaw) * fwd
        cx_n = n0 + np.cos(yaw) * fwd
        hit = _hit_ground(pose, h)
        assert np.allclose(hit[:2], [cx_e, cx_n], atol=1e-6)


def test_nadir_straight_down():
    h, e0, n0 = 50.0, 10.0, 20.0
    pose = CameraPose(e0, n0, h, 0.0, -np.pi / 2, 0.7)
    hit = _hit_ground(pose, h)
    assert np.allclose(hit[:2], [e0, n0], atol=1e-6)


def test_orthonormal_det_plus_one():
    for roll, pitch, yaw in [(0.0, -0.3, 0.5), (0.4, -1.0, 2.0), (-0.6, -np.pi / 2, 1.2)]:
        R = CameraPose(0, 0, 100, roll, pitch, yaw).R_cam_to_enu()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_roll_preserves_forward_rotates_right():
    pitch, yaw = -0.4, 1.3
    R0 = CameraPose(0, 0, 100, 0.0, pitch, yaw).R_cam_to_enu()
    R1 = CameraPose(0, 0, 100, 0.5, pitch, yaw).R_cam_to_enu()
    fwd = np.array([0.0, 0.0, 1.0])
    assert np.allclose(R0 @ fwd, R1 @ fwd, atol=1e-12)
    right = np.array([1.0, 0.0, 0.0])
    assert not np.allclose(R0 @ right, R1 @ right, atol=1e-3)
