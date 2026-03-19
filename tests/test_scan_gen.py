from mujoco_lidar import scan_gen


def test_generate_hdl64():
    theta, phi = scan_gen.generate_HDL64()
    assert len(theta) > 0
    assert len(theta) == len(phi)


def test_generate_airy96():
    theta, phi = scan_gen.generate_airy96()
    assert len(theta) > 0
    assert len(theta) == len(phi)
