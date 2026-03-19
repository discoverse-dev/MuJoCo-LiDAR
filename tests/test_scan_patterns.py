import numpy as np

from mujoco_lidar import scan_gen


def test_generate_vlp32():
    theta, phi = scan_gen.generate_vlp32()
    assert len(theta) > 0
    assert len(theta) == len(phi)


def test_generate_os128():
    theta, phi = scan_gen.generate_os128()
    assert len(theta) > 0
    assert len(theta) == len(phi)


def test_grid_scan():
    theta, phi = scan_gen.generate_grid_scan_pattern(num_ray_cols=10, num_ray_rows=5)
    assert len(theta) == 50
    assert len(phi) == 50


def test_single_line_scan():
    theta, phi = scan_gen.create_lidar_single_line(horizontal_resolution=360)
    assert len(theta) == 360
    assert np.all(phi == 0)
