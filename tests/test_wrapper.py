import pytest

from mujoco_lidar import MjLidarWrapper


def test_wrapper_backend_selection(simple_model):
    """测试后端选择"""
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu")
    assert lidar.backend == "cpu"


def test_wrapper_invalid_backend(simple_model):
    """测试无效后端"""
    with pytest.raises((ValueError, KeyError, ImportError)):
        MjLidarWrapper(simple_model, site_name="lidar_site", backend="invalid")


def test_wrapper_cutoff_dist(simple_model):
    """测试截断距离设置"""
    lidar = MjLidarWrapper(simple_model, site_name="lidar_site", backend="cpu", cutoff_dist=5.0)
    assert lidar.cutoff_dist == 5.0
