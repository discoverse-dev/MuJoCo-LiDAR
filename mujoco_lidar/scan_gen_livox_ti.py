"""
Livox LiDAR 扫描模式生成器

注意：此模块需要 taichi 依赖
"""
from typing import Tuple, Dict, Any

import os
import numpy as np

# Import taichi for LivoxGenerator class
import taichi as ti

# ti.init(arch=ti.gpu) # Moved to __init__

@ti.data_oriented
class LivoxGeneratorTi:
    """Livox 扫描模式：预加载全部角度到 Taichi，采样返回 Taichi ndarray。"""
    livox_lidar_params = {
        "avia": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 70.4, "vertical_fov": 77.2, "samples": 24000},
        "HAP": {"laser_min_range": 0.1, "laser_max_range": 200.0, "samples": 45300, "downsample": 1},
        "horizon": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 81.7, "vertical_fov": 25.1, "samples": 24000},
        "mid40": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 81.7, "vertical_fov": 25.1, "samples": 24000},
        "mid70": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 70.4, "vertical_fov": 70.4, "samples": 10000},
        "mid360": {"laser_min_range": 0.1, "laser_max_range": 200.0, "samples": 24000},
        "tele": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 14.5, "vertical_fov": 16.1, "samples": 24000},
    }

    def __init__(self, name: str):
        # Initialize Taichi if not already done
        if not hasattr(ti, '_is_initialized') or not ti._is_initialized:
            ti.init(arch=ti.gpu)

        if name not in self.livox_lidar_params:
            raise ValueError(f"Invalid LiDAR name: {name}")
        p = self.livox_lidar_params[name]
        self.samples = p["samples"]
        pattern_npy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_mode", f"{name}.npy")
        if not os.path.isfile(pattern_npy_path):
            raise FileNotFoundError(f"Scan mode file not found: {pattern_npy_path}")
        ray_angles_np = np.load(pattern_npy_path).astype(np.float32)  # shape (N,2) -> (theta, phi)
        if ray_angles_np.shape[1] != 2:
            raise ValueError("scan_mode npy 第二维应为2 (theta, phi)")
        self.n_rays = ray_angles_np.shape[0]
        # 全量角度放入 field
        self.theta_all = ti.field(dtype=ti.f32, shape=self.n_rays)
        self.phi_all = ti.field(dtype=ti.f32, shape=self.n_rays)
        self.theta_all.from_numpy(ray_angles_np[:, 0])
        self.phi_all.from_numpy(ray_angles_np[:, 1])
        # 采样缓存（可重建）
        self._theta_sample = None
        self._phi_sample = None
        self._sample_size = 0
        self.currStartIndex = 0

    def _ensure_sample_buf(self, size: int):
        if self._theta_sample is None or self._sample_size != size:
            self._theta_sample = ti.ndarray(dtype=ti.f32, shape=size)
            self._phi_sample = ti.ndarray(dtype=ti.f32, shape=size)
            self._sample_size = size

    @ti.kernel
    def _gather_kernel(self, 
                       start: ti.i32, 
                       step: ti.i32, 
                       size: ti.i32, 
                       theta_out: ti.types.ndarray(dtype=ti.f32, ndim=1), 
                       phi_out: ti.types.ndarray(dtype=ti.f32, ndim=1), 
                       n_total: ti.i32
    ):
        for i in ti.ndrange(size):
            idx = (start + i * step) % n_total
            theta_out[i] = self.theta_all[idx]
            phi_out[i] = self.phi_all[idx]

    def sample_ray_angles_ti(self, downsample: int = 1) -> Tuple[ti.types.ndarray, ti.types.ndarray]:
        if downsample < 1:
            downsample = 1
        eff = self.samples // downsample if downsample > 1 else self.samples
        self._ensure_sample_buf(eff)
        self._gather_kernel(self.currStartIndex, downsample, eff, self._theta_sample, self._phi_sample, self.n_rays)
        # 前进"samples"步（保持与原算法一致）
        self.currStartIndex = (self.currStartIndex + self.samples) % self.n_rays
        return self._theta_sample, self._phi_sample

    # 兼容旧接口（需要时仍可得到 numpy，但会有拷贝）
    def sample_ray_angles(self, downsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        th_ti, ph_ti = self.sample_ray_angles_ti(downsample)
        return th_ti.to_numpy(), ph_ti.to_numpy()
