# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="Example_Rotary", help="Name of lidar config.")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

# Example for creating a RTX lidar sensor and publishing PCL data
simulation_app = SimulationApp({"headless": True})
import carb
import omni
import omni.kit.viewport.utility
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

from mujoco_lidar import (
    LidarSensor, LivoxGenerator,
    generate_vlp32, generate_HDL64, generate_os128
)
from mujoco_lidar.mj_lidar_utils import create_demo_scene

test_path = "/home/yiyi/Data/github/MuJoCo-LiDAR/models/scene.usd"
obj_path = "/home/yiyi/Data/github/MuJoCo-LiDAR/models/test_ply.obj"
mj_model, mj_data = create_demo_scene("floor")
lidar_sensor = LidarSensor(
    mj_model,
    site_name="lidar_site",
    backend="gpu", 
    obj_path=obj_path)

import taichi as ti
use_livox_lidar = False
lidar = "HDL64"
if lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
    livox_generator = LivoxGenerator(lidar)
    rays_theta, rays_phi = livox_generator.sample_ray_angles()
    use_livox_lidar = True
elif lidar == "HDL64":
    rays_theta, rays_phi = generate_HDL64()
elif lidar == "vlp32":
    rays_theta, rays_phi = generate_vlp32()
elif lidar == "os128":
    rays_theta, rays_phi = generate_os128()
else:
    raise ValueError(f"Unsupported lidar type: {lidar}")

rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)
n_rays = len(rays_theta)
_rays_phi = ti.ndarray(dtype=ti.f32, shape=n_rays)
_rays_theta = ti.ndarray(dtype=ti.f32, shape=n_rays)
_rays_phi.from_numpy(rays_phi)
_rays_theta.from_numpy(rays_theta)
rays_phi = _rays_phi
rays_theta = _rays_theta

# enable ROS bridge extension
enable_extension("isaacsim.util.debug_draw")

def debug_draw_clear_points():
    from isaacsim.util.debug_draw import _debug_draw

    draw_iface = _debug_draw.acquire_debug_draw_interface()
    draw_iface.clear_points()

def debug_draw_pointcloud(pointcloud_data, color, size, clear_existing=False):
    if not (isinstance(pointcloud_data, np.ndarray) and pointcloud_data.ndim == 2 and pointcloud_data.shape[1] == 3):
        print("Warning: pointcloud_data must be a NumPy array with shape (N, 3).")
        return

    from isaacsim.util.debug_draw import _debug_draw

    draw_iface = _debug_draw.acquire_debug_draw_interface()

    points_cloud = []
    colors_cloud = []
    sizes_cloud = []
    for i in range(pointcloud_data.shape[0]):
        points_cloud.append(pointcloud_data[i].tolist())
        colors_cloud.append(color)
        sizes_cloud.append(size)

    if clear_existing:
        debug_draw_clear_points()
    draw_iface.draw_points(points_cloud, colors_cloud, sizes_cloud)

simulation_app.update()

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

simulation_app.update()
# Loading the simple_room environment

# stage.add_reference_to_stage(
#     assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "/background"
# )
stage.add_reference_to_stage(
    test_path, "/background"
)
simulation_app.update()

lidar_config = args.config

simulation_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
simulation_app.update()


simulation_app.update()

simulation_context.play()

lidar_sensor.update(
    mj_data,
    rays_phi,
    rays_theta
)
while simulation_app.is_running():

    # Define colors (Red) and sizes for the points
    green_color = (0, 1, 0, 0.75)
    green_size = 4
    lidar_sensor.update(
        mj_data,
        rays_phi,
        rays_theta
    )
    points = lidar_sensor.get_data_in_local_frame()
    if len(points) == 0:
        num_points = 2000
        radius = 3.0
        points = np.random.normal(size=(num_points, 3))
        points = radius * points / np.linalg.norm(points, axis=1)[:, np.newaxis]

        # Move the points to the robot's location
        points += np.array([0, 0, 0])
    debug_draw_clear_points()
    debug_draw_pointcloud(points, green_color, green_size, clear_existing=True)

    simulation_app.update()

# cleanup and shutdown
simulation_context.stop()
simulation_app.close()
