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
simulation_app = SimulationApp({"headless": False})
import carb
import omni
import omni.kit.viewport.utility
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

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
test_path = "/home/yiyi/Data/github/MuJoCo-LiDAR/models/scene.usd" 
# stage.add_reference_to_stage(
#     assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "/background"
# )
stage.add_reference_to_stage(
    test_path, "/background"
)
simulation_app.update()

lidar_config = args.config

# Create the lidar sensor that generates data into "RtxSensorCpu"
# Possible config options are Example_Rotary and Example_Solid_State
_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path="/sensor",
    parent=None,
    config=lidar_config,
    translation=(0, 0, 1.0),
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
)
hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")

simulation_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
simulation_app.update()

# Create the debug draw pipeline in the post process graph
writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud" + "Buffer")
writer.attach([hydra_texture])

simulation_app.update()

simulation_context.play()

while simulation_app.is_running():
    num_points = 2000
    radius = 3.0
    points = np.random.normal(size=(num_points, 3))
    points = radius * points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    
    # Move the points to the robot's location
    points += np.array([0, 0, 0])
    
    # Define colors (Red) and sizes for the points
    green_color = (0, 1, 0, 0.75)
    green_size = 4
    
    debug_draw_clear_points()
    debug_draw_pointcloud(points, green_color, green_size)
    
    simulation_app.update()

# cleanup and shutdown
simulation_context.stop()
simulation_app.close()