import time
import argparse
from etils import epath

import mujoco
import numpy as np
from mujoco_lidar import MjLidarWrapper, scan_gen

np.set_printoptions(precision=3, suppress=True, linewidth=200)

parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与Unitree Go2 ROS2集成')
parser.add_argument('--backend', type=str, default='jax', help='LiDAR后端 (cpu, taichi, jax)', choices=['cpu', 'taichi', 'jax'])
args = parser.parse_args()

backend = args.backend

# Load model
mjcf_file = epath.Path(__file__).parent.parent / "models" / "scene_primitive.xml"
mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
mj_data = mujoco.MjData(mj_model)
mujoco.mj_step(mj_model, mj_data)

# Initialize LiDAR with selected backend
print(f"Initializing {backend.upper()} LiDAR...")
lidar = MjLidarWrapper(mj_model, site_name="lidar_site", backend=backend)

# Generate scan pattern
theta, phi = scan_gen.generate_airy96()

print(f"Number of rays: {len(theta)}")

# Run scan
print("Running scan...")
ranges = lidar.trace_rays(mj_data, theta, phi)

start = time.time()
num_runs = 10
for _ in range(num_runs):
    ranges = lidar.trace_rays(mj_data, theta, phi)
end = time.time()

print(f"Scan time: {1e3 * (end - start) / num_runs:.2f}ms")
print(f"Output shape: {ranges.shape}")
ranges = np.sort(ranges)
print(f"Sample ranges: {ranges[-10:]}")

# cpu    : Sample ranges: [11.292 11.298 11.301 11.322 11.324 11.326 11.328 11.33  11.332 11.333]
# taichi : Sample ranges: [11.292 11.298 11.301 11.322 11.324 11.326 11.328 11.33  11.332 11.333]
# jax    : 
