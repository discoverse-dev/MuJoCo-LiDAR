import time
import mujoco
import numpy as np
from mujoco_lidar import MjLidarWrapper, scan_gen

# Load model
model_path = "models/scene_jax.xml"
mj_model = mujoco.MjModel.from_xml_path(model_path)
mj_data = mujoco.MjData(mj_model)

# Initialize LiDAR with JAX backend
print("Initializing JAX LiDAR...")
lidar = MjLidarWrapper(mj_model, site_name="lidar_site", backend="jax")

# Generate scan pattern
generator = scan_gen.LivoxGenerator("mid360")
theta, phi = generator.sample_ray_angles()
print(f"Number of rays: {len(theta)}")

# Run scan
print("Running scan...")
start = time.time()
ranges = lidar.trace_rays(mj_data, theta, phi)
end = time.time()

print(f"Scan time: {end - start:.4f}s")
print(f"Output shape: {ranges.shape}")
print(f"Sample ranges: {ranges[:10]}")
