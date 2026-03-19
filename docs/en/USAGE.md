# Usage Examples

## Backend Selection

### CPU Backend
- **Advantages**: No GPU required, fewer dependencies
- **Use Cases**: Simple scenes, fewer rays (<10000)
- **Performance**: Uses MuJoCo's native `mj_multiRay`

### Taichi Backend
- **Advantages**: High performance, supports complex Mesh and Hfield scenes
- **Use Cases**: Complex scenes, large number of rays, Mesh or Hfield geometries
- **Performance**: GPU parallel computing with BVH acceleration

### JAX Backend
- **Advantages**: High performance, supports **Batch Simulation**
- **Use Cases**: Research involving JAX/MJX, large-scale parallel simulation
- **Note**: Supports Primitives and Height Fields, no Mesh support currently

## Using Wrapper (Recommended)

The Wrapper provides a unified interface for all backends.

### Example: CPU Backend

```python
import mujoco
from mujoco_lidar import MjLidarWrapper, scan_gen

# Define scene
xml = """
<mujoco>
  <worldbody>
    <body pos="2 0 0.5">
      <geom type="box" size="0.5 0.5 0.5"/>
    </body>
    <site name="lidar_site" pos="0 0 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Create LiDAR
lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",
    cutoff_dist=50.0
)

# Generate scan pattern
theta, phi = scan_gen.generate_HDL64()

# Trace rays
ranges = lidar.trace_rays(data, theta, phi)
print(f"Scanned {len(ranges)} points")
```

### Example: Taichi Backend

```python
lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="taichi",
    cutoff_dist=50.0,
    args={
        'max_candidates': 64,
        'ti_init_args': {'device_memory_GB': 4}
    }
)

ranges = lidar.trace_rays(data, theta, phi)
```

### Example: JAX Backend (Batch)

```python
import jax.numpy as jnp

lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="jax",
    cutoff_dist=50.0
)

# Batch simulation
batch_data = jnp.stack([data.qpos for _ in range(10)])
ranges = lidar.trace_rays_batch(batch_data, theta, phi)
```

## Scan Patterns

```python
from mujoco_lidar import scan_gen

# Velodyne HDL-64E
theta, phi = scan_gen.generate_HDL64()

# Velodyne VLP-32C
theta, phi = scan_gen.generate_vlp32()

# Ouster OS-128
theta, phi = scan_gen.generate_os128()

# RoboSense Airy-96
theta, phi = scan_gen.generate_airy96()

# Custom grid
theta, phi = scan_gen.generate_grid_scan_pattern(
    num_ray_cols=100,
    num_ray_rows=50
)

# Livox patterns
from mujoco_lidar.scan_gen_livox_ti import LivoxGenerator
livox = LivoxGenerator(pattern="mid360")
theta, phi = livox.sample_ray_angles()
```

## Advanced Usage

### Body Exclusion (CPU)

```python
lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",
    args={'bodyexclude': robot_body_id}
)
```

### Geometry Group Filter (CPU)

```python
import numpy as np

lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",
    args={
        'geomgroup': np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    }
)
```

---

## Running Examples

All commands must be run from the **project root directory** (`MuJoCo-LiDAR/`).

### Install Dependencies

```bash
# Taichi backend + example dependencies
uv sync --extra dev --extra taichi
uv add "onnxruntime==1.19.2" "etils[epath]"
```

### Unitree Go2

```bash
# Default: Livox mid360, Taichi backend, walking
uv run python examples/unitree_go2.py

# Airy-96 LiDAR
uv run python examples/unitree_go2.py --lidar airy

# CPU backend (no GPU required)
uv run python examples/unitree_go2.py --backend cpu

# Static pose (no walking)
uv run python examples/unitree_go2.py --stand
```

### Unitree G1

```bash
# Default: Livox mid360, Taichi backend
uv run python examples/unitree_g1.py

# Airy-96 LiDAR
uv run python examples/unitree_g1.py --lidar airy
```

### ROS2 Integration

```bash
uv run python examples/lidar_vis_ros2.py
uv run python examples/unitree_go2_ros2.py
uv run python examples/unitree_g1_ros2.py
```
