# MuJoCo-LiDAR: High-Performance LiDAR Simulation Based on MuJoCo

A high-performance LiDAR simulation tool based on MuJoCo, supporting CPU, Taichi, and JAX backends.

<p align="center">
  <img src="./assets/go2.png" width="49%" />
  <img src="./assets/g1.png" width="49%" />
</p>
<p align="center">
  <img src="./assets/g1_native.png" width="32%" />
  <img src="./assets/go2_native.png" width="32%" />
  <img src="./assets/lidar_rviz.png" width="33%" />
</p>

[中文文档](README_zh.md)

## 🌟 Features

- **Multi-Backend Support**:
  - **CPU Backend**: Based on MuJoCo's native `mj_multiRay` function, no GPU required, simple and easy to use.
  - **Taichi Backend**: Utilizes Taichi for efficient GPU parallel computing, supports mesh scenes with millions of faces.
  - **JAX Backend**: Utilizes JAX for GPU parallel computing and MJX integration.
- **High Performance**: GPU-accelerated backends can generate 1 million+ rays in milliseconds.
- **Dynamic Scenes**: Supports real-time BVH construction (Taichi backend) for dynamic scenes.
- **Multiple LiDAR Models**: Supports various scanning patterns:
  - Livox non-repetitive scanning modes: mid360, mid70, mid40, tele, avia
  - Velodyne HDL-64E, VLP-32C
  - Ouster OS-128
  - Customizable grid scanning patterns
- **Accurate Physical Simulation**: Ray tracing for MuJoCo geometry types.
- **Flexible API**: Provides unified Wrapper interface.
- **ROS Integration**: Ready-to-use ROS1 and ROS2 examples.

## 🔧 Installation

### System Requirements

**Basic Dependencies:**
- Python >= 3.8
- MuJoCo >= 3.2.0
- NumPy >= 1.20.0

**Optional Backend Dependencies:**
- **Taichi**: `taichi >= 1.6.0`, `tibvh`
- **JAX**: `jax`, `jaxlib`

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# 1. Install basic dependencies (CPU backend)
pip install -e .

# 2. Install Taichi backend dependencies
pip install -e ".[taichi]"

# 3. Install JAX backend dependencies
pip install -e ".[jax]"
```

## 📚 Usage Examples

[ROS Integration](#-ros-integration) provides quick start examples for ROS1/2, and [Unitree Go2/G1](#-more-examples).

MuJoCo-LiDAR provides usage approaches and backend options:

### Backend Selection

1. **CPU Backend**:
   - Advantages: No GPU required, fewer dependencies.
   - Use Cases: Simple scenes, fewer rays (<10000).
   - Performance: Uses MuJoCo's native `mj_multiRay`.

2. **Taichi Backend**:
   - Advantages: High performance, supports complex Mesh scenes.
   - Use Cases: Complex scenes, large number of rays, Mesh geometries.
   - Performance: GPU parallel computing with BVH acceleration.

3. **JAX Backend**:
   - Advantages: High performance, potential for batch simulation.
   - Use Cases: Research involving JAX/MJX, simple geometric scenes (Primitives).
   - Note: Does not support Mesh geometries currently.

### Approach: Using Wrapper (Recommended)

The Wrapper approach provides a unified interface that automatically handles CPU and GPU backend differences. This is the **recommended approach**.

#### Example 1: CPU Backend + Wrapper (Scene Defined via String)

```python
import time
import mujoco
import mujoco.viewer

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

# Define a simple MuJoCo scene
simple_demo_scene = """
<mujoco model="simple_demo">
    <worldbody>
        <!-- Ground + Four Walls -->
        <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.2 0.9 0.9 1"/>
        <geom name="wall1" type="box" size="1e-3 3 1" pos=" 3 0 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall2" type="box" size="1e-3 3 1" pos="-3 0 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall3" type="box" size="3 1e-3 1" pos="0  3 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall4" type="box" size="3 1e-3 1" pos="0 -3 1" rgba="0.9 0.9 0.9 1"/>

        <!-- Various Geometries -->
        <geom name="box1" type="box" size="0.5 0.5 0.5" pos="2 0 0.5" euler="45 -45 0" rgba="1 0 0 1"/>
        <geom name="sphere1" type="sphere" size="0.5" pos="0 2 0.5" rgba="0 1 0 1"/>
        <geom name="cylinder1" type="cylinder" size="0.4 0.6" pos="0 -2 0.4" euler="0 90 0" rgba="0 0 1 1"/>
        
        <!-- LiDAR Site -->
        <body name="lidar_base" pos="0 0 1" quat="1 0 0 0" mocap="true">
            <inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>
            <site name="lidar_site" size="0.001" type='sphere'/>
            <geom type="box" size="0.1 0.1 0.1" density="0" contype="0" conaffinity="0" rgba="0.3 0.6 0.3 0.2"/>
        </body>
    </worldbody>
</mujoco>
"""

# Create MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_string(simple_demo_scene)
mj_data = mujoco.MjData(mj_model)

# Generate scan pattern
rays_theta, rays_phi = scan_gen.generate_grid_scan_pattern(num_ray_cols=64, num_ray_rows=16)

# Get body ID to exclude (avoid LiDAR detecting itself)
exclude_body_id = mj_model.body("lidar_base").id

# Create CPU backend LiDAR sensor
lidar = MjLidarWrapper(
    mj_model, 
    site_name="lidar_site",
    backend="cpu",  # Use CPU backend
    cutoff_dist=50.0,  # Maximum detection distance of 50 meters
    args={'bodyexclude': exclude_body_id}  # CPU backend specific parameter: exclude body
)

# Use in simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        
        # Perform ray tracing (Wrapper automatically handles pose updates)
        lidar.trace_rays(mj_data, rays_theta, rays_phi)
        
        # Get point cloud data (in local coordinate system)
        points = lidar.get_hit_points()  # shape: (N, 3)
        distances = lidar.get_distances()  # shape: (N,)
        
        time.sleep(1./60.)
```

#### Example 2: Taichi Backend + Wrapper (Loading from MJCF File)

```python
import mujoco
from mujoco_lidar import MjLidarWrapper, scan_gen

# Load MuJoCo model from file
mj_model = mujoco.MjModel.from_xml_path("path/to/your/model.xml")
mj_data = mujoco.MjData(mj_model)

# Generate scan pattern (using Velodyne HDL-64)
rays_theta, rays_phi = scan_gen.generate_HDL64()

# Create Taichi backend LiDAR sensor
lidar = MjLidarWrapper(
    mj_model,
    site_name="lidar_site",
    backend="taichi",  # Use Taichi backend
    cutoff_dist=100.0,
    args={
        'max_candidates': 64,  # Taichi backend specific parameter: BVH candidate nodes
        'ti_init_args': {'device_memory_GB': 4.0}  # Taichi initialization parameters
    }
)

# Simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        
        # Taichi backend usage is the same as CPU
        lidar.trace_rays(mj_data, rays_theta, rays_phi)
        points = lidar.get_hit_points()
```

#### Wrapper Parameter Description

```python
MjLidarWrapper(
    mj_model,           # MuJoCo model
    site_name,          # LiDAR site name
    backend="cpu",      # "cpu" or "taichi"
    cutoff_dist=100.0,  # Maximum detection distance (meters)
    args={}             # Backend-specific parameters
)

# CPU backend parameters (args)
{
    'geomgroup': None,      # Geometry group filter (0-5, None means all)
    'bodyexclude': -1       # Body ID to exclude (-1 means no exclusion)
}

# Taichi backend parameters (args)
{
    'max_candidates': 32,   # Maximum BVH candidate nodes (16-128)
    'ti_init_args': {}      # Taichi initialization parameters
}
```

### Approach 2: Using Core Directly (Advanced Users)

The Core approach provides direct access to low-level APIs, suitable for advanced users who need fine-grained control.

#### Example 3: CPU Core Approach

```python
import numpy as np
import mujoco
from mujoco_lidar.core_cpu.mjlidar_cpu import MjLidarCPU
from mujoco_lidar import scan_gen

# Create model
mj_model = mujoco.MjModel.from_xml_string(xml_string)
mj_data = mujoco.MjData(mj_model)

# Generate scan pattern
rays_theta, rays_phi = scan_gen.generate_grid_scan_pattern(64, 16)

# Create CPU core instance
lidar_cpu = MjLidarCPU(
    mj_model,
    cutoff_dist=50.0,
    geomgroup=None,      # Detect all geometry groups
    bodyexclude=-1       # Don't exclude any body
)

# Simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        
        # Manually construct 4x4 pose matrix
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, 3] = mj_data.site("lidar_site").xpos
        pose_4x4[:3, :3] = mj_data.site("lidar_site").xmat.reshape(3, 3)
        
        # Update data and perform ray tracing
        lidar_cpu.update(mj_data)
        lidar_cpu.trace_rays(pose_4x4, rays_theta, rays_phi)
        
        # Get results
        points = lidar_cpu.get_hit_points()
        distances = lidar_cpu.get_distances()
```

#### Example 4: Taichi Core Approach

```python
import numpy as np
import mujoco
import taichi as ti
from mujoco_lidar.core_ti.mjlidar_ti import MjLidarTi
from mujoco_lidar import scan_gen_livox_ti

# Initialize Taichi (must be done before creating MjLidarTi)
ti.init(arch=ti.gpu, device_memory_GB=4.0)

# Create model
mj_model = mujoco.MjModel.from_xml_string(xml_string)
mj_data = mujoco.MjData(mj_model)

# Use Livox scan pattern (Taichi optimized version)
livox_gen = scan_gen_livox_ti.LivoxGeneratorTi("mid360")

# Create Taichi core instance
lidar_ti = MjLidarTi(
    mj_model,
    cutoff_dist=100.0,
    max_candidates=64  # BVH candidate nodes
)

# Get ray angles in Taichi format
rays_theta_ti, rays_phi_ti = livox_gen.sample_ray_angles_ti()

# Simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        
        # Manually construct pose matrix
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, 3] = mj_data.site("lidar_site").xpos
        pose_4x4[:3, :3] = mj_data.site("lidar_site").xmat.reshape(3, 3)
        
        # Update BVH and perform ray tracing
        lidar_ti.update(mj_data)
        lidar_ti.trace_rays(pose_4x4, rays_theta_ti, rays_phi_ti)
        
        # For Livox, resample angles each time
        rays_theta_ti, rays_phi_ti = livox_gen.sample_ray_angles_ti()
        
        # Get results (copy from Taichi to CPU)
        points = lidar_ti.get_hit_points()  # Returns numpy array
        distances = lidar_ti.get_distances()
```

## 🤖 ROS Integration

MuJoCo-LiDAR provides complete ROS1 and ROS2 integration examples, supporting point cloud publishing and scene visualization.

### ROS1 Example

ROS1 related dependencies need to be installed in advance

```bash
# First terminal: Start ROS core
roscore

# Second terminal: Run LiDAR simulation (using Taichi backend). RViz will be automatically launched.
python examples/lidar_vis_ros1_wrapper.py --lidar mid360 --rate 12
```

### ROS2 Examples

**Approach 1: Using Wrapper (Recommended)**

```bash
# Run LiDAR simulation. RViz2 will be automatically launched.
python examples/lidar_vis_ros2_wrapper.py --lidar mid360 --rate 12
```

**Approach 2: Using Core (Advanced)**

```bash
# Using low-level Taichi Core API. RViz2 will be automatically launched.
python examples/lidar_vis_ros2.py --lidar mid360 --rate 12
```

### ROS Example Command Line Arguments

Both ROS examples support the following command line arguments:

```bash
python examples/lidar_vis_ros2_wrapper.py [options]

Options:
  --lidar MODEL      Specify LiDAR model, available values:
                     - Livox series: avia, mid40, mid70, mid360, tele
                     - Velodyne series: HDL64, vlp32
                     - Ouster series: os128
                     - Custom: custom
                     Default: mid360
  --verbose          Show detailed output, including position, orientation, and performance statistics
  --rate HZ          Set point cloud publishing rate (Hz), default: 12
```

**Usage Examples:**

```bash
# Use HDL64 LiDAR, enable verbose output, set publishing rate to 10Hz
python examples/lidar_vis_ros2_wrapper.py --lidar HDL64 --verbose --rate 10

# Use Velodyne VLP-32, default rate
python examples/lidar_vis_ros2_wrapper.py --lidar vlp32

# Use custom scan pattern
python examples/lidar_vis_ros2_wrapper.py --lidar custom

```

### Keyboard Controls

In ROS examples, you can control the LiDAR's position and orientation using the keyboard:

**Movement Controls:**
- `W`: Move forward
- `S`: Move backward
- `A`: Move left
- `D`: Move right
- `Q`: Move up
- `E`: Move down

**Orientation Controls:**
- `↑`: Pitch up
- `↓`: Pitch down
- `←`: Yaw left
- `→`: Yaw right

**Other:**
- `ESC`: Exit program

### ROS Topics

Example programs publish the following ROS topics:

| Topic Name | Message Type | Description |
|-----------|-------------|-------------|
| `/lidar_points` | `sensor_msgs/PointCloud2` | LiDAR point cloud data |
| `/mujoco_scene` | `visualization_msgs/MarkerArray` | MuJoCo scene geometry visualization |
| `/tf` | `tf2_msgs/TFMessage` | LiDAR coordinate transforms |

### Wrapper vs Core in ROS

**`lidar_vis_ros2_wrapper.py` (Wrapper Approach)**:
- Uses `MjLidarWrapper` class
- Automatically handles data format conversion (numpy ↔ Taichi)
- More concise code, easier to maintain
- Suitable for most application scenarios

```python
from mujoco_lidar import MjLidarWrapper

# Create Wrapper instance
lidar = MjLidarWrapper(mj_model, site_name="lidar_site", backend="taichi")

# Simple call
lidar.trace_rays(mj_data, rays_theta, rays_phi)
points = lidar.get_hit_points()  # Automatically returns numpy array
```

**`lidar_vis_ros2.py` (Core Approach)**:
- Directly uses `MjLidarTi` class
- Need to manually manage Taichi data format
- Need to manually construct 4x4 pose matrix
- More room for performance optimization, suitable for advanced users

```python
from mujoco_lidar.core_ti.mjlidar_ti import MjLidarTi
import taichi as ti

# Must initialize Taichi first
ti.init(arch=ti.gpu)

# Create Core instance
lidar = MjLidarTi(mj_model)

# Need Taichi ndarray format
rays_theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
rays_phi_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
rays_theta_ti.from_numpy(rays_theta)
rays_phi_ti.from_numpy(rays_phi)

# Manually construct pose matrix
pose_4x4 = np.eye(4, dtype=np.float32)
pose_4x4[:3, 3] = mj_data.site("lidar_site").xpos
pose_4x4[:3, :3] = mj_data.site("lidar_site").xmat.reshape(3, 3)

# Call
lidar.update(mj_data)
lidar.trace_rays(pose_4x4, rays_theta_ti, rays_phi_ti)
points = lidar.get_hit_points()  # Copy from GPU to CPU
```

## 🤝 More Examples

We also provide ROS2 integration examples with Unitree Go2 quadruped robot and G1 humanoid robot.

```bash
# Install onnx runtime
pip install onnxruntime

# go2 example
python examples/unitree_go2_ros2.py --lidar mid360
# Choose other lidar, for example: --lidar airy

# g1 example
python examples/unitree_g1_ros2.py --lidar mid360
```

## ⚡ Performance Optimization and Best Practices

### 1. Reduce Ray Tracing Frequency

LiDAR doesn't need to run at the same frequency as physics simulation:

```python
lidar_rate = 10  # LiDAR at 10Hz
physics_rate = 60  # Physics simulation at 60Hz
step_cnt = 0

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # High-frequency physics simulation
        mujoco.mj_step(mj_model, mj_data)
        step_cnt += 1
        
        # Low-frequency LiDAR scanning
        if step_cnt % (physics_rate // lidar_rate) == 0:
            lidar.trace_rays(mj_data, rays_theta, rays_phi)
            points = lidar.get_hit_points()
```

### 2. Reuse Ray Angle Arrays

For fixed scan patterns (non-Livox), generate angle arrays only once:

```python
# ✅ Correct: Generate once outside loop
rays_theta, rays_phi = scan_gen.generate_HDL64()

while True:
    lidar.trace_rays(mj_data, rays_theta, rays_phi)

# ❌ Wrong: Regenerate every loop (wasteful)
while True:
    rays_theta, rays_phi = scan_gen.generate_HDL64()  # Unnecessary!
    lidar.trace_rays(mj_data, rays_theta, rays_phi)
```

### 3. Use Taichi Arrays with Taichi Backend

When using Taichi Core approach, avoid frequent numpy↔Taichi conversions:

```python
import taichi as ti

# ✅ Correct: Use Taichi ndarray
rays_theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
rays_phi_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
rays_theta_ti.from_numpy(rays_theta)  # Convert only once
rays_phi_ti.from_numpy(rays_phi)

while True:
    lidar.trace_rays(pose_4x4, rays_theta_ti, rays_phi_ti)  # Use directly

# ❌ Wrong: Convert every time (high overhead)
while True:
    theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
    theta_ti.from_numpy(rays_theta)  # Frequent conversion!
    # ...
```

### 4. Livox Scan Pattern Optimization

When using Taichi backend, for Livox non-repetitive scanning, use Taichi optimized version:

```python
from mujoco_lidar import scan_gen_livox_ti
import taichi as ti

ti.init(arch=ti.gpu)

# ✅ Taichi optimized version: Returns Taichi arrays directly, no conversion needed
livox_gen = scan_gen_livox_ti.LivoxGeneratorTi("mid360")
rays_theta_ti, rays_phi_ti = livox_gen.sample_ray_angles_ti()

# ❌ CPU version: Need numpy→Taichi conversion every time
livox_gen = scan_gen.LivoxGenerator("mid360")
rays_theta, rays_phi = livox_gen.sample_ray_angles()
# Still need to convert to Taichi format...
```

### 5. Properly Set Scene Complexity

- Remove geometries outside the field of view
- Use geomgroup to organize the scene
- Simplify geometry shapes of unimportant objects
- For mesh models, consider reducing face count

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📜 Citation

If you find MuJoCo-LiDAR useful in your research, please consider citing our work:

```bibtex
@article{jia2025discoverse,
    title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
    author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
    journal={arXiv preprint arXiv:2507.21981},
    year={2025},
    url={https://arxiv.org/abs/2507.21981}
}
```