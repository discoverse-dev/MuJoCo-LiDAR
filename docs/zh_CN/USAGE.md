# 使用示例

## 后端选择

### CPU 后端
- **优势**：无需 GPU，依赖少
- **适用场景**：简单场景，射线数量较少（< 10000）
- **性能**：使用 MuJoCo 原生 `mj_multiRay`

### Taichi 后端
- **优势**：高性能，支持复杂 Mesh 和 Hfield 场景
- **适用场景**：复杂场景，大量射线，Mesh 或 Hfield 几何体
- **性能**：GPU 并行计算 + BVH 加速

### JAX 后端
- **优势**：高性能，支持**批量仿真**
- **适用场景**：JAX/MJX 研究，大规模并行仿真
- **注意**：支持基本几何体和高度场，暂不支持 Mesh

---

## 使用 Wrapper（推荐）

Wrapper 为所有后端提供统一接口。

### CPU 后端示例

```python
import mujoco
from mujoco_lidar import MjLidarWrapper, scan_gen

# 定义场景
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

# 创建 LiDAR
lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",
    cutoff_dist=50.0
)

# 生成扫描模式
theta, phi = scan_gen.generate_HDL64()

# 执行射线追踪
ranges = lidar.trace_rays(data, theta, phi)
print(f"扫描点数：{len(ranges)}")
```

### Taichi 后端示例

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

### JAX 后端示例（批量仿真）

```python
lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="jax",
    cutoff_dist=50.0
)

ranges = lidar.trace_rays(data, theta, phi)
```

---

## 扫描模式

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

# 自定义网格
theta, phi = scan_gen.generate_grid_scan_pattern(
    num_ray_cols=100,
    num_ray_rows=50
)

# Livox 模式
from mujoco_lidar.scan_gen_livox_ti import LivoxGenerator
livox = LivoxGenerator(pattern="mid360")
theta, phi = livox.sample_ray_angles()
```

---

## 高级用法

### 排除机器人本体（CPU）

```python
robot_body_id = model.body("robot").id

lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",
    args={'bodyexclude': robot_body_id}
)
```

### 几何组过滤（CPU）

```python
import numpy as np

lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",
    args={
        'geomgroup': np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        # 仅检测第 0、1、2 组几何体
    }
)
```

### 获取点云

```python
# 执行射线追踪
ranges = lidar.trace_rays(data, theta, phi)

# 获取局部坐标系下的点云（N×3）
hit_points = lidar.get_hit_points()

# 过滤未命中点（距离为 0）
valid_mask = ranges > 0
valid_points = hit_points[valid_mask]
```

---

## 运行示例

所有命令须在**项目根目录**（`MuJoCo-LiDAR/`）下执行。

### 安装依赖

```bash
uv sync --extra dev --extra taichi
uv sync --extra examples
```

### Unitree Go2

```bash
# 默认：Livox mid360，Taichi 后端，行走模式
uv run python examples/unitree_go2.py

# Airy-96 LiDAR
uv run python examples/unitree_go2.py --lidar airy

# CPU 后端（无需 GPU）
uv run python examples/unitree_go2.py --backend cpu

# 静止姿态（不行走）
uv run python examples/unitree_go2.py --stand
```

### Unitree G1

```bash
# 默认：Livox mid360，Taichi 后端
uv run python examples/unitree_g1.py

# Airy-96 LiDAR
uv run python examples/unitree_g1.py --lidar airy
```

### ROS2 集成

```bash
uv run python examples/lidar_vis_ros2.py
uv run python examples/unitree_go2_ros2.py
uv run python examples/unitree_g1_ros2.py
```
