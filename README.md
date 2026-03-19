# MuJoCo-LiDAR: High-Performance LiDAR Simulation

[![CI](https://github.com/TATP-233/MuJoCo-LiDAR/actions/workflows/ci.yml/badge.svg)](https://github.com/TATP-233/MuJoCo-LiDAR/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mujoco-lidar)](https://pypi.org/project/mujoco-lidar/)
[![Python](https://img.shields.io/pypi/pyversions/mujoco-lidar)](https://pypi.org/project/mujoco-lidar/)

High-performance LiDAR simulation for MuJoCo with CPU, Taichi, and JAX backends.

<p align="center">
  <img src="./assets/go2.png" width="49%" />
  <img src="./assets/g1.png" width="49%" />
</p>
<p align="center">
  <img src="./assets/g1_native.png" width="32%" />
  <img src="./assets/go2_native.png" width="32%" />
  <img src="./assets/lidar_rviz.png" width="33%" />
</p>

[中文文档](README_zh.md) | [Installation](docs/INSTALLATION.md) | [Usage Guide](docs/USAGE.md) | [Development](docs/DEVELOPMENT.md) | [Contributing](CONTRIBUTING.md)

## Features

- **Multi-Backend Support**:
  - **CPU**: MuJoCo native `mj_multiRay`, no GPU required
  - **Taichi**: GPU parallel computing, supports Mesh and Hfield
  - **JAX**: GPU + MJX integration, batch simulation support
- **High Performance**: 1M+ rays/sec on GPU, real-time BVH construction
- **Multiple LiDAR Models**: Velodyne (HDL-64E, VLP-32C), Livox (mid360, avia), Ouster (OS-128), custom patterns
- **ROS Integration**: Ready-to-use ROS1/ROS2 examples

## Quick Start

### Installation

```bash
# Basic (CPU only)
pip install mujoco-lidar

# With Taichi backend (GPU)
pip install mujoco-lidar[taichi]

# With JAX backend (GPU + batch)
pip install mujoco-lidar[jax]
```

See [Installation Guide](docs/INSTALLATION.md) for details.

### Basic Usage

```python
import mujoco
from mujoco_lidar import MjLidarWrapper, scan_gen

# Load model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Create LiDAR
lidar = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="cpu",  # or "taichi", "jax"
    cutoff_dist=50.0
)

# Generate scan pattern
theta, phi = scan_gen.generate_HDL64()

# Trace rays
ranges = lidar.trace_rays(data, theta, phi)
```

See [Usage Guide](docs/USAGE.md) for more examples.

## Performance

| Backend | Rays/sec | Hardware | Batch Support |
|---------|----------|----------|---------------|
| CPU     | ~10M     | Native   | No            |
| Taichi  | ~1000M   | GPU      | Yes           |
| JAX     | ~500M    | GPU      | Yes           |

Run benchmarks: `make benchmark`

## Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed installation instructions
- [Usage Guide](docs/USAGE.md) - Examples and tutorials
- [API Reference](docs/API.md) - Complete API documentation
- [Development Guide](docs/DEVELOPMENT.md) - Contributing and testing
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Codebase organization

## Examples

- [examples/example_native.py](examples/example_native.py) - CPU backend
- [examples/example_taichi.py](examples/example_taichi.py) - Taichi backend
- [examples/lidar_vis_ros2.py](examples/lidar_vis_ros2.py) - ROS2 integration
- [examples/unitree_go2_ros2.py](examples/unitree_go2_ros2.py) - Unitree Go2 robot

## Development

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR
uv sync --extra dev

make test      # Run tests
make lint      # Check code quality
make benchmark # Run performance tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{mujoco_lidar,
  title = {MuJoCo-LiDAR: High-Performance LiDAR Simulation},
  author = {Yufei Jia},
  year = {2024},
  url = {https://github.com/TATP-233/MuJoCo-LiDAR}
}
```
