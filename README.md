# MuJoCo-LiDAR: High-Performance LiDAR Simulation

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

[中文文档](README_zh.md) | [Installation](docs/en/INSTALLATION.md) | [Usage Guide](docs/en/USAGE.md) | [Development](docs/en/DEVELOPMENT.md) | [Contributing](CONTRIBUTING.md)

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

**From PyPI:**

```bash
# Basic (CPU only)
uv add mujoco-lidar

# With Taichi backend (GPU)
uv add "mujoco-lidar[taichi]"

# With JAX backend (GPU + batch)
uv add "mujoco-lidar[jax]"
```

**From Source:**

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

uv sync --extra dev                        # CPU only
uv sync --extra dev --extra taichi         # with Taichi backend
uv sync --extra dev --extra taichi --extra jax  # all backends
```

See [Installation Guide](docs/en/INSTALLATION.md) for details.

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

See [Usage Guide](docs/en/USAGE.md) for more examples.

## Performance

| Backend | Rays/sec | Hardware | Batch Support |
|---------|----------|----------|---------------|
| CPU     | ~9M      | Native   | No            |
| Taichi  | ~62M     | GPU      | Yes           |
| JAX     | ~231M    | GPU      | Yes           |

Run benchmarks: `make benchmark`

## Documentation

- [Installation Guide](docs/en/INSTALLATION.md) - Detailed installation instructions
- [Usage Guide](docs/en/USAGE.md) - Examples and tutorials
- [API Reference](docs/en/API.md) - Complete API documentation
- [Development Guide](docs/en/DEVELOPMENT.md) - Contributing and testing
- [Project Structure](docs/en/PROJECT_STRUCTURE.md) - Codebase organization

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
@article{jia2025discoverse,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      journal={arXiv preprint arXiv:2507.21981},
      year={2025},
      url={https://arxiv.org/abs/2507.21981}
}
```
