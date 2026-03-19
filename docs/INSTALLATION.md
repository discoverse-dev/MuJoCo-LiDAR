# Installation Guide

## System Requirements

**Basic Dependencies:**
- Python >= 3.9
- MuJoCo >= 3.2.0
- NumPy >= 1.20.0

**Optional Backend Dependencies:**
- **Taichi**: `taichi >= 1.6.0`, `tibvh >= 0.1.2`
- **JAX**: `jax[cuda12]`

## Quick Installation

### From PyPI

```bash
# 1. Install basic dependencies (CPU backend)
pip install mujoco-lidar

# Verify installation
python -c "import mujoco_lidar; print(mujoco_lidar.__version__)"

# 2. (Optional) Install Taichi backend
pip install mujoco-lidar[taichi]

# Verify Taichi
python -c "import taichi as ti; ti.init(ti.gpu)"

# 3. (Optional) Install JAX backend
pip install mujoco-lidar[jax]

# Verify JAX
python -c "import jax; print(jax.default_backend())"
```

### From Source

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# Basic installation
pip install -e .

# With Taichi
pip install -e ".[taichi]"

# With JAX
pip install -e ".[jax]"

# Development
uv sync --extra dev
```

## Backend Notes

- **CPU**: No GPU required, works out-of-the-box
- **Taichi**: Requires NVIDIA GPU with CUDA
- **JAX**: Supports batch environments, no Mesh support currently
