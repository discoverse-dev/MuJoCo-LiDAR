# Installation Guide

## System Requirements

**Basic Dependencies:**
- Python >= 3.10 (3.10 – 3.13 supported)
- MuJoCo >= 3.2.0
- NumPy >= 1.20.0

**Optional Backend Dependencies:**
- **Taichi**: `taichi >= 1.6.0`, `tibvh >= 0.1.2`
- **JAX**: `jax[cuda12]`

## Quick Installation

### From PyPI

```bash
# Basic (CPU backend)
uv add mujoco-lidar

# Verify
uv run python -c "import mujoco_lidar; print(mujoco_lidar.__version__)"

# With Taichi backend
uv add "mujoco-lidar[taichi]"

# With JAX backend
uv add "mujoco-lidar[jax]"
```

### From Source

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# Install with dev dependencies
uv sync --extra dev
uv sync --extra dev --extra taichi # With Taichi backend
uv sync --extra dev --extra jax    # With JAX backend

# Run tests
uv run pytest tests/
```

## Backend Notes

- **CPU**: No GPU required, works out-of-the-box
- **Taichi**: Requires NVIDIA GPU with CUDA
- **JAX**: Supports batch environments, no Mesh support currently
