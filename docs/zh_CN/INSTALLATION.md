# 安装指南

## 系统要求

**基础依赖：**
- Python >= 3.10
- MuJoCo >= 3.2.0
- NumPy >= 1.20.0

**可选后端依赖：**
- **Taichi**：`taichi >= 1.6.0`，`tibvh >= 0.1.2`
- **JAX**：`jax[cuda12]`

## 快速安装

### 从 PyPI 安装

```bash
# 基础安装（仅 CPU 后端）
uv add mujoco-lidar

# 验证安装
uv run python -c "import mujoco_lidar; print(mujoco_lidar.__version__)"

# 安装 Taichi 后端
uv add "mujoco-lidar[taichi]"

# 安装 JAX 后端
uv add "mujoco-lidar[jax]"
```

### 从源码安装

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# 安装开发依赖
uv sync --extra dev
uv sync --extra dev --extra taichi # 安装 Taichi 后端
uv sync --extra dev --extra jax    # 安装 JAX 后端

# 运行测试
uv run pytest tests/
```

## 后端说明

- **CPU**：无需 GPU，开箱即用
- **Taichi**：需要 NVIDIA GPU 和 CUDA
- **JAX**：支持批量仿真，暂不支持 Mesh 几何体
