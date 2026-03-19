# Development Guide

## Setup

```bash
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

uv sync --extra dev                              # CPU only
uv sync --extra dev --extra taichi               # with Taichi
uv sync --extra dev --extra taichi --extra jax   # all backends
```

## Running Tests

```bash
make test

# Specific file
uv run pytest tests/test_core.py -v

# With coverage
uv run pytest --cov=mujoco_lidar
```

## Running Benchmarks

```bash
make benchmark

# Check regression against baseline
uv run python benchmarks/check_regression.py
```

## Code Quality

```bash
make format   # Format code
make lint     # Check linting
make check    # lint + test
```

## Workflow

```bash
# 1. Create feature branch
git checkout -b feat/your-feature

# 2. Develop and test
make test

# 3. Lint
make lint

# 4. Commit (Conventional Commits)
git commit -m "feat: add new feature"

# 5. Push and open PR
git push origin feat/your-feature
```

## Commit Convention

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `style:` formatting
- `refactor:` refactoring
- `test:` tests
- `chore:` build/tooling

**Never commit directly to main.**

## Adding a New Backend

1. Create `src/mujoco_lidar/core_xxx/` directory
2. Implement `update(mj_data)`, `trace_rays(pose, theta, phi)`, `get_hit_points()`, `get_distances()`
3. Register in `lidar_wrapper.py`
4. Add tests in `tests/`
