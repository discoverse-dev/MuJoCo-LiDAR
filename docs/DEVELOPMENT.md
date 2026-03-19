# Development Guide

## Project Structure

```
MuJoCo-LiDAR/
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   └── ISSUE_TEMPLATE/     # Issue templates
├── benchmarks/             # Performance benchmarks
│   ├── benchmark_core.py   # Core benchmarks
│   ├── check_regression.py # Regression detection
│   └── baselines/          # Performance baselines
├── docs/                   # Documentation
├── examples/               # Usage examples
├── mujoco_lidar/          # Source code
│   ├── core_cpu/          # CPU backend
│   ├── core_taichi/       # Taichi backend
│   └── core_jax/          # JAX backend
├── tests/                 # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_core.py       # Core tests
│   ├── test_wrapper.py    # Wrapper tests
│   ├── test_raytracing.py # Ray tracing tests
│   └── test_scan_patterns.py # Scan pattern tests
└── pyproject.toml         # Project config
```

## Running Tests

```bash
# All tests
make test

# Specific test file
uv run pytest tests/test_core.py -v

# With coverage
uv run pytest --cov=mujoco_lidar
```

## Running Benchmarks

```bash
# Run benchmarks
uv run python benchmarks/benchmark_core.py

# Check for performance regression
uv run python benchmarks/check_regression.py
```

## Code Quality

```bash
# Format code
make format

# Check linting
make lint

# Run all checks
make check
```
