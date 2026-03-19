# Project Structure

```
MuJoCo-LiDAR/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                    # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml            # Bug report template
│   │   └── feature_request.yml       # Feature request template
│   └── pull_request_template.md      # PR template
│
├── benchmarks/                       # Performance benchmarks
│   ├── __init__.py
│   ├── benchmark_core.py             # Core benchmark suite
│   ├── check_regression.py           # Regression detection
│   └── baselines/                    # Performance baselines
│       └── baseline.json
│
├── docs/                             # Documentation
│   ├── en/                           # English docs
│   │   ├── API.md                    # API reference
│   │   ├── INSTALLATION.md           # Installation guide
│   │   ├── USAGE.md                  # Usage examples
│   │   ├── DEVELOPMENT.md            # Development guide
│   │   └── PROJECT_STRUCTURE.md      # This file
│   └── zh_CN/                        # Chinese docs
│       ├── API.md
│       ├── INSTALLATION.md
│       ├── USAGE.md
│       ├── DEVELOPMENT.md
│       └── PROJECT_STRUCTURE.md
│
├── examples/                         # Usage examples
│   ├── example_*.py                  # Basic examples
│   ├── unitree_*.py                  # Robot examples
│   └── test/                         # Test scripts
│
├── src/                              # Source code (src layout)
│   └── mujoco_lidar/
│       ├── __init__.py
│       ├── lidar_wrapper.py          # Main wrapper
│       ├── scan_gen.py               # Scan pattern generators
│       ├── core_cpu/                 # CPU backend
│       ├── core_ti/                  # Taichi backend
│       └── core_jax/                 # JAX backend
│
├── tests/                            # Test suite
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_core.py                  # Core functionality
│   ├── test_wrapper.py               # Wrapper tests
│   ├── test_raytracing.py            # Ray tracing tests
│   ├── test_scan_patterns.py         # Scan pattern tests
│   ├── test_cpu_backend.py           # CPU backend tests
│   └── test_integration.py           # Integration tests
│
├── .gitignore
├── .pre-commit-config.yaml           # Pre-commit hooks
├── CLAUDE.md                         # Development guidelines
├── CONTRIBUTING.md                   # Contribution guide
├── LICENSE
├── Makefile                          # Development commands
├── pyproject.toml                    # Project configuration
├── README.md                         # English README
└── README_zh.md                      # Chinese README
```

## Key Directories

### `/src/mujoco_lidar`
Source code using src layout for proper test isolation. Prevents importing uninstalled code during testing.

### `/benchmarks`
Performance testing and regression detection. Run with `make benchmark`.

### `/tests`
Comprehensive test suite covering all backends and features. Run with `make test`.

### `/docs`
Project documentation including development guides and API references.

### `/.github`
CI/CD workflows, issue templates, and PR templates for automated quality checks.

