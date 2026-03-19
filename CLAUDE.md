# MuJoCo-LiDAR Project Instructions

## Execution Commands

- ALWAYS use `uv run <command>` to execute Python code
- NEVER use `python` or `python3` directly

## Code Quality Standards

When writing or modifying code, you MUST:

1. **Minimize code**: Write the absolute minimum code needed. No extra features, no "nice-to-haves"
2. **Delete ruthlessly**: Remove unused code immediately. Never comment out code—delete it
3. **Refactor immediately**: If you spot poor design while working, fix it before adding features
4. **Prioritize clarity**: Code must be self-explanatory. Clarity > brevity > flexibility
5. **Optimize for performance**: This is GPU-intensive research code. Slow code wastes expensive compute time

## Design Priorities (in order)

1. Clear, understandable code structure
2. Minimal, focused implementation
3. Good architecture over quick functionality
4. Performance optimization
5. Testability over debuggability

## What NOT to do

- Do NOT add defensive code for impossible scenarios
- Do NOT create abstractions for single-use cases
- Do NOT add features beyond what's explicitly requested
- Do NOT leave commented-out code
- Do NOT add unnecessary error handling or validation in internal code

## Git Commit Guidelines

Use Conventional Commits format:
- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 代码格式化
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建/工具配置

**重要规则:**
- 及时 commit，保持提交历史清晰
- **永远不在 main 分支 commit**
- 始终在 feature 分支工作
- PR 合并后删除 feature 分支

---

## CI/CD 系统化方案

### 现状评估

**当前状态:**
- ✅ CI/CD 流程已建立
- ✅ 自动化测试（17 个测试用例）
- ✅ 代码质量检查（ruff）
- ⚠️ 手动发布到 PyPI（暂不自动化）
- ✅ 性能回归检测
- ✅ 多后端隔离测试

**风险:**
- 破坏性变更直接进入 main
- 性能退化无法及时发现
- 依赖冲突（taichi/jax）未验证
- 发布质量无保障

### 优化方案

#### 1. 代码质量门禁（Pre-commit）

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
```

**原则**: 垃圾代码不进仓库，格式化自动完成。

#### 2. CI 流水线架构

**核心流程:**
```
PR → Lint → Type Check → Unit Tests → Integration Tests → Performance Benchmark → Approve
```

**关键设计:**
- **矩阵测试**: Python 3.9/3.10/3.11 × CPU/Taichi/JAX
- **隔离环境**: 每个后端独立测试，避免依赖污染
- **快速失败**: Lint 失败立即终止，节省 CI 时间
- **性能基线**: 每次 PR 对比 main 分支性能

#### 3. GitHub Actions 配置

**`.github/workflows/ci.yml`** - 主流程:
```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install ruff mypy
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy mujoco_lidar --ignore-missing-imports

  test-cpu:
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e .
      - run: pip install pytest pytest-cov
      - run: pytest tests/ -v --cov=mujoco_lidar --cov-report=xml
      - uses: codecov/codecov-action@v4

  test-taichi:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e .[taichi]
      - run: pytest tests/test_taichi*.py -v

  test-jax:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e .[jax]
      - run: pytest tests/test_jax*.py -v

  benchmark:
    needs: [test-cpu, test-taichi, test-jax]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e .[taichi]
      - run: python benchmarks/run.py --compare-with=main
      - run: python benchmarks/check_regression.py
```

**`.github/workflows/release.yml`** - 发布流程:
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install build twine
      - run: python -m build
      - run: twine check dist/*
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

#### 4. 测试架构

**目录结构:**
```
tests/
├── conftest.py              # pytest fixtures
├── test_core.py             # 核心功能测试
├── test_cpu_backend.py      # CPU 后端
├── test_taichi_backend.py   # Taichi 后端
├── test_jax_backend.py      # JAX 后端
├── test_scan_patterns.py    # 扫描模式
└── test_integration.py      # 集成测试

benchmarks/
├── run.py                   # 性能测试入口
├── check_regression.py      # 回归检测
└── baselines/               # 性能基线数据
```

**测试原则:**
- **单元测试**: 覆盖核心算法逻辑
- **集成测试**: 验证多后端一致性
- **性能测试**: 监控关键路径（ray generation, BVH build）
- **回归测试**: 阻止性能下降 >5%

#### 5. 性能监控

**关键指标:**
- Ray generation speed (rays/ms)
- BVH construction time (Taichi)
- Memory usage per backend
- Batch processing throughput

**实现:**
```python
# benchmarks/run.py
import time
import numpy as np
from mujoco_lidar import LidarWrapper

def benchmark_ray_generation(backend, n_rays=100000):
    lidar = LidarWrapper(backend=backend)
    start = time.perf_counter()
    for _ in range(10):
        lidar.scan()
    elapsed = time.perf_counter() - start
    return n_rays * 10 / elapsed  # rays/sec

# 对比 main 分支基线
# 如果下降 >5%，CI 失败
```

#### 6. 发布流程

**版本管理:**
- 语义化版本: `v0.2.5` → `v0.3.0` (breaking) / `v0.2.6` (feature) / `v0.2.5.1` (fix)
- 自动生成 CHANGELOG
- Git tag 触发自动发布

**发布检查清单:**
1. ✅ 所有 CI 通过
2. ✅ 性能无回归
3. ✅ 文档已更新
4. ✅ CHANGELOG 已生成
5. ✅ 版本号已更新

#### 7. 依赖管理

**问题**: Taichi 和 JAX 都是大型依赖，不应强制安装。

**方案:**
```toml
# pyproject.toml (已正确配置)
[project.optional-dependencies]
taichi = ["taichi>=1.6.0", "tibvh>=0.1.2"]
jax = ["jax[cuda12]"]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
```

**CI 中分别测试:**
- `pip install -e .` → 仅 CPU
- `pip install -e .[taichi]` → Taichi
- `pip install -e .[jax]` → JAX

#### 8. 文档自动化

**API 文档:**
```yaml
# .github/workflows/docs.yml
- run: pip install sphinx sphinx-rtd-theme
- run: sphinx-build -b html docs/ docs/_build
- uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./docs/_build
```

**README 同步:**
- 中英文文档一致性检查
- 示例代码可执行性验证

---

## 实施优先级

### 实施状态

**Phase 1: 基础设施** ✅
- ✅ `.pre-commit-config.yaml`
- ✅ `tests/` 目录（7 个测试文件）
- ✅ GitHub Actions CI

**Phase 2: 质量保障** ✅
- ✅ 17 个单元测试
- ✅ 集成测试（多后端）
- ⚠️ codecov（待配置）

**Phase 3: 性能监控** ✅
- ✅ 性能基准测试（benchmarks/）
- ✅ 回归检测（5% 阈值）
- ⚠️ 性能基线（首次运行时建立）

**Phase 4: 自动化发布** ⚠️
- 手动发布到 PyPI
- 手动维护 CHANGELOG

---

## 开发工作流

### 日常开发
```bash
# 1. 安装依赖
uv sync

# 2. 开发功能
git checkout -b feature/xxx

# 3. 本地测试
make test

# 4. 代码检查
make lint

# 5. 提交
git commit -m "feat: xxx"

# 6. 推送并创建 PR
git push origin feature/xxx
```

### PR 审查清单
- [ ] CI 全部通过
- [ ] 代码覆盖率未下降
- [ ] 性能无回归
- [ ] 文档已更新
- [ ] 符合设计原则

### 发布流程
```bash
# 1. 更新版本号
# mujoco_lidar/__init__.py: __version__ = "0.3.0"

# 2. 更新 CHANGELOG
# 记录 breaking changes / features / fixes

# 3. 创建 tag
git tag v0.3.0
git push origin v0.3.0

# 4. GitHub Actions 自动发布到 PyPI
```

---

## 工具配置

### Ruff (Linter + Formatter)
```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "A", "C4", "SIM"]
ignore = ["E501"]  # 行长度由 formatter 处理

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Mypy (Type Checker)
```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # 渐进式类型检查
```

### Pytest
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --strict-markers"
```

---

## 性能要求

### 基准指标（参考）
- CPU backend: >10K rays/sec
- Taichi backend: >1M rays/sec (GPU)
- JAX backend: >500K rays/sec (GPU)
- BVH build: <100ms (1M triangles)

### 回归阈值
- 性能下降 >5%: CI 失败
- 内存增长 >10%: 警告
- 测试时间增长 >20%: 需优化

---

## 常见问题

### Q: 为什么不用 tox？
A: GitHub Actions 矩阵测试更直观，且与 CI 环境一致。

### Q: 为什么选择 Ruff 而不是 Black + Flake8？
A: Ruff 速度快 10-100 倍，功能覆盖全面，单一工具减少配置复杂度。

### Q: 性能测试需要 GPU 吗？
A: CPU backend 测试不需要。Taichi/JAX 可以用 GitHub Actions GPU runner（付费）或本地测试。

### Q: 如何处理 breaking changes？
A: 主版本号递增（0.x.x → 1.0.0），在 CHANGELOG 中明确标注，提供迁移指南。

---

## 参考资源

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [PyPI 发布指南](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Ruff 配置](https://docs.astral.sh/ruff/)
- [Pytest 最佳实践](https://docs.pytest.org/en/stable/goodpractices.html)
