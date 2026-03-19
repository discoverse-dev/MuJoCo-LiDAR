# 开发指南

## 项目结构

```
MuJoCo-LiDAR/
├── .github/
│   ├── workflows/          # CI/CD 流水线
│   └── ISSUE_TEMPLATE/     # Issue 模板
├── benchmarks/             # 性能基准测试
│   ├── benchmark_core.py   # 核心基准
│   ├── check_regression.py # 回归检测
│   └── baselines/          # 性能基线数据
├── docs/
│   ├── en/                 # 英文文档
│   └── zh_CN/              # 中文文档
├── examples/               # 使用示例
├── src/
│   └── mujoco_lidar/       # 源码（src 布局）
│       ├── core_cpu/       # CPU 后端
│       ├── core_ti/        # Taichi 后端
│       └── core_jax/       # JAX 后端
├── tests/                  # 测试套件
│   ├── conftest.py         # pytest fixtures
│   ├── test_core.py        # 核心功能测试
│   ├── test_cpu_backend.py # CPU 后端测试
│   ├── test_scan_gen.py    # 扫描模式测试
│   └── test_integration.py # 集成测试
└── pyproject.toml          # 项目配置
```

## 环境搭建

```bash
# 克隆仓库
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# 安装开发依赖（需要 Python 3.10+）
uv sync --extra dev
```

## 运行测试

```bash
# 全部测试
make test

# 指定测试文件
uv run pytest tests/test_core.py -v

# 带覆盖率
uv run pytest --cov=mujoco_lidar
```

## 运行基准测试

```bash
# 执行性能测试
make benchmark

# 检查性能回归（与基线对比）
uv run python benchmarks/check_regression.py
```

## 代码质量

```bash
# 格式化代码
make format

# 检查 lint
make lint

# 运行全部检查
make check
```

## 开发工作流

```bash
# 1. 创建功能分支
git checkout -b feat/your-feature

# 2. 开发并测试
make test

# 3. 代码检查
make lint

# 4. 提交（遵循 Conventional Commits）
git commit -m "feat: 添加新功能"

# 5. 推送并创建 PR
git push origin feat/your-feature
```

## Commit 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 代码格式化
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建/工具配置

**重要规则：永远不在 main 分支直接提交。**

## 添加新后端

1. 在 `src/mujoco_lidar/` 下创建 `core_xxx/` 目录
2. 实现 `update(mj_data)`、`trace_rays(pose, theta, phi)`、`get_hit_points()`、`get_distances()` 方法
3. 在 `lidar_wrapper.py` 中注册新后端
4. 在 `tests/` 中添加对应测试
