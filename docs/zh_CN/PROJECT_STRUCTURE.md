# 项目结构

```
MuJoCo-LiDAR/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                    # CI/CD 流水线
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml            # Bug 报告模板
│   │   └── feature_request.yml       # 功能请求模板
│   └── pull_request_template.md      # PR 模板
│
├── benchmarks/                       # 性能基准测试
│   ├── __init__.py
│   ├── benchmark_core.py             # 核心基准套件
│   ├── check_regression.py           # 回归检测
│   └── baselines/                    # 性能基线数据
│       └── baseline.json
│
├── docs/                             # 文档
│   ├── en/                           # 英文文档
│   │   ├── API.md
│   │   ├── INSTALLATION.md
│   │   ├── USAGE.md
│   │   ├── DEVELOPMENT.md
│   │   └── PROJECT_STRUCTURE.md
│   └── zh_CN/                        # 中文文档
│       ├── API.md
│       ├── INSTALLATION.md
│       ├── USAGE.md
│       ├── DEVELOPMENT.md
│       └── PROJECT_STRUCTURE.md
│
├── examples/                         # 使用示例
│   ├── example_*.py                  # 基础示例
│   ├── unitree_*.py                  # 机器人示例
│   └── test/                         # 测试脚本
│
├── src/                              # 源码（src 布局）
│   └── mujoco_lidar/
│       ├── __init__.py
│       ├── lidar_wrapper.py          # 主 Wrapper
│       ├── scan_gen.py               # 扫描模式生成器
│       ├── core_cpu/                 # CPU 后端
│       ├── core_ti/                  # Taichi 后端
│       └── core_jax/                 # JAX 后端
│
├── tests/                            # 测试套件
│   ├── conftest.py                   # pytest fixtures
│   ├── test_core.py                  # 核心功能测试
│   ├── test_cpu_backend.py           # CPU 后端测试
│   ├── test_scan_gen.py              # 扫描模式测试
│   └── test_integration.py          # 集成测试
│
├── .gitignore
├── .pre-commit-config.yaml           # Pre-commit hooks
├── CLAUDE.md                         # 开发规范
├── LICENSE
├── Makefile                          # 开发命令
├── pyproject.toml                    # 项目配置
├── README.md                         # 英文 README
└── README_zh.md                      # 中文 README
```

## 关键目录说明

### `/src/mujoco_lidar`
采用 src 布局，确保测试时使用已安装的包，而非本地目录，避免环境污染。

### `/benchmarks`
性能测试与回归检测。运行 `make benchmark` 执行，回归阈值为 5%。

### `/tests`
覆盖所有后端和核心功能的测试套件。运行 `make test` 执行。

### `/docs`
分语言存放文档：`en/` 为英文，`zh_CN/` 为中文。

### `/.github`
CI/CD 工作流、Issue 模板和 PR 模板，用于自动化质量检查。
