# Contributing to MuJoCo-LiDAR

## Development Setup

```bash
# Clone repository
git clone https://github.com/TATP-233/MuJoCo-LiDAR.git
cd MuJoCo-LiDAR

# Install with dev dependencies
uv sync --extra dev

# Run tests
make test

# Run linter
make lint
```

## Code Quality

- All code must pass `ruff` linting
- Tests must pass before merging
- Follow existing code style

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Run `make check` locally
4. Push and create PR
5. Wait for CI to pass
