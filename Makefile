.PHONY: help install test test-cov lint format check benchmark clean

UV_RUN := uv run --python python3.10
UV_RUN_DEV := $(UV_RUN) --extra dev

help:
	@echo "MuJoCo-LiDAR Development Commands"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run all tests"
	@echo "  make test-cov    - Run tests with coverage"
	@echo "  make lint        - Run linter"
	@echo "  make format      - Format code"
	@echo "  make check       - Run lint + test"
	@echo "  make benchmark   - Run performance benchmarks"
	@echo "  make clean       - Clean build artifacts"

install:
	uv sync --extra dev

test:
	$(UV_RUN_DEV) python -m pytest tests/ -v

test-cov:
	$(UV_RUN_DEV) python -m pytest tests/ -v --cov=mujoco_lidar --cov-report=html --cov-report=term

lint:
	$(UV_RUN_DEV) python -m ruff check .

format:
	$(UV_RUN_DEV) python -m ruff format .
	$(UV_RUN_DEV) python -m ruff check --fix .

check: lint test

benchmark:
	$(UV_RUN) python benchmarks/benchmark_core.py

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache htmlcov/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
