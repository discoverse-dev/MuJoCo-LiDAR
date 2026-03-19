# JAX backend for MuJoCo-LiDAR
# Lazy import to avoid loading jax when not needed

__all__ = [
    "mjlidar_jax",
    "MjLidarJax",
]

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import for JAX backend to avoid importing jax unless needed."""
    if name == "mjlidar_jax":
        from . import mjlidar_jax

        return mjlidar_jax
    elif name == "MjLidarJax":
        from .mjlidar_jax import MjLidarJax

        return MjLidarJax
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
