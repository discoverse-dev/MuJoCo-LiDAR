# Lazy import to avoid loading taichi when not needed
# These modules should only be imported when Taichi backend is explicitly requested

__all__ = [
    'mjlidar_ti',
    'MjLidarTi',
]

def __getattr__(name):
    """Lazy import for Taichi backend to avoid importing taichi unless needed."""
    if name == 'mjlidar_ti':
        from . import mjlidar_ti
        return mjlidar_ti
    elif name == 'MjLidarTi':
        from .mjlidar_ti import MjLidarTi
        return MjLidarTi
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
