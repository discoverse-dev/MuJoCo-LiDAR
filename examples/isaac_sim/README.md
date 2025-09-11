# For testing only

Install Isaac Sim and IsaacLab following: https://github.com/isaac-sim/IsaacLab

```bash
uv pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

Run example with IsaacSim
```bash
python examples/isaac_sim/mujoco_isaacSim_lidar.py --obj-path models/test_ply.obj --rate 20
```

TL;DR:
This is what's drawing the points in IsaacSim
```python
# DONOT MOVE THIS TO THE FRONT OF THE SCRIPT, ISSAC WOULD CRASH
from isaacsim.util.debug_draw import _debug_draw
draw_iface = _debug_draw.acquire_debug_draw_interface()
points_cloud = pointcloud_data
colors_cloud = np.tile(np.array(color), (pointcloud_data.shape[0], 1))
sizes_cloud = np.full((pointcloud_data.shape[0],), size)
if clear_existing:

    self._debug_draw_clear_points()

draw_iface.draw_points(points_cloud, colors_cloud, sizes_cloud)


```