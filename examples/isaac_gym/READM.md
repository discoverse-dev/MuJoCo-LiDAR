# For testing only
1. Install isaacgym 
2. transfer ply file to obj file 
3. Create an venv in python 3.8
4. Install dependencies for this repo

```bash
python examples/isaac_gym/mujoco_isaacgym_lidar.py --obj-path models/test_ply.obj --backend gpu
```

TL;DR:
This is what's drawing the points in Isaacgym
```python
def draw_lidar_vis(gym, viewer, env, pts):
    """ Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
    """
    # draw height lines
    sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(0, 1, 0))
    # randomly generat
    for pt in pts:
        sphere_pose = gymapi.Transform(gymapi.Vec3(pt[0], pt[2], pt[1]), r=None)
        gymutil.draw_lines(sphere_geom, gym, viewer, env, sphere_pose)
```