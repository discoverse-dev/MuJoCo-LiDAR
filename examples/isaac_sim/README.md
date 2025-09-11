# For testing only

Install Isaac Sim and IsaacLab following: https://github.com/isaac-sim/IsaacLab

```bash
uv pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

Run example with IsaacSim
```bash
python examples/isaac_sim/mujoco_isaacSim_lidar.py --obj-path models/test_ply.obj --rate 20
```