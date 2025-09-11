from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from helper import downsample_spherical_points_vectorized
import pandas as pd
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

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 2, gymapi.SIM_FLEX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()

# create the ground plane
gym.add_ground(sim, plane_params)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

asset_root = "isaacgym/assets"
asset_file = "resources/robots/go2/urdf/go2.urdf"

asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = True
asset_options.disable_gravity = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.4, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

env = gym.create_env(sim, lower, upper, 8)
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(5, 2, 5), gymapi.Vec3(0, 1, 0))

actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)



# load the files
TOTOAL = 100
pt_cloud = np.zeros((TOTOAL, 24000, 3))
for i in range(TOTOAL):
    df = pd.read_csv(f'/home/yiyi/Data/github/MuJoCo-LiDAR/mujoco_lidar/{i}.csv', header=None)
    pt_cloud[i] = df.to_numpy()

i = 0
while not gym.query_viewer_has_closed(viewer):
    gym.clear_lines(viewer)
    
    # Downsampler
    if i >= 100: 
        break
    draw_lidar_vis(gym, viewer, env, pt_cloud[i])
    i += 1
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
