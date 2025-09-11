import time
import argparse

from loop_rate_limiters import RateLimiter
from isaacgym import gymapi
from isaacgym import gymutil
import pandas as pd 
import sys
import os
import torch
# TODO: remove
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from mujoco_lidar import (
    LidarSensor, LivoxGenerator,
    generate_vlp32, generate_HDL64, generate_os128
)
from mujoco_lidar.mj_lidar_utils import create_demo_scene, KeyboardListener



def draw_lidar_points(gym, viewer, env, points):
    """ Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
    """
    # draw height lines
    
    sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(0, 1, 0))

    # randomly generat
    for pt in points:
        sphere_pose = gymapi.Transform(gymapi.Vec3(pt[0], pt[2], pt[1]), r=None)
        gymutil.draw_lines(sphere_geom, gym, viewer, env, sphere_pose)

def select_lidar_angles(args):
    """Select LiDAR angles based on args.lidar."""
    if args.lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
        livox_generator = LivoxGenerator(args.lidar)
        rays_theta, rays_phi = livox_generator.sample_ray_angles()
        use_livox_lidar = True
    elif args.lidar == "HDL64":
        rays_theta, rays_phi = generate_HDL64()
        use_livox_lidar = False
    elif args.lidar == "vlp32":
        rays_theta, rays_phi = generate_vlp32()
        use_livox_lidar = False
    elif args.lidar == "os128":
        rays_theta, rays_phi = generate_os128()
        use_livox_lidar = False
    else:
        raise ValueError(f"Unsupported LiDAR type: {args.lidar}")
    return rays_theta, rays_phi, use_livox_lidar

def main():
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR visualization (Isaac Gym)')
    parser.add_argument('--lidar', type=str, default='mid360',
                        choices=['avia', 'HAP', 'horizon', 'mid40', 'mid70', 'mid360', 'tele', 'HDL64', 'vlp32', 'os128'])
    parser.add_argument('--backend', type=str, default='gpu', choices=['cpu', 'gpu'], help='LiDAR backend (cpu/gpu)')
    parser.add_argument('--obj-path', type=str, help='OBJ path for GPU mode (default: models/scene.obj)')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output')
    parser.add_argument('--rate', type=int, default=30, help='LiDAR update rate Hz')
    parser.add_argument('--max_distance', type=float, default=500.0)
    parser.add_argument('--min_distance', type=float, default=0.05)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MuJoCo LiDAR visualization (Isaac Gym)")
    print("=" * 70)
    print(f"- LiDAR type: {args.lidar}")
    print(f"- Backend: {args.backend}")
    if args.backend == 'gpu':
        print(f"- OBJ: {args.obj_path if args.obj_path else 'default scene.obj'}")
    print(f"- Update rate: {args.rate} Hz")
    print(f"- Range: {args.min_distance}-{args.max_distance} m")
    print(f"- Verbose: {'on' if args.verbose else 'off'}")
    print("=" * 70)

    # Isaac Gym setup
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim = gym.create_sim(0, 2, gymapi.SIM_FLEX, sim_params)
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    env = gym.create_env(sim, gymapi.Vec3(-2,0,-2), gymapi.Vec3(2,2,2), 1)
    asset_root = "isaacgym/assets"
    asset_file = "urdf/anymal_b_simple_description/urdf/anymal.urdf"
    asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    
    env = gym.create_env(sim, lower, upper, 8)
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))
    actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

    import numpy as np
    import mujoco
    import taichi as ti

    # MuJoCo-LiDAR setup
    mj_model, mj_data = create_demo_scene("demo")
    rays_theta, rays_phi, use_livox_lidar = select_lidar_angles(args)
    rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
    rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)

    if args.backend == "gpu":
        if args.obj_path and os.path.exists(args.obj_path):
            obj_path = args.obj_path
        else:
            obj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models", "scene.obj")
    else:
        obj_path = None

    lidar = LidarSensor(
        mj_model,
        site_name="lidar_site",
        backend=args.backend,
        obj_path=obj_path
    )

    n_rays = len(rays_theta)
    if lidar.backend == "gpu":
        _rays_phi = ti.ndarray(dtype=ti.f32, shape=n_rays)
        _rays_theta = ti.ndarray(dtype=ti.f32, shape=n_rays)
        _rays_phi.from_numpy(rays_phi)
        _rays_theta.from_numpy(rays_theta)
        rays_phi = _rays_phi
        rays_theta = _rays_theta

    print(f"Number of rays: {n_rays}")

    # Keyboard control (optional, can be removed for minimalism)
    lidar_base_position = mj_model.body("lidar_base").pos
    lidar_base_orientation = mj_model.body("lidar_base").quat[[1,2,3,0]]
    kb_listener = KeyboardListener(lidar_base_position, lidar_base_orientation)

    step_cnt = 0
    step_gap = max(1, 60 // args.rate)
    pts_world = None
    
    rate = RateLimiter(frequency=args.rate)
    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as mj_viewer:
            mj_viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
            mj_viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value

            print("\nWASD/QE move, arrow keys rotate, ESC to exit (MuJoCo window)")
            while not gym.query_viewer_has_closed(viewer) and mj_viewer.is_running and kb_listener.running:
                # Update LiDAR pose from keyboard
                rate.sleep()
                site_position, site_orientation = kb_listener.update_lidar_pose(1./ args.rate)
                mj_model.body("lidar_base").pos[:] = site_position[:]
                mj_model.body("lidar_base").quat[:] = site_orientation[[3,0,1,2]]

                # Step MuJoCo
                mujoco.mj_step(mj_model, mj_data)
                mj_viewer.sync()

                # Step Isaac Gym
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

                step_cnt += 1
                if step_cnt % step_gap == 0:
                    # Resample Livox if needed
                    if use_livox_lidar:
                        if lidar.backend == "cpu":
                            rays_theta, rays_phi = kb_listener.livox_generator.sample_ray_angles()
                            rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
                            rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)
                        else:
                            rays_theta, rays_phi = kb_listener.livox_generator.sample_ray_angles_ti()
                    lidar.update(mj_data, rays_phi, rays_theta)
                    pts_world = lidar.get_data_in_world_frame()
                    if pts_world is not None and pts_world.shape[0] > 0:
                        draw_lidar_points(gym, viewer, env, pts_world)
                    if args.verbose and pts_world is not None:
                        print(f"LiDAR points: {pts_world.shape[0]}")

    except KeyboardInterrupt:
        print("Interrupted by user, exiting...")
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pts_world is not None:
            np.save("mesh_test_hit_points.npy", pts_world)
        print("Simulation ended.")

if __name__ == "__main__":
    main()
