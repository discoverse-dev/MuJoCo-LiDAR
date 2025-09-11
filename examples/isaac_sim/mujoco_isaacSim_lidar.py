import time
import argparse
import traceback
import numpy as np
import os

import mujoco
import mujoco.viewer
import pandas as pd
import taichi as ti
from scipy.spatial.transform import Rotation
from loop_rate_limiters import RateLimiter
# Isaac Sim/Omniverse imports
from isaacsim import SimulationApp
# DONOT MOVE THIS TO THE FRONT OF THE SCRIPT, ISSAC WOULD CRASH
kit = SimulationApp({"headless": True})


import carb
import omni
import omni.kit.viewport.utility
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Gf

import omni.usd
from pxr import UsdGeom, Gf

import sys
# TODO: remove
sys.path.append("/home/yiyi/Data/github/MuJoCo-LiDAR")
from mujoco_lidar import (
    LidarSensor, LivoxGenerator,
    generate_vlp32, generate_HDL64, generate_os128
)
from mujoco_lidar.mj_lidar_utils import create_demo_scene, KeyboardListener

# DONOT MOVE THIS TO THE FRONT OF THE SCRIPT, ISSAC WOULD CRASH
enable_extension("isaacsim.util.debug_draw")

class SimpleLidarVisualizerIsaac:
    """Simplified real-time LiDAR visualizer using Isaac Sim PointInstancer for point cloud visualization"""

    def __init__(self, args):
        # 启动Isaac Sim
        self.sim_app = kit
        self.rate = args.rate
        self.stage = omni.usd.get_context().get_stage()

        # 创建MuJoCo场景
        self.mj_model, self.mj_data = create_demo_scene("demo")

        # Set LiDAR type
        self.use_livox_lidar = False
        if args.lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
            self.livox_generator = LivoxGenerator(args.lidar)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.use_livox_lidar = True
        elif args.lidar == "HDL64":
            self.rays_theta, self.rays_phi = generate_HDL64()
        elif args.lidar == "vlp32":
            self.rays_theta, self.rays_phi = generate_vlp32()
        elif args.lidar == "os128":
            self.rays_theta, self.rays_phi = generate_os128()
        else:
            raise ValueError(f"Unsupported LiDAR type: {args.lidar}")

        # 优化内存布局
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        # 选择 OBJ (仅 GPU 后端需要)
        if args.backend == "gpu":
            if args.obj_path and os.path.exists(args.obj_path):
                obj_path = args.obj_path
            else:
                obj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models", "scene.obj")
        else:
            obj_path = None  # CPU 不使用

        # Create LiDAR sensor (new unified interface)
        self.lidar = LidarSensor(
            self.mj_model,
            site_name="lidar_site",
            backend=args.backend,
            obj_path=obj_path
        )

        # 如果是 GPU，需要把角度转成 taichi ndarray
        n_rays = len(self.rays_theta)
        if self.lidar.backend == "gpu":
            _rays_phi = ti.ndarray(dtype=ti.f32, shape=n_rays)
            _rays_theta = ti.ndarray(dtype=ti.f32, shape=n_rays)
            _rays_phi.from_numpy(self.rays_phi)
            _rays_theta.from_numpy(self.rays_theta)
            self.rays_phi = _rays_phi
            self.rays_theta = _rays_theta

        print(f"射线数量: {n_rays}")

        # 获取激光雷达初始位置和方向
        lidar_base_position = self.mj_model.body("lidar_base").pos
        lidar_base_orientation = self.mj_model.body("lidar_base").quat[[1,2,3,0]]

        # 创建键盘监听器
        self.kb_listener = KeyboardListener(lidar_base_position, lidar_base_orientation)

        # 配置参数
        self.args = args
        self.running = True


    def update_visualization(self, points: np.ndarray):
        """更新Isaac Sim中的点云PointInstancer"""
        green_color = (0, 1, 0, 0.75)
        green_size = 4
        # self._debug_draw_clear_points()
        self._debug_draw_pointcloud(points, green_color, green_size, clear_existing=True)



    def _debug_draw_clear_points(self):
        # DONOT MOVE THIS TO THE FRONT OF THE SCRIPT, ISSAC WOULD CRASH
        from isaacsim.util.debug_draw import _debug_draw

        draw_iface = _debug_draw.acquire_debug_draw_interface()
        draw_iface.clear_points()

    def _debug_draw_pointcloud(self, pointcloud_data, color, size, clear_existing=False):
        import time
        t_start = time.time()
        if not (isinstance(pointcloud_data, np.ndarray) and pointcloud_data.ndim == 2 and pointcloud_data.shape[1] == 3):
            print("Warning: pointcloud_data must be a NumPy array with shape (N, 3).")
            return
        t_check = time.time()
        if hasattr(self, 'args') and getattr(self.args, 'verbose', False):
            print(f"[DEBUG] pointcloud_data check took {(t_check-t_start)*1000:.2f}ms")

        # DONOT MOVE THIS TO THE FRONT OF THE SCRIPT, ISSAC WOULD CRASH
        from isaacsim.util.debug_draw import _debug_draw
        t_import = time.time()
        if hasattr(self, 'args') and getattr(self.args, 'verbose', False):
            print(f"[DEBUG] import _debug_draw took {(t_import-t_check)*1000:.2f}ms")

        draw_iface = _debug_draw.acquire_debug_draw_interface()
        t_iface = time.time()
        if hasattr(self, 'args') and getattr(self.args, 'verbose', False):
            print(f"[DEBUG] acquire_debug_draw_interface took {(t_iface-t_import)*1000:.2f}ms")

        # pointcloud_data: (N, 3) numpy array
        # color: tuple of length 3 or 4 (e.g., (0, 1, 0, 0.75))
        # size: scalar

        t_np_start = time.time()
        points_cloud = pointcloud_data
        colors_cloud = np.tile(np.array(color), (pointcloud_data.shape[0], 1))
        sizes_cloud = np.full((pointcloud_data.shape[0],), size)

        t_np_end = time.time()
        if hasattr(self, 'args') and getattr(self.args, 'verbose', False):
            print(f"[DEBUG] numpy tolist/tile/full took {(t_np_end-t_np_start)*1000:.2f}ms")

        if clear_existing:
            t_clear = time.time()
            self._debug_draw_clear_points()
            t_clear_end = time.time()
            if hasattr(self, 'args') and getattr(self.args, 'verbose', False):
                print(f"[DEBUG] _debug_draw_clear_points took {(t_clear_end-t_clear)*1000:.2f}ms")
        t_draw = time.time()
        draw_iface.draw_points(points_cloud, colors_cloud, sizes_cloud)
        t_draw_end = time.time()
        if hasattr(self, 'args') and getattr(self.args, 'verbose', False):
            print(f"[DEBUG] draw_points took {(t_draw_end-t_draw)*1000:.2f}ms")
            print(f"[DEBUG] _debug_draw_pointcloud total: {(t_draw_end-t_start)*1000:.2f}ms")

    def _resample_livox_if_needed(self):
        """Resample Livox angles according to backend (keep consistent with ros2 example)"""
        if not self.use_livox_lidar:
            return
        if self.lidar.backend == "cpu":
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
            self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)
        else:  # gpu
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles_ti()

    def run(self):
        """Start simulation and visualization"""
        step_cnt = 0
        step_gap = max(1, 60 // self.args.rate)
        pts_world = None

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.sim_app.close()
            sys.exit()
        stage.add_reference_to_stage(
            "/home/yiyi/Data/github/MuJoCo-LiDAR/models/scene.usd", "/background"
        )
        rate = RateLimiter(self.rate)
        i = 0
        try:
            print("\n" + "=" * 60)
            print("控制说明: WASD/QE 移动, 方向键旋转, ESC 退出")
            print("=" * 60)

            while (self.running and self.kb_listener.running and self.sim_app.is_running()):
                # if i > 100:
                #     break
                loop_start = time.time()
                rate.sleep()
                t0 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] rate.sleep took {(t0-loop_start)*1000:.2f}ms")
                # 更新激光雷达位置和方向
                site_position, site_orientation = self.kb_listener.update_lidar_pose(1./60.)
                t1 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] update_lidar_pose took {(t1-t0)*1000:.2f}ms")
                self.mj_model.body("lidar_base").pos[:] = site_position[:]
                self.mj_model.body("lidar_base").quat[:] = site_orientation[[3,0,1,2]]
                t2 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] set pos/quats took {(t2-t1)*1000:.2f}ms")

                # 更新仿真
                mujoco.mj_step(self.mj_model, self.mj_data)
                step_cnt += 1
                t3 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] mujoco.mj_step took {(t3-t2)*1000:.2f}ms")

                # 按频率更新点云
                # if step_cnt % step_gap == 0:
                t4 = time.time()
                self._resample_livox_if_needed()
                t5 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] _resample_livox_if_needed took {(t5-t4)*1000:.2f}ms")
                self.lidar.update(self.mj_data, self.rays_phi, self.rays_theta)
                t6 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] lidar.update took {(t6-t5)*1000:.2f}ms")
                pts_world = self.lidar.get_data_in_world_frame()
                # df = pd.DataFrame(pts_world)
                # df.to_csv(f"{i}.csv", header=False, index=False)
                # i += 1
                t7 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] get_data_in_world_frame took {(t7-t6)*1000:.2f}ms")
                self.update_visualization(pts_world)
                t8 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] update_visualization took {(t8-t7)*1000:.2f}ms")
                if self.args.verbose:
                    quat = Rotation.from_matrix(self.lidar.sensor_rotation).as_quat()
                    euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
                    print(f"位置: [{self.lidar.sensor_position[0]:.2f},{self.lidar.sensor_position[1]:.2f},{self.lidar.sensor_position[2]:.2f}] "+
                            f"范围: x=({pts_world[:,0].min():.2f} {pts_world[:,0].max():.2f}), y=({pts_world[:,1].min():.2f} {pts_world[:,1].max():.2f}), z=({pts_world[:,2].min():.2f} {pts_world[:,2].max():.2f}) "+
                            f"欧拉: [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°] "
                            f"点数: {pts_world.shape[0]}")
                    print(f"[DEBUG TIMING] total loop: {(t8-loop_start)*1000:.2f}ms")
                # Isaac Sim渲染循环
                t9 = time.time()
                self.sim_app.update()
                t10 = time.time()
                if self.args.verbose:
                    print(f"[DEBUG] sim_app.update took {(t10-t9)*1000:.2f}ms")

        except KeyboardInterrupt:
            print("用户中断，正在退出...")
        except Exception as e:
            print(f"仿真出错: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            if pts_world is not None:
                np.save("mesh_test_hit_points.npy", pts_world)
            self.sim_app.close()


def main():
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR简化可视化（Isaac Sim PointInstancer）')
    parser.add_argument('--lidar', type=str, default='mid360',
                        choices=['avia', 'HAP', 'horizon', 'mid40', 'mid70', 'mid360', 'tele', 'HDL64', 'vlp32', 'os128'])
    parser.add_argument('--backend', type=str, default='gpu', choices=['cpu', 'gpu'], help='LiDAR后端 (cpu/gpu)')
    parser.add_argument('--obj-path', type=str, help='GPU模式下用于构建BVH的OBJ路径 (默认使用models/scene.obj)')
    parser.add_argument('--profiling', action='store_true', help='(保留参数, 当前接口未实现详细profiling输出)')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    parser.add_argument('--rate', type=int, default=30, help='LiDAR更新频率Hz')
    parser.add_argument('--max_distance', type=float, default=500.0)
    parser.add_argument('--min_distance', type=float, default=0.05)
    args = parser.parse_args()
    print("\n" + "=" * 70)
    print("MuJoCo LiDAR简化可视化 (Isaac Sim PointInstancer, 新LidarSensor接口)")
    print("=" * 70)
    print(f"- LiDAR型号: {args.lidar}")
    print(f"- 后端: {args.backend}")
    if args.backend == 'gpu':
        print(f"- OBJ: {args.obj_path if args.obj_path else '默认 scene.obj'}")
    print(f"- 更新频率: {args.rate} Hz")
    print(f"- 可视化范围: {args.min_distance}-{args.max_distance} m")
    print(f"- 详细输出: {'启用' if args.verbose else '禁用'}")
    print("=" * 70)
    try:
        app = SimpleLidarVisualizerIsaac(args)
        app.run()
    except KeyboardInterrupt:
        print("用户中断，程序退出")
    except Exception as e:
        print(f"程序出错: {e}")
        traceback.print_exc()
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()
