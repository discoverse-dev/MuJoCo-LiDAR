import os
import argparse

import time
import mujoco
import numpy as np
from plyfile import PlyData, PlyElement

from mujoco_lidar import (
    LidarSensor, LivoxGenerator, 
    generate_HDL64, generate_vlp32, generate_os128, generate_grid_scan_pattern
)
from mujoco_lidar.mj_lidar_utils import create_demo_scene


def create_height_colors(points, colormap='viridis'):
    """
    根据z轴高度创建渐变颜色
    Args:
        points: 点云数据 (N, 3)
        colormap: 颜色映射类型 ('viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow')
    Returns:
        colors: RGB颜色数组 (N, 3)，值范围0-255
    """
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    
    # 归一化z值到0-1范围
    if z_max > z_min:
        z_normalized = (z_values - z_min) / (z_max - z_min)
    else:
        z_normalized = np.zeros_like(z_values)
    
    # 定义颜色映射
    if colormap == 'viridis':
        # 紫-蓝-绿-黄渐变
        colors = np.zeros((len(points), 3))
        colors[:, 0] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [68, 59, 53, 122, 253])  # R
        colors[:, 1] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [1, 82, 144, 191, 231])  # G
        colors[:, 2] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [84, 139, 49, 99, 37])   # B
    elif colormap == 'plasma':
        # 紫-粉-黄渐变
        colors = np.zeros((len(points), 3))
        colors[:, 0] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [13, 62, 135, 221, 240])  # R
        colors[:, 1] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [8, 4, 85, 203, 249])     # G
        colors[:, 2] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [135, 168, 132, 85, 33])  # B
    elif colormap == 'jet':
        # 蓝-青-绿-黄-红渐变
        colors = np.zeros((len(points), 3))
        colors[:, 0] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [0, 0, 0, 255, 255])      # R
        colors[:, 1] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [0, 255, 255, 255, 0])    # G
        colors[:, 2] = np.interp(z_normalized, [0, 0.25, 0.5, 0.75, 1], [128, 255, 0, 0, 0])      # B
    elif colormap == 'rainbow':
        # 彩虹色渐变
        colors = np.zeros((len(points), 3))
        colors[:, 0] = np.interp(z_normalized, [0, 0.2, 0.4, 0.6, 0.8, 1], [255, 255, 0, 0, 0, 255])    # R
        colors[:, 1] = np.interp(z_normalized, [0, 0.2, 0.4, 0.6, 0.8, 1], [0, 255, 255, 255, 0, 0])    # G
        colors[:, 2] = np.interp(z_normalized, [0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0, 0, 255, 255, 255])    # B
    else:
        # 默认使用简单的蓝-红渐变
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_normalized * 255  # R: 低处蓝色，高处红色
        colors[:, 1] = 0                   # G: 固定为0
        colors[:, 2] = (1 - z_normalized) * 255  # B: 低处蓝色，高处无蓝色
    
    return colors.astype(np.uint8)


def save_ply_with_colors(points, colors, filename):
    """
    保存带颜色的PLY点云文件
    Args:
        points: 点云数据 (N, 3)
        colors: 颜色数据 (N, 3)，值范围0-255
        filename: 输出文件名
    """
    # 创建PLY数据
    vertex_data = np.zeros(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    
    # 创建PLY元素
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # 保存PLY文件
    PlyData([vertex_element]).write(filename)
    print(f"Saved {points.shape[0]} colored points to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run single LiDAR scan and save points')
    parser.add_argument('obj_path', type=str, help='Path to OBJ file to use as mesh')
    parser.add_argument('-t', '--pos', type=float, nargs=3, default=[0.0, 0.0, 1.0], help='LiDAR origin position x y z')
    parser.add_argument('-r', '--quat', type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0], help='LiDAR quaternion w x y z')
    parser.add_argument('--lidar', type=str, default='HDL64', choices=['mid360','vlp32','HDL64','os128','grid'], help='LiDAR model (no livox)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (.npy)')
    parser.add_argument('--no-save', action='store_true', help='Do not save output file')
    parser.add_argument('--test-speed', action='store_true', help='Do not save output file')
    parser.add_argument('--colormap', type=str, default='viridis', 
                       choices=['viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow'],
                       help='Color map for height-based coloring')
    args = parser.parse_args()

    # create a simple scene that contains a lidar site
    # use mesh_scene only if obj-path provided, otherwise use floor
    if args.obj_path:
        # create a scene with mesh placeholder and then replace mesh file path in xml
        model, data = create_demo_scene('mesh_scene')
        # if model has asset mesh named 'scene' try to set its file attribute
        try:
            for i in range(model.nmesh):
                name = model.mesh(i).name
                if name == 'scene' or name == 'eight':
                    model.mesh(i).file = args.obj_path
        except Exception:
            pass
    else:
        model, data = create_demo_scene('floor')

    # prepare rays
    use_livox_lidar = False
    if args.lidar == 'mid360':
        livox_generator = LivoxGenerator(args.lidar)
        rays_theta, rays_phi = livox_generator.sample_ray_angles()
        use_livox_lidar = True
    elif args.lidar == 'HDL64':
        rays_theta, rays_phi = generate_HDL64()
    elif args.lidar == 'vlp32':
        rays_theta, rays_phi = generate_vlp32()
    elif args.lidar == 'os128':
        rays_theta, rays_phi = generate_os128()
    else:
        # grid: small grid pattern
        rays_theta, rays_phi = generate_grid_scan_pattern(azimuth_n=64, elevation_n=16)

    rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
    rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)

    # create lidar sensor using cpu backend
    lidar = LidarSensor(model, site_name='lidar_site', backend='gpu', obj_path=args.obj_path)

    # set site pose from provided pos+quat
    # mujoco stores quat as [w,x,y,z] in model? MjData.site.xpos and xmat used in sensor_pose update
    # We'll directly set site position and orientation matrix
    data.site(model.site('lidar_site').id).xpos[:] = np.array(args.pos, dtype=np.float64)
    # convert quat w,x,y,z to rotation matrix
    w, x, y, z = args.quat
    # normalize
    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ], dtype=np.float64)
    data.site(model.site('lidar_site').id).xmat[:] = R.reshape(9)

    # perform one update
    for _ in range(3):
        if use_livox_lidar:
            rays_theta, rays_phi = livox_generator.sample_ray_angles()
        lidar.update(data, rays_phi, rays_theta)

    if args.test_speed:
        time_lst = []
        for _ in range(5):
            if use_livox_lidar:
                rays_theta, rays_phi = livox_generator.sample_ray_angles()
            start_time = time.time()
            lidar.update(data, rays_phi, rays_theta)
            end_time = time.time()
            time_lst.append(end_time - start_time)
        # print(f"Number of mesh faces : {lidar.ti_lidar.n_faces}")
        # print(f"Average update took {np.mean(time_lst)*1e3:.2f} milliseconds, ray count: {rays_theta.shape[0]}, rays per second: {(rays_theta.shape[0]/np.mean(time_lst)):.2f}")
        print(f"'{args.lidar}': {lidar.ti_lidar.n_faces}, {np.mean(time_lst)*1e3:.2f}, {rays_theta.shape[0]}, {(rays_theta.shape[0]/np.mean(time_lst)):.2f}")

    if not args.no_save:
        points_local = lidar.get_data_in_local_frame()
        # save to .npy
        # out_dir = os.path.dirname(os.path.abspath(args.output))
        # if out_dir and not os.path.exists(out_dir):
        #     os.makedirs(out_dir, exist_ok=True)
        # np.save(args.output, points_local)
        # print(f"Saved {points_local.shape} points to {args.output}")

        # 生成PLY文件名（替换.npy扩展名为.ply）
        ply_output = os.path.splitext(args.output)[0] + '.ply'
        
        # 根据z轴高度生成渐变颜色
        colors = create_height_colors(points_local, args.colormap)
        
        # 保存带颜色的PLY文件
        save_ply_with_colors(points_local, colors, ply_output)


if __name__ == '__main__':
    main()

"""
print(f"'{args.lidar}': {lidar.ti_lidar.n_faces}, {np.mean(time_lst)*1e3:.2f}, {rays_theta.shape[0]}, {(rays_theta.shape[0]/np.mean(time_lst)):.2f}")

'mid360': 6873, 0.57, 24000, 42309724.28
'vlp32': 6873, 1.57, 120000, 76233506.51
'HDL64': 6873, 1.49, 110016, 73942881.00
'os128': 6873, 2.25, 260096, 115806638.20

'mid360': 977968, 0.92, 24000, 26071819.74
'vlp32': 977968, 2.01, 120000, 59746501.74
'HDL64': 977968, 1.93, 110016, 57075778.80
'os128': 977968, 3.07, 260096, 84694286.27

'mid360': 1938266, 1.40, 24000, 17183315.01
'vlp32': 1938266, 1.96, 120000, 61116957.49
'HDL64': 1938266, 2.07, 110016, 53225125.60
'os128': 1938266, 4.33, 260096, 60111508.69

'mid360': 2160099, 1.59, 24000, 15116878.81
'vlp32': 2160099, 2.54, 120000, 47255769.93
'HDL64': 2160099, 2.51, 110016, 43806538.02
'os128': 2160099, 3.80, 260096, 68381767.72

'mid360': 241119, 1.07, 24000, 22406467.52
'vlp32': 241119, 1.47, 120000, 81707220.78
'HDL64': 241119, 1.80, 110016, 60996767.86
'os128': 241119, 3.11, 260096, 83551995.37

'mid360': 6625274, 1.32, 24000, 18152576.19
'vlp32': 6625274, 1.58, 120000, 75979180.00
'HDL64': 6625274, 2.00, 110016, 55118439.15
'os128': 6625274, 2.89, 260096, 90107434.04


"""