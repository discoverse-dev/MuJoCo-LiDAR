import os
import sys
import time
import numpy as np
import taichi as ti

# 加入项目根目录 (tests 上一级即包根目录)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from lidar_scanner import StaticBVHLidar
from scan_gen import generate_os128, create_lidar_single_line, generate_grid_scan_pattern

# 初始化 Taichi
ti.init(arch=ti.gpu)

def pick_obj():
    # 优先使用命令行参数，其次桌面 N6.obj，再次使用项目自带 scene.obj
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    # desktop_obj = os.path.expanduser('~/Desktop/N6.obj')
    # if os.path.isfile(desktop_obj):
    #     return desktop_obj
    builtin = os.path.join(project_root, '../models', 'scene.obj')
    return builtin

def prepare_angles(mode='os128'):
    if mode == 'os128':
        return generate_os128()
    elif mode == 'single_line':
        return create_lidar_single_line(360, 2*np.pi)
    elif mode == 'grid':
        return generate_grid_scan_pattern(512, 64, phi_range=(-np.pi*0.3, np.pi*0.3))
    elif mode == 'max':
        return generate_grid_scan_pattern(3600, 720, phi_range=(-np.pi*0.3, np.pi*0.3))
    else:
        raise ValueError('未知模式')

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    obj_path = pick_obj()
    if not os.path.isfile(obj_path):
        print(f'找不到 OBJ: {obj_path}')
        sys.exit(1)
    print(f'使用模型: {obj_path}')

    # 1. 角度生成（一次）
    theta, phi = prepare_angles('os128')
    n_rays = theta.shape[0]
    print(f'生成 {n_rays} 条射线')

    # 2. 转为 Taichi ndarray（一次）
    theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
    phi_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
    theta_ti.from_numpy(theta.astype(np.float32))
    phi_ti.from_numpy(phi.astype(np.float32))

    # 3. 构建 Lidar（一次 BVH）
    lidar = StaticBVHLidar(obj_path=obj_path)

    build_start = time.time()
    lidar.lbvh.build()
    ti.sync()
    print(f'BVH 构建耗时: {(time.time()-build_start)*1e3:.2f} ms')

    # 4. 位姿（可后续更新），这里只用单位矩阵
    pose = np.eye(4, dtype=np.float32)

    # 5. 预热
    print('预热...')
    lidar.trace_ti(pose, theta_ti, phi_ti)
    ti.sync()

    # 6. 正式多次测量
    iters = 5
    times = []
    for _ in range(iters):
        t0 = time.time()
        lidar.trace_ti(pose, theta_ti, phi_ti)
        t1 = time.time()
        times.append(t1 - t0)
    avg = sum(times)/len(times)

    hit_pts = lidar.get_hit_points()
    dists = lidar.get_hit_distances()

    # 7. 结果统计
    valid_mask = dists > 0
    hit_count = valid_mask.sum()
    throughput = n_rays/avg/1e6  # M rays/s

    stats = lidar.get_stats()

    print(f'平均单帧: {avg*1e3:.3f} ms  ({throughput:.2f} MRays/s)')
    print(f'命中: {hit_count}/{n_rays}  命中率: {hit_count/n_rays*100:.2f}%')
    print(f'候选溢出次数: {stats["overflow_count"]}')

    # 8. 保存结果
    np.save('mesh_test_hit_points.npy', hit_pts)
    np.save('mesh_test_distances.npy', dists)
    print('结果已保存: mesh_test_hit_points.npy / mesh_test_distances.npy')