import os
import argparse
import numpy as np

# 可选: 如果用户未安装请提示
try:
    import open3d as o3d
except ImportError as e:
    raise ImportError("需要安装 open3d: pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple") from e

import matplotlib.cm as cm  # 新增: 用于颜色映射


def load_points(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"数据形状应为 (N,3)，当前为 {pts.shape}")
    return pts.astype(np.float32)


def filter_points(pts: np.ndarray, min_dist: float, max_dist: float | None):
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > min_dist
    if max_dist is not None:
        mask &= norms <= max_dist
    return pts[mask], norms[mask]


def build_o3d_cloud(pts: np.ndarray, dists: np.ndarray, color_mode: str, voxel: float | None, cmap_name: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if color_mode == 'distance':
        # 归一化距离映射到伪彩
        if len(dists) > 0:
            d_min, d_max = dists.min(), dists.max()
            span = max(d_max - d_min, 1e-6)
            norm = (dists - d_min) / span  # 0~1
            # 简单 jet 风格 (r,g,b)
            r = np.clip(1.5 - np.abs(4 * (norm - 0.75)), 0, 1)
            g = np.clip(1.5 - np.abs(4 * (norm - 0.50)), 0, 1)
            b = np.clip(1.5 - np.abs(4 * (norm - 0.25)), 0, 1)
            colors = np.stack([r, g, b], axis=1)
        else:
            colors = np.zeros_like(pts)
    elif color_mode == 'z':
        if len(pts) > 0:
            z = pts[:, 2]
            z_min, z_max = z.min(), z.max()
            span = max(z_max - z_min, 1e-6)
            norm = (z - z_min) / span
            cmap_func = cm.get_cmap(cmap_name)
            colors = cmap_func(norm)[:, :3]
        else:
            colors = np.zeros_like(pts)
    else:  # 固定颜色
        colors = np.tile(np.array([[0.2, 0.7, 1.0]]), (pts.shape[0], 1))

    pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def main():
    parser = argparse.ArgumentParser(description='Open3D 点云可视化 (mesh_test_hit_points.npy)')
    parser.add_argument('-f', '--file', default='mesh_test_hit_points.npy', help='npy文件路径')
    parser.add_argument('--min-dist', type=float, default=1e-6, help='最小有效距离过滤')
    parser.add_argument('--max-dist', type=float, default=None, help='最大距离过滤')
    parser.add_argument('--voxel', type=float, default=None, help='体素降采样大小')
    parser.add_argument('--color', choices=['distance', 'z', 'fixed'], default='z', help='着色模式')
    parser.add_argument('--cmap', type=str, default='viridis', help='用于 --color z 或 distance 的颜色映射')
    parser.add_argument('--show-bbox', action='store_true', help='显示包围盒')
    parser.add_argument('--frame', action='store_true', help='显示坐标系')
    args = parser.parse_args()

    pts = load_points(args.file)
    print(f"原始点数: {len(pts)}")
    pts_f, dists = filter_points(pts, args.min_dist, args.max_dist)
    print(f"过滤后点数: {len(pts_f)}")
    if len(pts_f) == 0:
        print("无有效点, 退出")
        return

    pcd = build_o3d_cloud(pts_f, dists, args.color, args.voxel, args.cmap)

    geoms = [pcd]
    if args.show_bbox:
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        geoms.append(aabb)
    if args.frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    o3d.visualization.draw_geometries(geoms, window_name='Hit Points', width=960, height=720)


if __name__ == '__main__':
    main()
