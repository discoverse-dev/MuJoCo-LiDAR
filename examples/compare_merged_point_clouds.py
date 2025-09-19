#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import json
from plyfile import PlyData
import argparse

# 添加GS-LiDAR目录到路径
sys.path.append('/home/xyys2003/ws/gslidar/GS-LiDAR')
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from chamfer.fscore import fscore


def read_ply(file_path):
    """读取PLY文件，返回点云数据"""
    try:
        plydata = PlyData.read(file_path)
        vertex = plydata['vertex']
        
        # 提取点坐标
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        points = np.column_stack((x, y, z))
        
        return points
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None


def remove_origin_points(points: np.ndarray) -> np.ndarray:
    """剔除位于原点(0,0,0)的点。"""
    if points is None or points.size == 0:
        return points
    mask = ~((points[:, 0] == 0) & (points[:, 1] == 0) & (points[:, 2] == 0))
    return points[mask]


def merged_point_cloud_comparison(file1, file2, output_file="merged_comparison.json"):
    """比较两个单独的PLY点云文件并计算指标"""
    # 读取两个PLY文件
    points1 = read_ply(file1)
    points2 = read_ply(file2)

    if points1 is None:
        print(f"无法读取点云文件: {file1}")
        return None
    if points2 is None:
        print(f"无法读取点云文件: {file2}")
        return None

    # 剔除原点处的点
    orig_count1, orig_count2 = points1.shape[0], points2.shape[0]
    points1 = remove_origin_points(points1)
    points2 = remove_origin_points(points2)
    removed1, removed2 = orig_count1 - points1.shape[0], orig_count2 - points2.shape[0]

    merged_dir1_points = points1
    merged_dir2_points = points2

    print(f"文件1点云数: {merged_dir1_points.shape[0]} 个点 (剔除原点 {removed1} 个)")
    print(f"文件2点云数: {merged_dir2_points.shape[0]} 个点 (剔除原点 {removed2} 个)")

    # 转换为PyTorch张量

    dir1_tensor = torch.tensor(merged_dir1_points, dtype=torch.float32).cuda()
    dir2_tensor = torch.tensor(merged_dir2_points, dtype=torch.float32).cuda()

    print("\n计算合并后的点云指标...")
    
    # 计算Chamfer距离
    chamfer_func = chamfer_3DDist()
    dist1, dist2, _, _ = chamfer_func(dir1_tensor.unsqueeze(0), dir2_tensor.unsqueeze(0))
    
    # 计算C-D和F-score
    cd = dist1.mean() + dist2.mean()
    f_score, precision, recall = fscore(dist1, dist2, threshold=0.05)

    
    
    # 将结果转换为CPU张量，并提取标量值
    cd_value = cd.item()
    f_score_value = f_score.item()
    precision_value = precision.item()
    recall_value = recall.item()
    
    # 保存结果
    results = {
        "merged": {
            "C-D": cd_value,
            "F-score": f_score_value,
            "Precision": precision_value,
            "Recall": recall_value
        },
        "points_stats": {
            "file1_points": int(merged_dir1_points.shape[0]),
            "file2_points": int(merged_dir2_points.shape[0]),
            "file1_removed_origin": int(removed1),
            "file2_removed_origin": int(removed2)
        },
        "files": {
            "file1": os.path.basename(file1),
            "file2": os.path.basename(file2)
        }
    }
    
    # 打印结果
    print("-" * 60)
    print(f"比较后的点云指标:")
    print(f"文件1: {os.path.basename(file1)}")
    print(f"文件2: {os.path.basename(file2)}")
    print(f"Chamfer距离: {cd_value:.6f}")
    print(f"F-score: {f_score_value:.6f}")
    print(f"精确率: {precision_value:.6f}")
    print(f"召回率: {recall_value:.6f}")
    print(f"点数: 目录1 {merged_dir1_points.shape[0]} (剔除 {removed1}) vs 目录2 {merged_dir2_points.shape[0]} (剔除 {removed2})")
    print("-" * 60)
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估完成！结果已保存到 {output_file}")
    
    # 无需临时目录清理（未使用批处理）
    
    return results


def main():
    parser = argparse.ArgumentParser(description='比较合并后的两个目录点云')
    parser.add_argument('--file1', type=str, 
                        required=True,
                        help='第一个PLY点云文件路径')
    parser.add_argument('--file2', type=str, 
                        required=True,
                        help='第二个PLY点云文件路径')
    parser.add_argument('--output', type=str, default='merged_point_cloud_comparison.json', help='输出文件')
    args = parser.parse_args()
    
    merged_point_cloud_comparison(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main() 