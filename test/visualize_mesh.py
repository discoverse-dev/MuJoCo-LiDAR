import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import mujoco

def draw_aabb_wireframe(ax, min_bounds, max_bounds, color='red', linewidth=2):
    """绘制AABB包围盒的线框"""
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    
    # 定义包围盒的8个顶点
    vertices = np.array([
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_max, y_max, z_min],  # 2
        [x_min, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_max, y_max, z_max],  # 6
        [x_min, y_max, z_max],  # 7
    ])
    
    # 定义包围盒的12条边（每条边连接两个顶点）
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面的4条边
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面的4条边
        [0, 4], [1, 5], [2, 6], [3, 7],  # 4条竖直边
    ]
    
    # 绘制每条边
    for edge in edges:
        start, end = edge
        ax.plot3D(
            [vertices[start][0], vertices[end][0]],
            [vertices[start][1], vertices[end][1]], 
            [vertices[start][2], vertices[end][2]],
            color=color, linewidth=linewidth
        )

def visualize_mesh_triangles(mj_model, mj_scene):
    """可视化模型中所有mesh的三角面片"""
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 存储所有三角形面片
    all_triangles = []
    mesh_colors = []
    
    print(f"模型中总共有 {mj_model.nmesh} 个mesh")
    
    # 遍历所有mesh
    for mesh_id in range(mj_model.nmesh):
        mesh_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        print(f"\n处理 Mesh {mesh_id}: {mesh_name}")
        
        # 获取mesh的顶点数据
        vert_addr = mj_model.mesh_vertadr[mesh_id]
        vert_num = mj_model.mesh_vertnum[mesh_id]
        vertices = mj_model.mesh_vert[vert_addr:vert_addr + vert_num].reshape(-1, 3)
        
        # 获取mesh的面片数据
        face_addr = mj_model.mesh_faceadr[mesh_id]
        face_num = mj_model.mesh_facenum[mesh_id]
        
        print(f"  顶点数量: {vert_num}, 面片数量: {face_num}")
        
        # 为当前mesh生成随机颜色
        mesh_color = (random.random(), random.random(), random.random(), 0.7)
        
        # 获取对应的几何体变换信息
        mesh_geom_pos = None
        mesh_geom_rot = None
        
        # 在场景中找到对应的几何体
        for geom_idx in range(mj_scene.ngeom):
            geom = mj_scene.geoms[geom_idx]
            if geom.type == 7 and geom.dataid == mesh_id:  # type=7是mesh
                mesh_geom_pos = geom.pos.copy()
                mesh_geom_rot = geom.mat.reshape(3, 3).copy()
                print(f"  几何体位置: {mesh_geom_pos}")
                print(f"  几何体旋转矩阵:\n{mesh_geom_rot}")
                break
        
        # 处理每个三角形面片
        for face_idx in range(face_addr, face_addr + face_num):
            # 获取三角形的三个顶点索引
            triangle_indices = mj_model.mesh_face[face_idx]
            
            # 获取三角形的三个顶点坐标
            triangle_vertices = vertices[triangle_indices]
            
            # 应用几何体的变换（旋转+平移）
            if mesh_geom_pos is not None and mesh_geom_rot is not None:
                # 先应用旋转，再应用平移
                transformed_vertices = (mesh_geom_rot @ triangle_vertices.T).T + mesh_geom_pos
            else:
                transformed_vertices = triangle_vertices
            
            # 添加到总的三角形列表
            all_triangles.append(transformed_vertices)
            mesh_colors.append(mesh_color)
    
    # 创建3D多边形集合
    if all_triangles:
        poly_collection = Poly3DCollection(all_triangles, 
                                         facecolors=mesh_colors,
                                         edgecolors='black',
                                         linewidths=0.5,
                                         alpha=0.7)
        ax.add_collection3d(poly_collection)
        
        # 计算边界框以设置坐标轴范围
        all_vertices = np.array([tri for tri in all_triangles]).reshape(-1, 3)
        
        x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
        y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
        z_min, z_max = all_vertices[:, 2].min(), all_vertices[:, 2].max()
        
        # 添加一些边距
        margin = 0.5
        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        ax.set_zlim([z_min - margin, z_max + margin])

        print(f"\n总共可视化了 {len(all_triangles)} 个三角形面片")
        print(f"坐标范围: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
    
    # 绘制AABB包围盒
    draw_aabb_wireframe(ax, [-0.2, -0.48, -1.], [0.21, 0.48, 1.], color='red', linewidth=2)

    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh三角面片可视化 (红色线框为AABB包围盒)')
    
    # 设置等比例坐标轴
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, all_triangles

def print_mesh_details(mj_model, mesh_id=0):

    """打印指定mesh的详细三角形信息"""
    
    mesh_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    print(f"Mesh {mesh_id} ({mesh_name}) 详细信息:")
    
    # 获取顶点数据
    vert_addr = mj_model.mesh_vertadr[mesh_id]
    vert_num = mj_model.mesh_vertnum[mesh_id]
    vertices = mj_model.mesh_vert[vert_addr:vert_addr + vert_num].reshape(-1, 3)
    
    print(f"\n顶点坐标 (共{vert_num}个):")
    for i, vertex in enumerate(vertices[:10]):  # 只显示前10个顶点
        print(f"  顶点{i}: [{vertex[0]:.3f}, {vertex[1]:.3f}, {vertex[2]:.3f}]")
    if vert_num > 10:
        print(f"  ... 还有{vert_num-10}个顶点")
    
    # 获取面片数据
    face_addr = mj_model.mesh_faceadr[mesh_id]
    face_num = mj_model.mesh_facenum[mesh_id]
    
    print(f"\n三角形面片 (共{face_num}个):")
    for i in range(min(5, face_num)):  # 只显示前5个面片
        face_idx = face_addr + i
        triangle_indices = mj_model.mesh_face[face_idx]
        triangle_vertices = vertices[triangle_indices]
        print(f"  面片{i}: 顶点索引{triangle_indices}")
        for j, vertex in enumerate(triangle_vertices):
            print(f"    顶点{j}: [{vertex[0]:.3f}, {vertex[1]:.3f}, {vertex[2]:.3f}]")
    if face_num > 5:
        print(f"  ... 还有{face_num-5}个面片")

def get_mesh_triangles_for_taichi(mj_model, mj_scene, mesh_id):
    """
    获取指定mesh的三角形数据，格式适用于taichi的ray_triangle_intersection函数
    
    Returns:
        triangles: list of numpy arrays, 每个array的shape为(3, 3)，表示一个三角形的三个顶点
    """
    triangles = []
    
    # 获取mesh的顶点数据
    vert_addr = mj_model.mesh_vertadr[mesh_id]
    vert_num = mj_model.mesh_vertnum[mesh_id]
    vertices = mj_model.mesh_vert[vert_addr:vert_addr + vert_num].reshape(-1, 3)
    
    # 获取mesh的面片数据
    face_addr = mj_model.mesh_faceadr[mesh_id]
    face_num = mj_model.mesh_facenum[mesh_id]
    
    # 获取对应的几何体变换信息
    mesh_geom_pos = None
    mesh_geom_rot = None
    
    # 在场景中找到对应的几何体
    for geom_idx in range(mj_scene.ngeom):
        geom = mj_scene.geoms[geom_idx]
        if geom.type == 7 and geom.dataid == mesh_id:  # type=7是mesh
            mesh_geom_pos = geom.pos.copy()
            mesh_geom_rot = geom.mat.reshape(3, 3).copy()
            break
    
    # 处理每个三角形面片
    for face_idx in range(face_addr, face_addr + face_num):
        # 获取三角形的三个顶点索引
        triangle_indices = mj_model.mesh_face[face_idx]
        
        # 获取三角形的三个顶点坐标
        triangle_vertices = vertices[triangle_indices]
        
        # 应用几何体的变换（旋转+平移）
        if mesh_geom_pos is not None and mesh_geom_rot is not None:
            # 先应用旋转，再应用平移
            transformed_vertices = (mesh_geom_rot @ triangle_vertices.T).T + mesh_geom_pos
        else:
            transformed_vertices = triangle_vertices
        
        # 转换为适合taichi的格式 (3x3)
        triangle_for_taichi = transformed_vertices.astype(np.float32)
        triangles.append(triangle_for_taichi)
    
    return triangles

# 示例用法
if __name__ == "__main__":
    print("mesh可视化模块已加载。请导入你的MuJoCo模型并调用相应函数。") 