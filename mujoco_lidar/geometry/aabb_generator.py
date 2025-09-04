"""
几何体AABB生成器

为各种几何体类型生成轴对齐包围盒(AABB)

支持的几何体类型：
- TRIANGLE: 三角形 为mesh三角形提供AABB计算

- PLANE (0): 平面
- SPHERE (2): 球体  
- CAPSULE (3): 胶囊体
- ELLIPSOID (4): 椭球体
- CYLINDER (5): 圆柱体
- BOX (6): 盒子
- MESH (7): 网格
"""

import taichi as ti

@ti.func
def compute_triangle_aabb(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3) -> tuple:
    """
    直接从三角形的三个顶点计算AABB
    
    Args:
        v0, v1, v2: 三角形的三个顶点坐标
        
    Returns:
        tuple: (aabb_min, aabb_max)
    """
    # 计算三个顶点的最小值和最大值
    aabb_min = ti.Vector([
        ti.min(ti.min(v0.x, v1.x), v2.x),
        ti.min(ti.min(v0.y, v1.y), v2.y),
        ti.min(ti.min(v0.z, v1.z), v2.z)
    ])
    
    aabb_max = ti.Vector([
        ti.max(ti.max(v0.x, v1.x), v2.x),
        ti.max(ti.max(v0.y, v1.y), v2.y),
        ti.max(ti.max(v0.z, v1.z), v2.z)
    ])
    
    return aabb_min, aabb_max


@ti.func
def compute_geometry_aabb(geom_type, position, rotation, size) -> tuple:
    """
    计算单个几何体的AABB
    
    Args:
        geom_type: 几何体类型
        position: 几何体位置
        rotation: 几何体旋转
        size: 几何体尺寸
        
    Returns:
        tuple: (aabb_min, aabb_max)
    """
    
    # 初始化默认AABB
    aabb_min = position - ti.Vector([0.01, 0.01, 0.01])
    aabb_max = position + ti.Vector([0.01, 0.01, 0.01])
    
    # 根据几何体类型计算AABB
    if geom_type == 0:  # PLANE
        aabb_min, aabb_max = _compute_plane_aabb(position, rotation, size)
    elif geom_type == 2:  # SPHERE
        aabb_min, aabb_max = _compute_sphere_aabb(position, size)
    elif geom_type == 3:  # CAPSULE
        aabb_min, aabb_max = _compute_capsule_aabb(position, rotation, size)
    elif geom_type == 4:  # ELLIPSOID
        aabb_min, aabb_max = _compute_ellipsoid_aabb(position, rotation, size)
    elif geom_type == 5:  # CYLINDER
        aabb_min, aabb_max = _compute_cylinder_aabb(position, rotation, size)
    elif geom_type == 6:  # BOX
        aabb_min, aabb_max = _compute_box_aabb(position, rotation, size)
    
    return aabb_min, aabb_max

@ti.func
def _compute_plane_aabb(position: ti.types.vector(3, ti.f32), 
                        rotation: ti.types.matrix(3, 3, ti.f32),
                        size: ti.types.vector(3, ti.f32)) -> tuple:
    """计算平面AABB"""
    # 平面的AABB需要考虑其法向量和尺寸
    # size[0] = width, size[1] = height
    half_width = size[0] if size[0] > 0 else 1000.0  # 默认大平面
    half_height = size[1] if size[1] > 0 else 1000.0
    thickness = 0.01  # 平面厚度
    
    # 局部坐标系中的AABB顶点
    local_vertices = ti.Matrix([
        [-half_width, -half_height, -thickness],
        [half_width, -half_height, -thickness],
        [-half_width, half_height, -thickness],
        [half_width, half_height, -thickness],
        [-half_width, -half_height, thickness],
        [half_width, -half_height, thickness],
        [-half_width, half_height, thickness],
        [half_width, half_height, thickness]
    ])
    
    # 转换到世界坐标并计算AABB
    aabb_min = ti.Vector([1e10, 1e10, 1e10])
    aabb_max = ti.Vector([-1e10, -1e10, -1e10])
    
    for i in range(8):
        world_vertex = rotation @ ti.Vector([local_vertices[i, 0], local_vertices[i, 1], local_vertices[i, 2]]) + position
        aabb_min = ti.min(aabb_min, world_vertex)
        aabb_max = ti.max(aabb_max, world_vertex)
    
    return aabb_min, aabb_max

@ti.func
def _compute_sphere_aabb(position: ti.types.vector(3, ti.f32),
                        size: ti.types.vector(3, ti.f32)) -> tuple:
    """计算球体AABB"""
    radius = size[0]
    radius_vec = ti.Vector([radius, radius, radius])
    
    aabb_min = position - radius_vec
    aabb_max = position + radius_vec
    
    return aabb_min, aabb_max

@ti.func
def _compute_capsule_aabb(position: ti.types.vector(3, ti.f32),
                        rotation: ti.types.matrix(3, 3, ti.f32),
                        size: ti.types.vector(3, ti.f32)) -> tuple:
    """计算胶囊体AABB"""
    # size[0] = radius, size[2] = half_height
    radius = size[0]
    half_height = size[2]
    
    # 胶囊体的总高度（包括两端半球）
    total_half_height = half_height + radius
    
    # 局部坐标系中的关键点（胶囊体沿z轴）
    key_points = ti.Matrix([
        [0.0, 0.0, total_half_height],   # 顶端
        [0.0, 0.0, -total_half_height],  # 底端
        [radius, 0.0, 0.0],              # x正方向
        [-radius, 0.0, 0.0],             # x负方向
        [0.0, radius, 0.0],              # y正方向
        [0.0, -radius, 0.0]              # y负方向
    ])
    
    # 转换到世界坐标并计算AABB
    aabb_min = ti.Vector([1e10, 1e10, 1e10])
    aabb_max = ti.Vector([-1e10, -1e10, -1e10])
    
    for i in range(6):
        world_point = rotation @ ti.Vector([key_points[i, 0], key_points[i, 1], key_points[i, 2]]) + position
        aabb_min = ti.min(aabb_min, world_point)
        aabb_max = ti.max(aabb_max, world_point)
    
    return aabb_min, aabb_max

@ti.func
def _compute_ellipsoid_aabb(position: ti.types.vector(3, ti.f32),
                            rotation: ti.types.matrix(3, 3, ti.f32),
                            size: ti.types.vector(3, ti.f32)) -> tuple:
    """计算椭球体AABB"""
    # size包含三个轴的半径
    radii = size
    
    # 椭球体的关键点（沿三个主轴）
    key_points = ti.Matrix([
        [radii[0], 0.0, 0.0],     # x轴正方向
        [-radii[0], 0.0, 0.0],    # x轴负方向
        [0.0, radii[1], 0.0],     # y轴正方向
        [0.0, -radii[1], 0.0],    # y轴负方向
        [0.0, 0.0, radii[2]],     # z轴正方向
        [0.0, 0.0, -radii[2]]     # z轴负方向
    ])
    
    # 转换到世界坐标并计算AABB
    aabb_min = ti.Vector([1e10, 1e10, 1e10])
    aabb_max = ti.Vector([-1e10, -1e10, -1e10])
    
    for i in range(6):
        world_point = rotation @ ti.Vector([key_points[i, 0], key_points[i, 1], key_points[i, 2]]) + position
        aabb_min = ti.min(aabb_min, world_point)
        aabb_max = ti.max(aabb_max, world_point)
    
    return aabb_min, aabb_max

@ti.func
def _compute_cylinder_aabb(position: ti.types.vector(3, ti.f32),
                            rotation: ti.types.matrix(3, 3, ti.f32),
                            size: ti.types.vector(3, ti.f32)) -> tuple:
    """计算圆柱体AABB"""
    # size[0] = radius, size[2] = half_height
    radius = size[0]
    half_height = size[2]
    
    # 圆柱体的关键点（底面圆周 + 顶底面中心）
    n_circle_points = 8  # 圆周采样点数
    
    aabb_min = ti.Vector([1e10, 1e10, 1e10])
    aabb_max = ti.Vector([-1e10, -1e10, -1e10])
    
    # 采样圆周点
    for i in range(n_circle_points):
        angle = 2.0 * 3.14159 * i / n_circle_points
        cos_a = ti.cos(angle)
        sin_a = ti.sin(angle)
        
        # 顶面圆周
        local_point_top = ti.Vector([radius * cos_a, radius * sin_a, half_height])
        world_point_top = rotation @ local_point_top + position
        aabb_min = ti.min(aabb_min, world_point_top)
        aabb_max = ti.max(aabb_max, world_point_top)
        
        # 底面圆周
        local_point_bottom = ti.Vector([radius * cos_a, radius * sin_a, -half_height])
        world_point_bottom = rotation @ local_point_bottom + position
        aabb_min = ti.min(aabb_min, world_point_bottom)
        aabb_max = ti.max(aabb_max, world_point_bottom)
    
    return aabb_min, aabb_max

@ti.func
def _compute_box_aabb(position: ti.types.vector(3, ti.f32),
                    rotation: ti.types.matrix(3, 3, ti.f32),
                    size: ti.types.vector(3, ti.f32)) -> tuple:
    """计算盒子AABB"""
    # size包含三个方向的半长度
    half_sizes = size
    
    # 盒子的8个顶点
    vertices = ti.Matrix([
        [-half_sizes[0], -half_sizes[1], -half_sizes[2]],
        [half_sizes[0], -half_sizes[1], -half_sizes[2]],
        [-half_sizes[0], half_sizes[1], -half_sizes[2]],
        [half_sizes[0], half_sizes[1], -half_sizes[2]],
        [-half_sizes[0], -half_sizes[1], half_sizes[2]],
        [half_sizes[0], -half_sizes[1], half_sizes[2]],
        [-half_sizes[0], half_sizes[1], half_sizes[2]],
        [half_sizes[0], half_sizes[1], half_sizes[2]]
    ])
    
    # 转换到世界坐标并计算AABB
    aabb_min = ti.Vector([1e10, 1e10, 1e10])
    aabb_max = ti.Vector([-1e10, -1e10, -1e10])
    
    for i in range(8):
        world_vertex = rotation @ ti.Vector([vertices[i, 0], vertices[i, 1], vertices[i, 2]]) + position
        aabb_min = ti.min(aabb_min, world_vertex)
        aabb_max = ti.max(aabb_max, world_vertex)
    
    return aabb_min, aabb_max
