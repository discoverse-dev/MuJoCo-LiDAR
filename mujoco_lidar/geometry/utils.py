import taichi as ti

@ti.func
def _transform_ray_to_local(ray_start, ray_direction, center, rotation):
    """将射线从世界坐标系转换到物体的局部坐标系"""
    # 先平移射线起点
    local_start = ray_start - center
    
    # 旋转矩阵的转置是其逆（假设正交矩阵）
    rot_transpose = ti.Matrix.zero(ti.f32, 3, 3)
    for i in range(3):
        for j in range(3):
            rot_transpose[i, j] = rotation[j, i]
    
    # 应用旋转
    local_start = rot_transpose @ local_start
    local_direction = rot_transpose @ ray_direction
    
    return local_start, local_direction

@ti.func
def _transform_point_to_world(local_point, center, rotation):
    """将点从局部坐标系转换回世界坐标系"""
    # 应用旋转
    world_point = rotation @ local_point
    # 应用平移
    world_point = world_point + center
    return world_point