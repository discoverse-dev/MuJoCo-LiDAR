"""
几何体相交测试

为各种几何体类型进行相交测试

支持的几何体类型：
- PLANE (0): 平面
- SPHERE (2): 球体  
- CAPSULE (3): 胶囊体
- ELLIPSOID (4): 椭球体
- CYLINDER (5): 圆柱体
- BOX (6): 盒子
"""

import taichi as ti
from .utils import _transform_ray_to_local, _transform_point_to_world

@ti.func
def ray_geom_intersection(geom_type, ray_start, ray_direction, center, size, rotation):
    """
    射线与几何体相交测试
    """
    if geom_type == 0: # PLANE
        return ray_plane_intersection(ray_start, ray_direction, center, size, rotation)
    elif geom_type == 2: # SPHERE
        return ray_sphere_intersection(ray_start, ray_direction, center, size, rotation)
    elif geom_type == 3: # CAPSULE
        return ray_capsule_intersection(ray_start, ray_direction, center, size, rotation)
    elif geom_type == 4: # ELLIPSOID
        return ray_ellipsoid_intersection(ray_start, ray_direction, center, size, rotation)
    elif geom_type == 5: # CYLINDER
        return ray_cylinder_intersection(ray_start, ray_direction, center, size, rotation)
    elif geom_type == 6: # BOX
        return ray_box_intersection(ray_start, ray_direction, center, size, rotation)
    else: # UNKNOWN
        return ti.math.vec4(0.0, 0.0, 0.0, -1.0)

@ti.func
def ray_plane_intersection(ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
    """计算射线与平面的交点"""
    # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
    
    # 转换射线到平面的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)
    
    # 在局部坐标系中，平面的法向量是z轴
    normal = ti.math.vec3(0.0, 0.0, 1.0)
    half_width = size[0]
    half_height = size[1]
    
    hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
    denom = local_direction.dot(normal)
    
    # 避免除以零，检查光线是否与平面平行
    if ti.abs(denom) >= 1e-6:
        # 局部坐标系中平面在原点，所以只需要计算到原点的距离
        t = -local_start.z / denom
        
        # 如果t为正，表示有有效交点
        if t >= 0:
            local_hit = local_start + t * local_direction
            
            # 检查交点是否在平面范围内
            if ti.abs(local_hit.x) <= half_width and ti.abs(local_hit.y) <= half_height:
                # 将交点转换回世界坐标系
                world_hit = _transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
    
    return hit_result

@ti.func
def ray_sphere_intersection(ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
    """计算射线与球体的交点"""
    # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
    # 注意：球体旋转不会改变其形状，所以可以简化计算
    radius = size[0]
    
    hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
    oc = ray_start - center
    a = ray_direction.dot(ray_direction)
    b = 2.0 * oc.dot(ray_direction)
    c = oc.dot(oc) - radius * radius
    
    discriminant = b * b - 4 * a * c
    
    # 计算交点
    if discriminant >= 0:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        
        # 如果t为负，则使用较大的t值
        if t < 0:
            t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
        
        # 如果t为正，表示有有效交点
        if t >= 0:
            hit_pos = ray_start + t * ray_direction
            hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
    
    return hit_result

@ti.func
def ray_box_intersection(ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
    """计算射线与盒子的交点"""
    # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
    
    # 转换射线到盒子的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)
    
    hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
    
    # 处理局部坐标系中的射线方向为零的情况
    inv_dir = ti.math.vec3(
        1.0 / (local_direction.x if ti.abs(local_direction.x) > 1e-6 else 1e10),
        1.0 / (local_direction.y if ti.abs(local_direction.y) > 1e-6 else 1e10),
        1.0 / (local_direction.z if ti.abs(local_direction.z) > 1e-6 else 1e10)
    )
    
    t_min = -1e10  # 使用大数而不是无穷
    t_max = 1e10
    
    # 检查x轴
    t1 = (-size.x - local_start.x) * inv_dir.x
    t2 = (size.x - local_start.x) * inv_dir.x
    t_min = ti.max(t_min, ti.min(t1, t2))
    t_max = ti.min(t_max, ti.max(t1, t2))
    
    # 检查y轴
    t1 = (-size.y - local_start.y) * inv_dir.y
    t2 = (size.y - local_start.y) * inv_dir.y
    t_min = ti.max(t_min, ti.min(t1, t2))
    t_max = ti.min(t_max, ti.max(t1, t2))
    
    # 检查z轴
    t1 = (-size.z - local_start.z) * inv_dir.z
    t2 = (size.z - local_start.z) * inv_dir.z
    t_min = ti.max(t_min, ti.min(t1, t2))
    t_max = ti.min(t_max, ti.max(t1, t2))
    
    # 如果有有效的交点
    if t_max >= t_min and t_max >= 0:
        t = t_min if t_min >= 0 else t_max
        if t >= 0:
            # 计算局部坐标系中的交点
            local_hit = local_start + t * local_direction
            # 转换回世界坐标系
            world_hit = _transform_point_to_world(local_hit, center, rotation)
            hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
    
    return hit_result

@ti.func
def ray_cylinder_intersection(ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
    """计算射线与圆柱体的交点"""
    # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
    # size[0]是半径，size[1]是半高
    
    # 转换射线到圆柱体的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)
    
    radius = size[0]
    half_height = size[2]
    
    hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
    
    # 在局部坐标系中，圆柱体的中心轴与z轴平行
    # 仅考虑xy平面上的方向分量
    ray_dir_xy = ti.math.vec2(local_direction.x, local_direction.y)
    oc_xy = ti.math.vec2(local_start.x, local_start.y)
    
    # 解二次方程 at² + bt + c = 0
    a = ray_dir_xy.dot(ray_dir_xy)
    
    # 如果a很小，射线几乎与z轴平行
    if a < 1e-6:
        # 检查射线是否在圆柱体内部
        if oc_xy.norm() <= radius:
            # 计算与顶部或底部平面的交点
            t1 = (half_height - local_start.z) / local_direction.z
            t2 = (-half_height - local_start.z) / local_direction.z
            
            # 选择最小的正t值
            t = -1.0  # 默认为无效值
            if t1 >= 0 and (t2 < 0 or t1 < t2):
                t = t1
            elif t2 >= 0:
                t = t2
            
            if t >= 0:
                local_hit = local_start + t * local_direction
                world_hit = _transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
    else:
        # 标准的圆柱体-射线相交测试
        b = 2.0 * oc_xy.dot(ray_dir_xy)
        c = oc_xy.dot(oc_xy) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            # 计算圆柱侧面的两个可能交点
            sqrt_disc = ti.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            
            # 选择最小的正t值
            t = -1.0  # 默认为无效值
            if t1 >= 0:
                t = t1
            elif t2 >= 0:
                t = t2
            
            # 检查交点是否在圆柱体高度范围内
            if t >= 0:
                local_hit = local_start + t * local_direction
                
                if ti.abs(local_hit.z) <= half_height:
                    # 交点在圆柱体侧面上
                    world_hit = _transform_point_to_world(local_hit, center, rotation)
                    hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
                else:
                    # 侧面交点不在圆柱体高度范围内，检查与顶部或底部平面的交点
                    cap_t = -1.0
                    
                    # 射线从上方射向底平面
                    if local_direction.z < 0 and local_start.z > half_height:
                        cap_t = (half_height - local_start.z) / local_direction.z
                    # 射线从下方射向顶平面
                    elif local_direction.z > 0 and local_start.z < -half_height:
                        cap_t = (-half_height - local_start.z) / local_direction.z
                    
                    if cap_t >= 0:
                        local_hit = local_start + cap_t * local_direction
                        cap_xy = ti.math.vec2(local_hit.x, local_hit.y)
                        
                        # 检查交点是否在圆盘内
                        if cap_xy.norm() <= radius:
                            world_hit = _transform_point_to_world(local_hit, center, rotation)
                            hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, cap_t)
    
    return hit_result

@ti.func
def ray_ellipsoid_intersection(ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
    """计算射线与椭球体的交点"""
    # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
    
    # 转换射线到椭球体的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)
    
    hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
    
    # 将问题转换为单位球相交，通过缩放空间
    inv_size = ti.math.vec3(1.0/size.x, 1.0/size.y, 1.0/size.z)
    
    # 缩放局部坐标系中的射线（不要归一化方向向量，这会改变t的意义）
    scaled_start = ti.math.vec3(
        local_start.x * inv_size.x,
        local_start.y * inv_size.y,
        local_start.z * inv_size.z
    )
    scaled_dir = ti.math.vec3(
        local_direction.x * inv_size.x,
        local_direction.y * inv_size.y,
        local_direction.z * inv_size.z
    )
    
    # 解二次方程 at² + bt + c = 0
    a = scaled_dir.dot(scaled_dir)
    b = 2.0 * scaled_start.dot(scaled_dir)
    c = scaled_start.dot(scaled_start) - 1.0  # 单位球半径为1
    
    discriminant = b * b - 4 * a * c
    
    if discriminant >= 0:
        # 计算两个可能的t值，取最小的正值
        t1 = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + ti.sqrt(discriminant)) / (2.0 * a)
        
        t = t1 if t1 >= 0 else t2
        
        # 如果t为正，表示有有效交点
        if t >= 0:
            # 使用原始射线方程计算交点
            local_hit = local_start + t * local_direction
            
            # 转换回世界坐标系
            world_hit = _transform_point_to_world(local_hit, center, rotation)
            hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
    
    return hit_result

@ti.func
def ray_capsule_intersection(ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
    """计算射线与胶囊体的交点"""
    # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
    # 在MuJoCo中: size[0]是半径，size[1]是圆柱部分的半高
    
    # 转换射线到胶囊体的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)
    
    radius = size[0]
    half_height = size[2]
    
    hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
    
    # 计算胶囊体两个半球的中心（在局部坐标系中）
    # 半球中心点在距离胶囊体中心half_height处
    sphere1_center = ti.math.vec3(0.0, 0.0, half_height)
    sphere2_center = ti.math.vec3(0.0, 0.0, -half_height)
    
    # 为圆柱部分创建新的size
    cylinder_size = ti.math.vec3(radius, radius, half_height)
    identity_mat = ti.Matrix.identity(ti.f32, 3)  # 局部坐标系中用单位矩阵
    
    # 首先检查与圆柱体部分的交点（在局部坐标系中）
    cylinder_hit = ray_cylinder_intersection(local_start, local_direction, ti.math.vec3(0.0, 0.0, 0.0), cylinder_size, identity_mat)
    
    # 初始化最小距离为无穷大
    min_t = 1e10
    has_hit = False
    
    # 如果有圆柱体交点
    if cylinder_hit.w > 0 and cylinder_hit.w < min_t:
        min_t = cylinder_hit.w
        
        # 计算世界坐标系中的交点
        local_hit = local_start + cylinder_hit.w * local_direction
        world_hit = _transform_point_to_world(local_hit, center, rotation)
        hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
        has_hit = True
    
    # 然后检查与两个半球的交点
    sphere_size = ti.math.vec3(radius, radius, radius)
    
    # 上半球
    sphere1_hit = ray_sphere_intersection(local_start, local_direction, sphere1_center, sphere_size, identity_mat)
    if sphere1_hit.w > 0 and sphere1_hit.w < min_t:
        # 确保交点在半球内，而不是在完整球体的下半部分
        local_hit = local_start + sphere1_hit.w * local_direction
        local_z = local_hit.z - sphere1_center.z
        if local_z >= 0:  # 只取上半部分
            min_t = sphere1_hit.w
            world_hit = _transform_point_to_world(local_hit, center, rotation)
            hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
            has_hit = True
    
    # 下半球
    sphere2_hit = ray_sphere_intersection(local_start, local_direction, sphere2_center, sphere_size, identity_mat)
    if sphere2_hit.w > 0 and sphere2_hit.w < min_t:
        # 确保交点在半球内，而不是在完整球体的上半部分
        local_hit = local_start + sphere2_hit.w * local_direction
        local_z = local_hit.z - sphere2_center.z
        if local_z <= 0:  # 只取下半部分
            min_t = sphere2_hit.w
            world_hit = _transform_point_to_world(local_hit, center, rotation)
            hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
            has_hit = True
    
    # 如果没有任何交点，返回无效结果
    if not has_hit:
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
    
    return hit_result