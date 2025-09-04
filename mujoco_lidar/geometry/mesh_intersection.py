import taichi as ti

@ti.func
def ray_triangle_distance(ray_start, ray_direction, v0, v1, v2):
    """返回射线与三角形的命中距离t，未命中返回-1.0
    使用 Möller-Trumbore 算法
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = ray_direction.cross(edge2)
    a = edge1.dot(h)
    t_ret = -1.0
    if ti.abs(a) >= 1e-6:  # 不平行
        f = 1.0 / a
        s = ray_start - v0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(edge1)
            v = f * ray_direction.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                t = f * edge2.dot(q)
                if t > 1e-6:  # 正向命中
                    t_ret = t
    return t_ret
