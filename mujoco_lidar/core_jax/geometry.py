import jax
import jax.numpy as jnp

def ray_sphere_intersection(ray_origin, ray_dir, sphere_pos, sphere_radius):
    """
    Calculate intersection between a ray and a sphere.
    
    Args:
        ray_origin: (3,) Ray origin
        ray_dir: (3,) Ray direction (normalized)
        sphere_pos: (3,) Sphere center
        sphere_radius: scalar Sphere radius
        
    Returns:
        t: scalar Distance to intersection (inf if no intersection)
    """
    m = ray_origin - sphere_pos
    b = jnp.dot(m, ray_dir)
    c = jnp.dot(m, m) - sphere_radius * sphere_radius
    delta = b * b - c
    
    # If delta < 0, no intersection
    # If delta >= 0, two roots: -b +/- sqrt(delta)
    
    sqrt_delta = jnp.sqrt(jnp.maximum(0.0, delta))
    t1 = -b - sqrt_delta
    # t2 = -b + sqrt_delta
    
    # Check if inside
    dist_sq = jnp.dot(m, m)
    is_inside = dist_sq <= sphere_radius * sphere_radius
    
    # If outside, we want the first hit t1
    t = jnp.where(t1 > 0, t1, jnp.inf)
    
    # If inside, return 0.0
    t = jnp.where(is_inside, 0.0, t)
    
    t = jnp.where(delta < 0, jnp.inf, t)
    
    return t

def ray_plane_intersection(ray_origin, ray_dir, plane_pos, plane_rot, plane_size):
    """
    Calculate intersection between a ray and a plane.
    Plane is defined by position, rotation, and half-sizes.
    Local Z axis is normal.
    
    Args:
        ray_origin: (3,)
        ray_dir: (3,)
        plane_pos: (3,) Plane center
        plane_rot: (3, 3) Plane rotation
        plane_size: (3,) Plane half-sizes (x, y, z ignored)
        
    Returns:
        t: scalar
    """
    # Transform to local space
    ro = jnp.dot(plane_rot.T, ray_origin - plane_pos)
    rd = jnp.dot(plane_rot.T, ray_dir)
    
    # Intersection with z=0 plane
    # ro.z + t * rd.z = 0  =>  t = -ro.z / rd.z
    
    denom = rd[2]
    
    # Avoid division by zero
    safe_denom = denom + 1e-10 * jnp.sign(denom)
    t = -ro[2] / safe_denom
    
    # Check bounds
    hit_pos = ro + t * rd
    
    hx = plane_size[0]
    hy = plane_size[1]
    
    # Check if infinite (size=0)
    is_infinite = (hx == 0.0) & (hy == 0.0)
    
    in_bounds = (jnp.abs(hit_pos[0]) <= hx) & (jnp.abs(hit_pos[1]) <= hy)
    
    # Valid if t > 0 and (in_bounds or infinite) and not parallel
    is_parallel = jnp.abs(denom) < 1e-6
    
    valid = (t > 0) & (in_bounds | is_infinite) & (~is_parallel)
    
    return jnp.where(valid, t, jnp.inf)

def ray_box_intersection(ray_origin, ray_dir, box_pos, box_rot, box_size):
    """
    Ray-Box intersection using Slab method.
    Box is defined by center, rotation matrix, and half-sizes.
    
    Args:
        ray_origin: (3,)
        ray_dir: (3,)
        box_pos: (3,) Box center
        box_rot: (3, 3) Rotation matrix (local to world)
        box_size: (3,) Half-sizes (x, y, z)
    """
    # Transform ray to box local space
    # p_local = R^T * (p_world - box_pos)
    # dir_local = R^T * dir_world
    
    ray_origin_local = jnp.dot(box_rot.T, ray_origin - box_pos)
    ray_dir_local = jnp.dot(box_rot.T, ray_dir)
    
    # Slab method
    # t = (plane - origin) / dir
    
    # Avoid division by zero
    inv_dir = 1.0 / (ray_dir_local + 1e-10 * jnp.sign(ray_dir_local))
    
    t1 = (-box_size - ray_origin_local) * inv_dir
    t2 = (box_size - ray_origin_local) * inv_dir
    
    t_min = jnp.minimum(t1, t2)
    t_max = jnp.maximum(t1, t2)
    
    t_enter = jnp.max(t_min)
    t_exit = jnp.min(t_max)
    
    # Hit if t_enter <= t_exit and t_exit > 0
    hit = (t_enter <= t_exit) & (t_exit > 0)
    
    # If inside (t_enter < 0), return t_exit? No, usually we want entry point.
    # If origin is inside, t_enter will be negative.
    # If we want the first hit point:
    # If t_enter > 0: return t_enter
    # If t_enter <= 0 and t_exit > 0: return 0.0 (inside) or t_exit? 
    # MuJoCo usually returns 0 if inside, or we can return t_exit if we want to see "out".
    # Let's assume we want the first positive intersection.
    
    t = jnp.where(hit, jnp.where(t_enter > 0, t_enter, 0.0), jnp.inf)
    
    return t

def ray_capsule_intersection(ray_origin, ray_dir, cap_pos, cap_rot, cap_size):
    """
    Ray-Capsule intersection.
    Capsule is aligned with local Z axis (usually).
    MuJoCo capsules: size[0] is radius, size[1] is half-length (cylinder part).
    
    Args:
        ray_origin: (3,)
        ray_dir: (3,)
        cap_pos: (3,)
        cap_rot: (3, 3)
        cap_size: (3,) radius, half_length
    """
    radius = cap_size[0]
    half_length = cap_size[1]
    
    # Transform to local space
    ro = jnp.dot(cap_rot.T, ray_origin - cap_pos)
    rd = jnp.dot(cap_rot.T, ray_dir)
    
    # Check inside
    # Segment from (0,0,-hl) to (0,0,hl)
    z_clamped = jnp.clip(ro[2], -half_length, half_length)
    dist_sq = ro[0]**2 + ro[1]**2 + (ro[2] - z_clamped)**2
    is_inside = dist_sq <= radius**2
    
    # Capsule = Cylinder (Z-axis) + 2 Spheres
    
    # 1. Infinite Cylinder Intersection
    # Project to XY plane
    ro_xy = ro[:2]
    rd_xy = rd[:2]
    
    a = jnp.dot(rd_xy, rd_xy)
    b = 2 * jnp.dot(ro_xy, rd_xy)
    c = jnp.dot(ro_xy, ro_xy) - radius * radius
    
    delta = b*b - 4*a*c
    
    # Cylinder t
    t_cyl = jnp.inf
    
    # If a is close to 0, ray is parallel to Z axis
    # If parallel and inside radius, it might hit caps.
    
    # We use a mask for valid cylinder hits
    valid_cyl = (delta >= 0) & (a > 1e-6)
    sqrt_delta = jnp.sqrt(jnp.maximum(0.0, delta))
    t1 = (-b - sqrt_delta) / (2*a + 1e-10)
    # t2 = (-b + sqrt_delta) / (2*a + 1e-10)
    
    # Check z bounds for cylinder hits
    z1 = ro[2] + t1 * rd[2]
    # z2 = ro[2] + t2 * rd[2]
    
    in_bounds1 = jnp.abs(z1) <= half_length
    # in_bounds2 = jnp.abs(z2) <= half_length
    
    # t_cyl_cand = jnp.where(in_bounds1 & (t1 > 0), t1, jnp.where(in_bounds2 & (t2 > 0), t2, jnp.inf))
    t_cyl = jnp.where(valid_cyl & in_bounds1 & (t1 > 0), t1, jnp.inf)
    
    # 2. Sphere Caps Intersections
    # Top sphere: center (0, 0, half_length), radius
    # Bottom sphere: center (0, 0, -half_length), radius
    
    def intersect_local_sphere(center):
        m = ro - center
        b_s = jnp.dot(m, rd)
        c_s = jnp.dot(m, m) - radius * radius
        delta_s = b_s * b_s - c_s
        
        sqrt_d_s = jnp.sqrt(jnp.maximum(0.0, delta_s))
        t1_s = -b_s - sqrt_d_s
        # t2_s = -b_s + sqrt_d_s
        
        return jnp.where((delta_s >= 0) & (t1_s > 0), t1_s, jnp.inf)

    t_top = intersect_local_sphere(jnp.array([0.0, 0.0, half_length]))
    t_bottom = intersect_local_sphere(jnp.array([0.0, 0.0, -half_length]))
    
    t_final = jnp.minimum(t_cyl, jnp.minimum(t_top, t_bottom))
    
    return jnp.where(is_inside, 0.0, t_final)

def ray_cylinder_intersection(ray_origin, ray_dir, cyl_pos, cyl_rot, cyl_size):
    """
    Ray-Cylinder intersection (Finite).
    """
    radius = cyl_size[0]
    half_length = cyl_size[1]
    
    # Transform to local space
    ro = jnp.dot(cyl_rot.T, ray_origin - cyl_pos)
    rd = jnp.dot(cyl_rot.T, ray_dir)
    
    # Check inside
    inside_xy = (ro[0]**2 + ro[1]**2) <= radius**2
    inside_z = jnp.abs(ro[2]) <= half_length
    is_inside = inside_xy & inside_z
    
    # 1. Infinite Cylinder
    ro_xy = ro[:2]
    rd_xy = rd[:2]
    
    a = jnp.dot(rd_xy, rd_xy)
    b = 2 * jnp.dot(ro_xy, rd_xy)
    c = jnp.dot(ro_xy, ro_xy) - radius * radius
    
    delta = b*b - 4*a*c
    
    valid_cyl = (delta >= 0) & (a > 1e-6)
    sqrt_delta = jnp.sqrt(jnp.maximum(0.0, delta))
    t1 = (-b - sqrt_delta) / (2*a + 1e-10)
    # t2 = (-b + sqrt_delta) / (2*a + 1e-10) # We usually hit the outside first
    
    z1 = ro[2] + t1 * rd[2]
    in_bounds1 = jnp.abs(z1) <= half_length
    
    t_cyl = jnp.where(valid_cyl & in_bounds1 & (t1 > 0), t1, jnp.inf)
    
    # 2. Flat Caps (Planes at z = +/- half_length)
    # Plane normal (0,0,1) and (0,0,-1)
    
    # Top cap: z = half_length
    # t = (half_length - ro_z) / rd_z
    t_top = (half_length - ro[2]) / (rd[2] + 1e-10 * jnp.sign(rd[2]))
    p_top = ro + t_top * rd
    valid_top = (t_top > 0) & (jnp.dot(p_top[:2], p_top[:2]) <= radius*radius)
    
    # Bottom cap: z = -half_length
    t_bot = (-half_length - ro[2]) / (rd[2] + 1e-10 * jnp.sign(rd[2]))
    p_bot = ro + t_bot * rd
    valid_bot = (t_bot > 0) & (jnp.dot(p_bot[:2], p_bot[:2]) <= radius*radius)
    
    t_caps = jnp.minimum(
        jnp.where(valid_top, t_top, jnp.inf),
        jnp.where(valid_bot, t_bot, jnp.inf)
    )
    
    t_final = jnp.minimum(t_cyl, t_caps)
    
    return jnp.where(is_inside, 0.0, t_final)

def ray_ellipsoid_intersection(ray_origin, ray_dir, ell_pos, ell_rot, ell_size):
    """
    Ray-Ellipsoid intersection.
    """
    # Transform to local space
    ro = jnp.dot(ell_rot.T, ray_origin - ell_pos)
    rd = jnp.dot(ell_rot.T, ray_dir)
    
    # Scale to unit sphere space
    # ell_size is (rx, ry, rz)
    inv_size = 1.0 / (ell_size + 1e-10)
    
    ro_scaled = ro * inv_size
    rd_scaled = rd * inv_size
    
    # Intersection with unit sphere
    a = jnp.dot(rd_scaled, rd_scaled)
    b = 2.0 * jnp.dot(ro_scaled, rd_scaled)
    c = jnp.dot(ro_scaled, ro_scaled) - 1.0
    
    delta = b*b - 4*a*c
    
    # Check inside (c <= 0 means inside unit sphere)
    is_inside = c <= 0
    
    sqrt_delta = jnp.sqrt(jnp.maximum(0.0, delta))
    t1 = (-b - sqrt_delta) / (2*a + 1e-10)
    # t2 = (-b + sqrt_delta) / (2*a + 1e-10)
    
    t = jnp.where(t1 > 0, t1, jnp.inf)
    
    # If inside, return 0.0
    t = jnp.where(is_inside, 0.0, t)
    
    t = jnp.where(delta < 0, jnp.inf, t)
    
    return t
