import jax
import jax.numpy as jnp


def ray_sphere_intersection(
    ray_origin: jax.Array,
    ray_dir: jax.Array,
    sphere_pos: jax.Array,
    sphere_radius: float | jax.Array,
) -> jax.Array:
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


def ray_plane_intersection(
    ray_origin: jax.Array,
    ray_dir: jax.Array,
    plane_pos: jax.Array,
    plane_rot: jax.Array,
    plane_size: jax.Array,
) -> jax.Array:
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


def ray_box_intersection(
    ray_origin: jax.Array,
    ray_dir: jax.Array,
    box_pos: jax.Array,
    box_rot: jax.Array,
    box_size: jax.Array,
) -> jax.Array:
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


def ray_capsule_intersection(
    ray_origin: jax.Array,
    ray_dir: jax.Array,
    cap_pos: jax.Array,
    cap_rot: jax.Array,
    cap_size: jax.Array,
) -> jax.Array:
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
    dist_sq = ro[0] ** 2 + ro[1] ** 2 + (ro[2] - z_clamped) ** 2
    is_inside = dist_sq <= radius**2

    # Capsule = Cylinder (Z-axis) + 2 Spheres

    # 1. Infinite Cylinder Intersection
    # Project to XY plane
    ro_xy = ro[:2]
    rd_xy = rd[:2]

    a = jnp.dot(rd_xy, rd_xy)
    b = 2 * jnp.dot(ro_xy, rd_xy)
    c = jnp.dot(ro_xy, ro_xy) - radius * radius

    delta = b * b - 4 * a * c

    # Cylinder t
    t_cyl = jnp.inf

    # If a is close to 0, ray is parallel to Z axis
    # If parallel and inside radius, it might hit caps.

    # We use a mask for valid cylinder hits
    valid_cyl = (delta >= 0) & (a > 1e-6)
    sqrt_delta = jnp.sqrt(jnp.maximum(0.0, delta))
    t1 = (-b - sqrt_delta) / (2 * a + 1e-10)
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


def ray_cylinder_intersection(
    ray_origin: jax.Array,
    ray_dir: jax.Array,
    cyl_pos: jax.Array,
    cyl_rot: jax.Array,
    cyl_size: jax.Array,
) -> jax.Array:
    """
    Ray-Cylinder intersection (Finite).
    """
    radius = cyl_size[0]
    half_length = cyl_size[1]

    # Transform to local space
    ro = jnp.dot(cyl_rot.T, ray_origin - cyl_pos)
    rd = jnp.dot(cyl_rot.T, ray_dir)

    # Check inside
    inside_xy = (ro[0] ** 2 + ro[1] ** 2) <= radius**2
    inside_z = jnp.abs(ro[2]) <= half_length
    is_inside = inside_xy & inside_z

    # 1. Infinite Cylinder
    ro_xy = ro[:2]
    rd_xy = rd[:2]

    a = jnp.dot(rd_xy, rd_xy)
    b = 2 * jnp.dot(ro_xy, rd_xy)
    c = jnp.dot(ro_xy, ro_xy) - radius * radius

    delta = b * b - 4 * a * c

    valid_cyl = (delta >= 0) & (a > 1e-6)
    sqrt_delta = jnp.sqrt(jnp.maximum(0.0, delta))
    t1 = (-b - sqrt_delta) / (2 * a + 1e-10)
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
    valid_top = (t_top > 0) & (jnp.dot(p_top[:2], p_top[:2]) <= radius * radius)

    # Bottom cap: z = -half_length
    t_bot = (-half_length - ro[2]) / (rd[2] + 1e-10 * jnp.sign(rd[2]))
    p_bot = ro + t_bot * rd
    valid_bot = (t_bot > 0) & (jnp.dot(p_bot[:2], p_bot[:2]) <= radius * radius)

    t_caps = jnp.minimum(jnp.where(valid_top, t_top, jnp.inf), jnp.where(valid_bot, t_bot, jnp.inf))

    t_final = jnp.minimum(t_cyl, t_caps)

    return jnp.where(is_inside, 0.0, t_final)


def ray_ellipsoid_intersection(
    ray_origin: jax.Array,
    ray_dir: jax.Array,
    ell_pos: jax.Array,
    ell_rot: jax.Array,
    ell_size: jax.Array,
) -> jax.Array:
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

    delta = b * b - 4 * a * c

    # Check inside (c <= 0 means inside unit sphere)
    is_inside = c <= 0

    sqrt_delta = jnp.sqrt(jnp.maximum(0.0, delta))
    t1 = (-b - sqrt_delta) / (2 * a + 1e-10)
    # t2 = (-b + sqrt_delta) / (2*a + 1e-10)

    t = jnp.where(t1 > 0, t1, jnp.inf)

    # If inside, return 0.0
    t = jnp.where(is_inside, 0.0, t)

    t = jnp.where(delta < 0, jnp.inf, t)

    return t


def ray_triangle_intersection(
    ray_origin: jax.Array, ray_dir: jax.Array, v0: jax.Array, v1: jax.Array, v2: jax.Array
) -> jax.Array:
    """
    Ray-Triangle intersection using Moller-Trumbore algorithm.
    """
    epsilon = 1e-6
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = jnp.cross(ray_dir, edge2)
    a = jnp.dot(edge1, h)

    # Parallel check
    # if a > -epsilon and a < epsilon: return inf

    f = 1.0 / (a + 1e-10 * jnp.sign(a))
    s = ray_origin - v0
    u = f * jnp.dot(s, h)

    q = jnp.cross(s, edge1)
    v = f * jnp.dot(ray_dir, q)

    t = f * jnp.dot(edge2, q)

    # Check validity
    mask = (
        (jnp.abs(a) > epsilon)
        & (u >= 0.0)
        & (u <= 1.0)
        & (v >= 0.0)
        & (u + v <= 1.0)
        & (t > epsilon)
    )

    return jnp.where(mask, t, jnp.inf)


def ray_hfield_intersection(
    ray_origin: jax.Array,
    ray_dir: jax.Array,
    hfield_pos: jax.Array,
    hfield_rot: jax.Array,
    hfield_size: jax.Array,
    hfield_data: jax.Array,
    hfield_nrow: int | jax.Array = None,
    hfield_ncol: int | jax.Array = None,
) -> jax.Array:
    """
    Ray-Heightfield intersection.

    Args:
        ray_origin: (3,)
        ray_dir: (3,)
        hfield_pos: (3,)
        hfield_rot: (3, 3)
        hfield_size: (4,) (radius_x, radius_y, elevation_z, base_z)
        hfield_data: (nrow, ncol) Normalized elevation data
        hfield_nrow: int (optional)
        hfield_ncol: int (optional)
    """
    rx, ry, ez, bz = hfield_size[0], hfield_size[1], hfield_size[2], hfield_size[3]

    if hfield_nrow is None:
        nrow = hfield_data.shape[0]
    else:
        nrow = hfield_nrow

    if hfield_ncol is None:
        ncol = hfield_data.shape[1]
    else:
        ncol = hfield_ncol

    # Transform to local space
    ro = jnp.dot(hfield_rot.T, ray_origin - hfield_pos)
    rd = jnp.dot(hfield_rot.T, ray_dir)

    # AABB Intersection
    aabb_min = jnp.array([-rx, -ry, -bz])
    aabb_max = jnp.array([rx, ry, ez])

    inv_rd = 1.0 / (rd + 1e-10 * jnp.sign(rd))
    t1 = (aabb_min - ro) * inv_rd
    t2 = (aabb_max - ro) * inv_rd

    t_min = jnp.minimum(t1, t2)
    t_max = jnp.maximum(t1, t2)

    t_enter = jnp.max(t_min)
    t_exit = jnp.min(t_max)

    hit_aabb = (t_enter <= t_exit) & (t_exit > 0)

    # If no hit, return inf
    # We will mask the result at the end

    # Start point for traversal
    t_start = jnp.maximum(0.0, t_enter)
    p_start = ro + t_start * rd

    # Grid parameters
    dx = 2 * rx / (ncol - 1)
    dy = 2 * ry / (nrow - 1)

    # Check if we hit the "underground" part at entry
    # Map p_start to grid
    u_start = (p_start[0] + rx) / dx
    v_start = (p_start[1] + ry) / dy

    ix_start = jnp.clip(jnp.floor(u_start).astype(jnp.int32), 0, ncol - 2)
    iy_start = jnp.clip(jnp.floor(v_start).astype(jnp.int32), 0, nrow - 2)

    h_start = hfield_data[iy_start, ix_start] * ez

    # If p_start.z < h_start, we are hitting the side wall or base
    # (Assuming we hit the AABB, so we are within XY bounds)
    hit_underground = p_start[2] < h_start

    # Ray Marching Setup
    # We use a DDA-like approach

    # Initial cell
    ix = ix_start
    iy = iy_start

    # Step direction
    step_x = jnp.where(rd[0] >= 0, 1, -1)
    step_y = jnp.where(rd[1] >= 0, 1, -1)

    # Distance to next boundary
    # next_x_boundary = -rx + (ix + (1 if step_x > 0 else 0)) * dx
    next_x_idx = ix + jnp.where(step_x > 0, 1, 0)
    next_y_idx = iy + jnp.where(step_y > 0, 1, 0)

    next_x = -rx + next_x_idx * dx
    next_y = -ry + next_y_idx * dy

    t_max_x = (next_x - ro[0]) / (rd[0] + 1e-10 * jnp.sign(rd[0]))
    t_max_y = (next_y - ro[1]) / (rd[1] + 1e-10 * jnp.sign(rd[1]))

    t_delta_x = jnp.abs(dx / (rd[0] + 1e-10 * jnp.sign(rd[0])))
    t_delta_y = jnp.abs(dy / (rd[1] + 1e-10 * jnp.sign(rd[1])))

    # Loop state: (t_curr, ix, iy, t_max_x, t_max_y, hit_found, t_hit)
    init_val = (t_start, ix, iy, t_max_x, t_max_y, False, jnp.inf)

    def cond_fun(val):
        t, ix, iy, _, _, hit, _ = val
        # Continue if:
        # 1. Not hit yet
        # 2. t < t_exit
        # 3. Indices within bounds (0 to ncol-2, 0 to nrow-2 for cells)
        # Note: We need to check triangles in cell (ix, iy).
        # Valid cell indices are 0..ncol-2 and 0..nrow-2.
        # If ix == ncol-1, we are at the edge, maybe just exit?
        in_bounds = (ix >= 0) & (ix < ncol - 1) & (iy >= 0) & (iy < nrow - 1)
        return (~hit) & (t < t_exit + 1e-3) & in_bounds

    def body_fun(val):
        t_curr, ix, iy, t_mx, t_my, hit, t_h = val

        # Check intersection with triangles in current cell (ix, iy)
        # Vertices
        x0 = -rx + ix * dx
        y0 = -ry + iy * dy
        x1 = x0 + dx
        y1 = y0 + dy

        h00 = hfield_data[iy, ix] * ez
        h10 = hfield_data[iy, ix + 1] * ez
        h01 = hfield_data[iy + 1, ix] * ez
        h11 = hfield_data[iy + 1, ix + 1] * ez

        v00 = jnp.array([x0, y0, h00])
        v10 = jnp.array([x1, y0, h10])
        v01 = jnp.array([x0, y1, h01])
        v11 = jnp.array([x1, y1, h11])

        # Triangle 1: v00, v10, v01
        t_tri1 = ray_triangle_intersection(ro, rd, v00, v10, v01)

        # Triangle 2: v10, v11, v01
        t_tri2 = ray_triangle_intersection(ro, rd, v10, v11, v01)

        t_cell = jnp.minimum(t_tri1, t_tri2)

        # If hit within this cell?
        # ray_triangle_intersection returns inf if no hit.
        # We should check if t_cell is reasonable (e.g. >= t_curr - epsilon)
        # But ray_triangle checks if point is inside triangle.

        found = t_cell < jnp.inf

        # Update step
        # If t_mx < t_my, step X
        step_x_cond = t_mx < t_my

        next_t = jnp.minimum(t_mx, t_my)
        next_ix = ix + jnp.where(step_x_cond, step_x, 0)
        next_iy = iy + jnp.where(step_x_cond, 0, step_y)
        next_t_mx = t_mx + jnp.where(step_x_cond, t_delta_x, 0)
        next_t_my = t_my + jnp.where(step_x_cond, 0, t_delta_y)

        return (next_t, next_ix, next_iy, next_t_mx, next_t_my, found, t_cell)

    # Only march if not hit underground at start
    # And if we are not already outside (t_start > t_exit handled by cond)

    # We need to handle the case where we start "above" the terrain but inside AABB.
    # If hit_underground is True, we return t_start.
    # Else we march.

    final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)

    _, _, _, _, _, hit_march, t_march = final_val

    # Result logic
    # If hit_underground: t = t_start
    # Else if hit_march: t = t_march
    # Else: inf

    t_final = jnp.where(hit_underground, t_start, jnp.where(hit_march, t_march, jnp.inf))

    # Mask with AABB hit
    return jnp.where(hit_aabb, t_final, jnp.inf)
