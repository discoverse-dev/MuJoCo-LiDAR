import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjtGeom
from functools import partial
from typing import Optional, Tuple, Union

from .geometry import (
    ray_sphere_intersection,
    ray_box_intersection,
    ray_capsule_intersection,
    ray_cylinder_intersection,
    ray_plane_intersection,
    ray_ellipsoid_intersection
)

class MjLidarJax:
    def __init__(self, model: mujoco.MjModel, geom_ids: Optional[Union[np.ndarray, list]] = None, geomgroup: Optional[Union[np.ndarray, list]] = None, bodyexclude: int = -1):
        self.model = model
        
        # If geom_ids is None, use all geoms
        if geom_ids is None:
            self.geom_ids = np.arange(model.ngeom)
        else:
            self.geom_ids = np.array(geom_ids)
            
        # Filter by geomgroup if provided
        if geomgroup is not None:
            geomgroup = np.asarray(geomgroup)
            # model.geom_group is (ngeom,)
            # geomgroup is (mjNGROUP,) where 1 means include
            mask = geomgroup[model.geom_group[self.geom_ids]] == 1
            self.geom_ids = self.geom_ids[mask]

        # Filter by bodyexclude if provided
        if bodyexclude >= 0:
            # model.geom_bodyid is (ngeom,)
            mask = model.geom_bodyid[self.geom_ids] != bodyexclude
            self.geom_ids = self.geom_ids[mask]

        # Extract static properties
        all_types = np.array(model.geom_type)
        
        # Filter by geom_ids
        self.selected_types = all_types[self.geom_ids]
        
        # Group indices by type
        self.sphere_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_SPHERE]
        self.box_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_BOX]
        self.capsule_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_CAPSULE]
        self.cylinder_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_CYLINDER]
        self.plane_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_PLANE]
        self.ellipsoid_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_ELLIPSOID]
        
        # Convert to jnp arrays for JIT
        self.sphere_ids = jnp.array(self.sphere_ids)
        self.box_ids = jnp.array(self.box_ids)
        self.capsule_ids = jnp.array(self.capsule_ids)
        self.cylinder_ids = jnp.array(self.cylinder_ids)
        self.plane_ids = jnp.array(self.plane_ids)
        self.ellipsoid_ids = jnp.array(self.ellipsoid_ids)
        
        # Store sizes (static)
        self.geom_sizes = jnp.array(model.geom_size)
        
    @partial(jax.jit, static_argnums=(0,))
    def render(self, geom_xpos: jax.Array, geom_xmat: jax.Array, rays_origin: jax.Array, rays_direction: jax.Array) -> jax.Array:
        """
        Render LiDAR scan for a single environment.
        
        Args:
            geom_xpos: (Ngeom, 3) Geometry positions
            geom_xmat: (Ngeom, 9) or (Ngeom, 3, 3) Geometry rotation matrices
            rays_origin: (3,) World position of sensor
            rays_direction: (Nrays, 3) World direction of rays
            
        Returns:
            distances: (Nrays,)
        """
        # Handle rotation matrix shape
        if geom_xmat.ndim == 2 and geom_xmat.shape[-1] == 9:
            geom_xmat = geom_xmat.reshape(-1, 3, 3)
            
        # Initialize with inf
        min_dist = jnp.full(rays_direction.shape[0], jnp.inf)
        
        # 1. Spheres
        if self.sphere_ids.shape[0] > 0:
            pos = geom_xpos[self.sphere_ids]
            rad = self.geom_sizes[self.sphere_ids, 0]
            
            def dist_all_rays_all_spheres(ro, rd, pos, rad):
                def scan_fn(carry, x):
                    p, r = x
                    dists = jax.vmap(lambda d: ray_sphere_intersection(ro, d, p, r))(rd)
                    return jnp.minimum(carry, dists), None
                
                final_dist, _ = jax.lax.scan(scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rad))
                return final_dist
            
            d_spheres = dist_all_rays_all_spheres(rays_origin, rays_direction, pos, rad)
            min_dist = jnp.minimum(min_dist, d_spheres)

        # 2. Boxes
        if self.box_ids.shape[0] > 0:
            pos = geom_xpos[self.box_ids]
            rot = geom_xmat[self.box_ids]
            size = self.geom_sizes[self.box_ids]
            
            def dist_all_rays_all_boxes(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_box_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None
                
                final_dist, _ = jax.lax.scan(scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size))
                return final_dist
                
            d_boxes = dist_all_rays_all_boxes(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_boxes)

        # 3. Capsules
        if self.capsule_ids.shape[0] > 0:
            pos = geom_xpos[self.capsule_ids]
            rot = geom_xmat[self.capsule_ids]
            size = self.geom_sizes[self.capsule_ids]
            
            def dist_all_rays_all_capsules(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_capsule_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None
                
                final_dist, _ = jax.lax.scan(scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size))
                return final_dist
                
            d_capsules = dist_all_rays_all_capsules(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_capsules)

        # 4. Cylinders
        if self.cylinder_ids.shape[0] > 0:
            pos = geom_xpos[self.cylinder_ids]
            rot = geom_xmat[self.cylinder_ids]
            size = self.geom_sizes[self.cylinder_ids]
            
            def dist_all_rays_all_cylinders(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_cylinder_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None
                
                final_dist, _ = jax.lax.scan(scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size))
                return final_dist
            
            d_cylinders = dist_all_rays_all_cylinders(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_cylinders)

        # 5. Planes
        if self.plane_ids.shape[0] > 0:
            pos = geom_xpos[self.plane_ids]
            rot = geom_xmat[self.plane_ids]
            size = self.geom_sizes[self.plane_ids]
            
            def dist_all_rays_all_planes(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_plane_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None
                
                final_dist, _ = jax.lax.scan(scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size))
                return final_dist
            
            d_planes = dist_all_rays_all_planes(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_planes)

        # 6. Ellipsoids
        if self.ellipsoid_ids.shape[0] > 0:
            pos = geom_xpos[self.ellipsoid_ids]
            rot = geom_xmat[self.ellipsoid_ids]
            size = self.geom_sizes[self.ellipsoid_ids]
            
            def dist_all_rays_all_ellipsoids(ro, rd, pos, rot, size):
                def scan_fn(carry, x):
                    p, R, s = x
                    dists = jax.vmap(lambda d: ray_ellipsoid_intersection(ro, d, p, R, s))(rd)
                    return jnp.minimum(carry, dists), None
                
                final_dist, _ = jax.lax.scan(scan_fn, jnp.full(rd.shape[0], jnp.inf), (pos, rot, size))
                return final_dist
            
            d_ellipsoids = dist_all_rays_all_ellipsoids(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_ellipsoids)

        # Replace inf with 0.0 (no hit)
        distance = jnp.where(jnp.isinf(min_dist), 0.0, min_dist)
        
        return distance

    @partial(jax.jit, static_argnums=(0,))
    def trace_rays(self, geom_xpos: jax.Array, geom_xmat: jax.Array, sensor_pos: jax.Array, sensor_mat: jax.Array, ray_theta: jax.Array, ray_phi: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Full ray tracing pipeline: ray generation, transformation, and rendering.
        """
        # 1. Ray generation (local space)
        x = jnp.cos(ray_phi) * jnp.cos(ray_theta)
        y = jnp.cos(ray_phi) * jnp.sin(ray_theta)
        z = jnp.sin(ray_phi)
        local_rays = jnp.stack([x, y, z], axis=-1)
        
        # 2. Transform to world rays
        world_rays = local_rays @ sensor_mat.T
        
        # 3. Render
        distances = self.render(geom_xpos, geom_xmat, sensor_pos, world_rays)
        
        return distances, local_rays

    @partial(jax.jit, static_argnums=(0,))
    def render_batch(self, geom_xpos: jax.Array, geom_xmat: jax.Array, rays_origin: jax.Array, rays_direction: jax.Array) -> jax.Array:
        """
        Render LiDAR scan for a batch of environments.
        
        Args:
            geom_xpos: (B, Ngeom, 3) Geometry positions
            geom_xmat: (B, Ngeom, 9) or (B, Ngeom, 3, 3) Geometry rotation matrices
            rays_origin: (B, 3) World position of sensor per env
            rays_direction: (B, Nrays, 3) World direction of rays per env
            
        Returns:
            distances: (B, Nrays)
        """
        return jax.vmap(self.render)(geom_xpos, geom_xmat, rays_origin, rays_direction)

