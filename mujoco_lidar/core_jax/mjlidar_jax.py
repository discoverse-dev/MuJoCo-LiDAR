import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjtGeom
from functools import partial

from .geometry import (
    ray_sphere_intersection,
    ray_box_intersection,
    ray_capsule_intersection,
    ray_cylinder_intersection,
    ray_plane_intersection,
    ray_ellipsoid_intersection
)

class MjLidarJax:
    def __init__(self, model, geom_ids=None):
        self.model = model
        
        # If geom_ids is None, use all geoms
        if geom_ids is None:
            self.geom_ids = np.arange(model.ngeom)
        else:
            self.geom_ids = np.array(geom_ids)
            
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
    def render(self, geom_xpos, geom_xmat, rays_origin, rays_direction):
        """
        Render LiDAR scan.
        
        Args:
            geom_xpos: (Ngeom, 3) Geometry positions
            geom_xmat: (Ngeom, 9) Geometry rotation matrices (flattened)
            rays_origin: (3,) World position of sensor
            rays_direction: (N, 3) World direction of rays
            
        Returns:
            distances: (N,)
        """
        # Initialize with inf
        min_dist = jnp.full(rays_direction.shape[0], jnp.inf)
        
        # 1. Spheres
        if self.sphere_ids.shape[0] > 0:
            pos = geom_xpos[self.sphere_ids]
            rad = self.geom_sizes[self.sphere_ids, 0]
            
            def dist_all_rays_all_spheres(ro, rd, pos, rad):
                def dist_rays(p, r):
                    return jax.vmap(lambda d: ray_sphere_intersection(ro, d, p, r))(rd)
                dists = jax.vmap(dist_rays)(pos, rad) # (Nspheres, Nrays)
                return jnp.min(dists, axis=0) # (Nrays,)
            
            d_spheres = dist_all_rays_all_spheres(rays_origin, rays_direction, pos, rad)
            min_dist = jnp.minimum(min_dist, d_spheres)

        # 2. Boxes
        if self.box_ids.shape[0] > 0:
            pos = geom_xpos[self.box_ids]
            rot = geom_xmat[self.box_ids].reshape(-1, 3, 3)
            size = self.geom_sizes[self.box_ids]
            
            def dist_all_rays_all_boxes(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_box_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
                
            d_boxes = dist_all_rays_all_boxes(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_boxes)

        # 3. Capsules
        if self.capsule_ids.shape[0] > 0:
            pos = geom_xpos[self.capsule_ids]
            rot = geom_xmat[self.capsule_ids].reshape(-1, 3, 3)
            size = self.geom_sizes[self.capsule_ids]
            
            def dist_all_rays_all_capsules(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_capsule_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
                
            d_capsules = dist_all_rays_all_capsules(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_capsules)

        # 4. Cylinders
        if self.cylinder_ids.shape[0] > 0:
            pos = geom_xpos[self.cylinder_ids]
            rot = geom_xmat[self.cylinder_ids].reshape(-1, 3, 3)
            size = self.geom_sizes[self.cylinder_ids]
            
            def dist_all_rays_all_cylinders(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_cylinder_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
            
            d_cylinders = dist_all_rays_all_cylinders(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_cylinders)

        # 5. Planes
        if self.plane_ids.shape[0] > 0:
            pos = geom_xpos[self.plane_ids]
            rot = geom_xmat[self.plane_ids].reshape(-1, 3, 3)
            size = self.geom_sizes[self.plane_ids]
            
            def dist_all_rays_all_planes(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_plane_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
            
            d_planes = dist_all_rays_all_planes(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_planes)

        # 6. Ellipsoids
        if self.ellipsoid_ids.shape[0] > 0:
            pos = geom_xpos[self.ellipsoid_ids]
            rot = geom_xmat[self.ellipsoid_ids].reshape(-1, 3, 3)
            size = self.geom_sizes[self.ellipsoid_ids]
            
            def dist_all_rays_all_ellipsoids(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_ellipsoid_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
            
            d_ellipsoids = dist_all_rays_all_ellipsoids(rays_origin, rays_direction, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_ellipsoids)

        # Replace inf with 0.0 (no hit)
        return jnp.where(jnp.isinf(min_dist), 0.0, min_dist)

