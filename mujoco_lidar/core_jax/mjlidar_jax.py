import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco import mjtGeom
from functools import partial

from .geometry import (
    ray_sphere_intersection,
    ray_box_intersection,
    ray_capsule_intersection,
    ray_cylinder_intersection,
    ray_plane_intersection
)

class MjLidarJax:
    def __init__(self, mjx_model, geom_ids=None):
        self.model = mjx_model
        
        # If geom_ids is None, use all geoms
        if geom_ids is None:
            # We need to know ngeom. mjx.Model has ngeom.
            self.geom_ids = np.arange(mjx_model.ngeom)
        else:
            self.geom_ids = np.array(geom_ids)
            
        # Extract static properties
        # We convert to numpy to do the grouping
        # Note: This assumes mjx_model fields are accessible as arrays (they are)
        
        # We need to handle if mjx_model properties are on device.
        # We can cast to np.array.
        
        all_types = np.array(mjx_model.geom_type)
        
        # Filter by geom_ids
        self.selected_types = all_types[self.geom_ids]
        
        # Group indices by type
        # These are indices into the original model arrays
        self.sphere_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_SPHERE]
        self.box_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_BOX]
        self.capsule_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_CAPSULE]
        self.cylinder_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_CYLINDER]
        self.plane_ids = self.geom_ids[self.selected_types == mjtGeom.mjGEOM_PLANE]
        
        # Convert to jnp arrays for JIT
        self.sphere_ids = jnp.array(self.sphere_ids)
        self.box_ids = jnp.array(self.box_ids)
        self.capsule_ids = jnp.array(self.capsule_ids)
        self.cylinder_ids = jnp.array(self.cylinder_ids)
        self.plane_ids = jnp.array(self.plane_ids)
        
        # Store sizes (static)
        self.geom_sizes = mjx_model.geom_size
        
    @partial(jax.jit, static_argnums=(0,))
    def render(self, mjx_data, rays_origin, rays_direction):
        """
        Render LiDAR scan.
        
        Args:
            mjx_data: mjx.Data object (can be batched or single)
            rays_origin: (B, 3) or (3,) World position of sensor
            rays_direction: (B, N, 3) or (N, 3) World direction of rays
            
        Returns:
            distances: (B, N) or (N,)
        """
        # Determine if inputs are batched based on rays_origin rank
        # rays_origin: (3,) -> rank 1 (Single)
        # rays_origin: (B, 3) -> rank 2 (Batched)
        is_batched = rays_origin.ndim == 2
        
        if is_batched:
            # Check if rays_direction needs broadcasting
            # If rays_direction is (N, 3), we treat it as shared rays for all envs
            # If rays_direction is (B, N, 3), we map over it
            
            rd_ndim = rays_direction.ndim
            # (N, 3) -> 2, (B, N, 3) -> 3
            
            if rd_ndim == 2:
                # Shared rays: in_axes=(0, 0, None)
                return jax.vmap(self._render_single, in_axes=(0, 0, None))(mjx_data, rays_origin, rays_direction)
            else:
                # Unique rays: in_axes=(0, 0, 0)
                return jax.vmap(self._render_single, in_axes=(0, 0, 0))(mjx_data, rays_origin, rays_direction)
        else:
            return self._render_single(mjx_data, rays_origin, rays_direction)

    def _render_single(self, data, ro, rd):
        # data: Single mjx.Data
        # ro: (3,)
        # rd: (Nrays, 3)
        
        # We want to compute min_dist for each ray.
        # Initialize with inf
        min_dist = jnp.full(rd.shape[0], jnp.inf)
        
        # 1. Spheres
        if self.sphere_ids.shape[0] > 0:
            # pos: (N_spheres, 3)
            pos = data.geom_xpos[self.sphere_ids]
            # radius: (N_spheres,)
            rad = self.geom_sizes[self.sphere_ids, 0]
            
            # vmap over rays and spheres
            # We want (Nrays, Nspheres) distances
            
            # intersect_fn: (ro, rd, pos, rad) -> t
            # vmap over rd (0), pos (None), rad (None) -> (Nrays,)
            # vmap over pos (0), rad (0) -> (Nspheres,)
            
            # We want cross product of rays and spheres.
            # ray_sphere_intersection(ro, rd[i], pos[j], rad[j])
            
            def dist_all_rays_all_spheres(ro, rd, pos, rad):
                # rd: (Nrays, 3)
                # pos: (Nspheres, 3)
                # rad: (Nspheres,)
                
                # vmap over rays
                def dist_rays(p, r):
                    return jax.vmap(lambda d: ray_sphere_intersection(ro, d, p, r))(rd)
                
                # vmap over spheres
                dists = jax.vmap(dist_rays)(pos, rad) # (Nspheres, Nrays)
                
                return jnp.min(dists, axis=0) # (Nrays,)
            
            d_spheres = dist_all_rays_all_spheres(ro, rd, pos, rad)
            min_dist = jnp.minimum(min_dist, d_spheres)

        # 2. Boxes
        if self.box_ids.shape[0] > 0:
            pos = data.geom_xpos[self.box_ids]
            rot = data.geom_xmat[self.box_ids]
            size = self.geom_sizes[self.box_ids]
            
            def dist_all_rays_all_boxes(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_box_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
                
            d_boxes = dist_all_rays_all_boxes(ro, rd, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_boxes)

        # 3. Capsules
        if self.capsule_ids.shape[0] > 0:
            pos = data.geom_xpos[self.capsule_ids]
            rot = data.geom_xmat[self.capsule_ids]
            size = self.geom_sizes[self.capsule_ids]
            
            def dist_all_rays_all_capsules(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_capsule_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
                
            d_caps = dist_all_rays_all_capsules(ro, rd, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_caps)

        # 4. Cylinders
        if self.cylinder_ids.shape[0] > 0:
            pos = data.geom_xpos[self.cylinder_ids]
            rot = data.geom_xmat[self.cylinder_ids]
            size = self.geom_sizes[self.cylinder_ids]
            
            def dist_all_rays_all_cylinders(ro, rd, pos, rot, size):
                def dist_rays(p, R, s):
                    return jax.vmap(lambda d: ray_cylinder_intersection(ro, d, p, R, s))(rd)
                dists = jax.vmap(dist_rays)(pos, rot, size)
                return jnp.min(dists, axis=0)
                
            d_cyls = dist_all_rays_all_cylinders(ro, rd, pos, rot, size)
            min_dist = jnp.minimum(min_dist, d_cyls)

        # 5. Planes
        if self.plane_ids.shape[0] > 0:
            pos = data.geom_xpos[self.plane_ids]
            rot = data.geom_xmat[self.plane_ids]
            # Plane normal is local Z axis (0,0,1) rotated by rot
            # normal = rot @ [0,0,1] = rot[:, 2]
            normals = rot[:, :, 2]
            
            def dist_all_rays_all_planes(ro, rd, pos, normal):
                def dist_rays(p, n):
                    return jax.vmap(lambda d: ray_plane_intersection(ro, d, p, n))(rd)
                dists = jax.vmap(dist_rays)(pos, normals)
                return jnp.min(dists, axis=0)
                
            d_planes = dist_all_rays_all_planes(ro, rd, pos, normals)
            min_dist = jnp.minimum(min_dist, d_planes)
            
        return min_dist

