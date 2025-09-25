import mujoco
import numpy as np
import taichi as ti
from tibvh import AABB, LBVH
from tibvh.geometry import (
    ray_plane_distance,
    ray_sphere_distance,
    ray_capsule_distance,
    ray_cylinder_distance,
    ray_ellipsoid_distance,
    ray_box_distance,
    aabb_local2wolrd
)

@ti.data_oriented
class MjLidarWrapper:
    def __init__(self, mj_model:mujoco.MjModel, site_name:str, max_candidates: int = 32):
        self.max_candidates = max_candidates

        self.ngeom = mj_model.ngeom
        self.geom_positions = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_rotations = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n_geoms))

        self.geom_types = ti.field(dtype=ti.i32, shape=(self.n_geoms))
        self.geom_sizes = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_data_ids = ti.field(dtype=ti.i32, shape=(self.n_geoms))
        self.geom_aabb_center = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_aabb_size = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))

        self.geom_types.from_numpy(mj_model.geom_type.astype(np.int32))
        self.geom_sizes.from_numpy(mj_model.geom_size.astype(np.float32))
        self.geom_data_ids.from_numpy(mj_model.geom_dataid.astype(np.int32))
        self.geom_aabb_center.from_numpy(mj_model.geom_aabb[:,:3])
        self.geom_aabb_size.from_numpy(mj_model.geom_aabb[:,3:])

        # build mesh
        self.face_addr = ti.field(dtype=ti.i32, shape=mj_model.mesh_faceadr.shape)
        self.face_num = ti.field(dtype=ti.i32, shape=mj_model.mesh_facenum.shape)
        self.mesh_face = ti.field(dtype=ti.i32, shape=mj_model.mesh_face.shape)
        self.vert_addr = ti.field(dtype=ti.i32, shape=mj_model.mesh_vertadr.shape)
        self.vert_num = ti.field(dtype=ti.i32, shape=mj_model.mesh_vertnum.shape)
        self.mesh_vert = ti.field(dtype=ti.f32, shape=mj_model.mesh_vert.shape)

        self.face_addr.from_numpy(mj_model.mesh_faceadr.astype(np.int32))
        self.face_num.from_numpy(mj_model.mesh_facenum.astype(np.int32))
        self.mesh_face.from_numpy(mj_model.mesh_face.astype(np.int32))
        self.vert_addr.from_numpy(mj_model.mesh_vertadr.astype(np.int32))
        self.vert_num.from_numpy(mj_model.mesh_vertnum.astype(np.int32))
        self.mesh_vert.from_numpy(mj_model.mesh_vert.astype(np.float32))

        # build scene manager
        self.scene_aabb_manager = AABB(max_n_aabbs=self.ngeom)
        self.scene_lbvh = LBVH(self.scene_aabb_manager, max_candidates=self.max_candidates, profiling=False)

        self._overflow = ti.field(dtype=ti.i32, shape=())
        self._hit_points = None
        self._distances = None

    def update(self, mj_data:mujoco.MjData):
        self._update_geom_pose(mj_data.geom_xpos, mj_data.geom_xmat)
        self._update_aabb()
        ti.sync()
        self.scene_lbvh.build()

    def _update_geom_pose(self, geom_position, geom_xmat):
        self.geom_positions.from_numpy(geom_position)
        self.geom_rotations.from_numpy(geom_xmat.reshape((self.ngeom, 3, 3)))
    
    @ti.kernal
    def _update_aabb(self):
        for i in ti.ndrange(self.ngeom):
            aabb_min, aabb_max = aabb_local2wolrd(self.geom_aabb_center[i], self.geom_aabb_size[i], self.geom_positions[i], self.geom_rotations[i])
            self.scene_aabb_manager.aabbs[i].min = aabb_min
            self.scene_aabb_manager.aabbs[i].max = aabb_max

    def trace_rays(self, pose_4x4: np.ndarray, theta_ti: ti.ndarray, phi_ti: ti.ndarray):
        if theta_ti.shape[0] != phi_ti.shape[0]:
            raise ValueError("theta/phi shape mismatch")
        n_rays = theta_ti.shape[0]
        self._ensure_capacity(n_rays)
        rot_np = pose_4x4[:3, :3].astype(np.float32)
        origin_np = pose_4x4[:3, 3].astype(np.float32)

        rot_ti = ti.ndarray(dtype=ti.f32, shape=(3, 3))
        rot_ti.from_numpy(rot_np)
        origin_ti = ti.ndarray(dtype=ti.f32, shape=3)
        origin_ti.from_numpy(origin_np)        
        self._reset_overflow()
        self._trace_kernel(rot_ti, origin_ti, theta_ti, phi_ti, n_rays, self._hit_points, self._distances)
        ti.sync()

    def _ensure_capacity(self, n_rays: int):
        if self._hit_points is None or self._hit_points.shape[0] != n_rays:
            self._hit_points = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            self._distances = ti.field(dtype=ti.f32, shape=n_rays)

    @ti.kernel
    def _reset_overflow(self):
        self._overflow[None] = 0

    @ti.kernel
    def _trace_kernel(self,
                      rot: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      origin: ti.types.ndarray(dtype=ti.f32, ndim=1),
                      theta_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                      phi_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                      n_rays: ti.i32,
                      hit_pts: ti.template(),
                      distances: ti.template()
    ):
        o = ti.Vector([origin[0], origin[1], origin[2]])
        for i in ti.ndrange(n_rays):
            t_angle = theta_arr[i]
            p_angle = phi_arr[i]
            cos_t = ti.cos(t_angle)
            sin_t = ti.sin(t_angle)
            cos_p = ti.cos(p_angle)
            sin_p = ti.sin(p_angle)
            dir_local = ti.Vector([cos_p * cos_t, cos_p * sin_t, sin_p])
            ray_dir = ti.Vector([
                rot[0, 0] * dir_local.x + rot[0, 1] * dir_local.y + rot[0, 2] * dir_local.z,
                rot[1, 0] * dir_local.x + rot[1, 1] * dir_local.y + rot[1, 2] * dir_local.z,
                rot[2, 0] * dir_local.x + rot[2, 1] * dir_local.y + rot[2, 2] * dir_local.z
            ])
            candidates, candidates_count = self.scene_lbvh.collect_intersecting_elements(o, ray_dir)
            if candidates_count >= self.max_candidates-1:
                ti.atomic_add(self._overflow[None], 1)

            best_t = 1e10
            for c in range(candidates_count):
                geom_id = candidates[c]
                geom_type = self.geom_types[geom_id]
                geom_center = self.geom_positions[geom_id]
                geom_size = self.geom_sizes[geom_id]
                geom_rot = self.geom_rotations[geom_id]

                if geom_type == 0:  # PLANE
                    t_hit = ray_plane_distance(o, ray_dir, geom_center, geom_size, geom_rot)
                elif geom_type == 2:  # SPHERE
                    t_hit = ray_sphere_distance(o, ray_dir, geom_center, geom_size, geom_rot)
                elif geom_type == 3:  # CAPSULE
                    t_hit = ray_capsule_distance(o, ray_dir, geom_center, geom_size, geom_rot)
                elif geom_type == 4:  # ELLIPSOID
                    t_hit = ray_ellipsoid_distance(o, ray_dir, geom_center, geom_size, geom_rot)
                elif geom_type == 5:  # CYLINDER
                    t_hit = ray_cylinder_distance(o, ray_dir, geom_center, geom_size, geom_rot)
                elif geom_type == 6:  # BOX
                    t_hit = ray_box_distance(o, ray_dir, geom_center, geom_size, geom_rot)
                elif geom_type == 7:  # MESH
                    # TODO
                    # t_hit = self.ray_mesh_intersection(o, ray_dir, data_id, center, rotation)
                    pass

                if 0 <= t_hit and t_hit < best_t:
                    best_t = t_hit
            if best_t < 1e9:
                distances[i] = best_t
                hit_pts[i] = ti.Vector([o.x + best_t * ray_dir.x,
                                        o.y + best_t * ray_dir.y,
                                        o.z + best_t * ray_dir.z])
            else:
                distances[i] = -1.0
                hit_pts[i] = ti.Vector([0.0, 0.0, 0.0])