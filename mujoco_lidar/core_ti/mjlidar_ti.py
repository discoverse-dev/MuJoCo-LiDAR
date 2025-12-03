import mujoco
import numpy as np
import taichi as ti

from tibvh import AABB, LBVH
from tibvh.geometry import (
    ray_triangle_distance,
    ray_plane_distance,
    ray_sphere_distance,
    ray_capsule_distance,
    ray_cylinder_distance,
    ray_ellipsoid_distance,
    ray_box_distance,
    aabb_local2wolrd
)

@ti.data_oriented
class MjLidarTi:
    def __init__(self, 
                 mj_model:mujoco.MjModel, cutoff_dist:float=100.0, 
                 geomgroup:np.ndarray=None, bodyexclude:int=-1, max_candidates: int = 32):
        self.max_candidates = max_candidates
        self._cutoff = min(cutoff_dist, 1e9)

        self.nface = 0
        self.ngeom = mj_model.ngeom
        model_nface = mj_model.mesh_face.shape[0]

        self.geom_xpos = np.zeros((self.ngeom, 3), dtype=np.float32)
        self.geom_xmat = np.zeros((self.ngeom, 9), dtype=np.float32)

        self.geom_positions = ti.Vector.field(3, dtype=ti.f32, shape=(self.ngeom))
        self.geom_rotations = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.ngeom))

        self.geom_types = ti.field(dtype=ti.i32, shape=(self.ngeom))
        self.geom_sizes = ti.Vector.field(3, dtype=ti.f32, shape=(self.ngeom))
        self.geom_aabb_center = ti.Vector.field(3, dtype=ti.f32, shape=(self.ngeom))
        self.geom_aabb_size = ti.Vector.field(3, dtype=ti.f32, shape=(self.ngeom))

        aabb_center = mj_model.geom_aabb[:,:3].astype(np.float32).copy()
        aabb_size = mj_model.geom_aabb[:,3:].astype(np.float32).copy()

        geom_types = mj_model.geom_type.astype(np.int32).copy()
        geom_sizes = mj_model.geom_size.astype(np.float32).copy()
        cylinder_capsule_args = np.where((geom_types == 3) | (geom_types == 5))[0]
        geom_sizes[cylinder_capsule_args, 2] = geom_sizes[cylinder_capsule_args, 1]
        geom_sizes[cylinder_capsule_args, 1] = geom_sizes[cylinder_capsule_args, 0]

        plane_args = np.where(mj_model.geom_type.astype(np.int32) == 0)[0]
        mesh_args = np.where(mj_model.geom_type.astype(np.int32) == 7)[0]
        aabb_center[plane_args, :] = 0
        aabb_size[plane_args, :] = mj_model.geom_size[plane_args, :]
        aabb_center[mesh_args, :] = 0.0
        aabb_size[mesh_args, :] = 1e-9

        exclude_geoms_args = np.where(mj_model.geom_bodyid == bodyexclude)[0]
        geom_types[exclude_geoms_args] = -1
        aabb_size[exclude_geoms_args, :] = 1e-9

        if geomgroup is not None:
            for i in range(mujoco.mjNGROUP):
                if not geomgroup[i]:
                    exclude_group_args = np.where(mj_model.geom_group == i)[0]
                    geom_types[exclude_group_args] = -1
                    aabb_size[exclude_group_args, :] = 1e-9

        self.geom_types.from_numpy(geom_types)
        self.geom_sizes.from_numpy(geom_sizes)
        self.geom_aabb_center.from_numpy(aabb_center)
        self.geom_aabb_size.from_numpy(aabb_size)

        if model_nface:
            mesh_id_np = np.array([], dtype=np.int32)
            v0, v1, v2 = np.array([]), np.array([]), np.array([])
            for i, data_id in enumerate(mj_model.geom_dataid):
                if data_id < 0 or data_id >= mj_model.nmesh or geom_types[i] == -1:
                    continue
                mesh_id_np = np.append(mesh_id_np, [i] * mj_model.mesh_facenum[data_id])
                self.nface += mj_model.mesh_facenum[data_id]
                new_v0 = mj_model.mesh_vert[mj_model.mesh_vertadr[data_id]+mj_model.mesh_face[mj_model.mesh_faceadr[data_id]:mj_model.mesh_faceadr[data_id]+mj_model.mesh_facenum[data_id], 0]].copy().astype(np.float32)
                new_v1 = mj_model.mesh_vert[mj_model.mesh_vertadr[data_id]+mj_model.mesh_face[mj_model.mesh_faceadr[data_id]:mj_model.mesh_faceadr[data_id]+mj_model.mesh_facenum[data_id], 1]].copy().astype(np.float32)
                new_v2 = mj_model.mesh_vert[mj_model.mesh_vertadr[data_id]+mj_model.mesh_face[mj_model.mesh_faceadr[data_id]:mj_model.mesh_faceadr[data_id]+mj_model.mesh_facenum[data_id], 2]].copy().astype(np.float32)
                v0 = np.append(v0, new_v0, axis=0) if v0.size else new_v0
                v1 = np.append(v1, new_v1, axis=0) if v1.size else new_v1
                v2 = np.append(v2, new_v2, axis=0) if v2.size else new_v2
            if self.nface:
                self.mesh_id = ti.field(dtype=ti.i32, shape=self.nface)
                self.mesh_id.from_numpy(mesh_id_np.astype(np.int32))

                self.tri_static_v0 = ti.Vector.field(3, dtype=ti.f32, shape=self.nface)
                self.tri_static_v1 = ti.Vector.field(3, dtype=ti.f32, shape=self.nface)
                self.tri_static_v2 = ti.Vector.field(3, dtype=ti.f32, shape=self.nface)
                self.tri_static_v0.from_numpy(v0.astype(np.float32))
                self.tri_static_v1.from_numpy(v1.astype(np.float32))
                self.tri_static_v2.from_numpy(v2.astype(np.float32))

                tri = np.stack([v0, v1, v2], axis=1).astype(np.float32)  # (n,3,3)
                mesh_aabb_mins = ti.Vector.field(3, dtype=ti.f32, shape=self.nface)
                mesh_aabb_maxs = ti.Vector.field(3, dtype=ti.f32, shape=self.nface)
                mesh_aabb_mins.from_numpy(tri.min(axis=1).astype(np.float32))
                mesh_aabb_maxs.from_numpy(tri.max(axis=1).astype(np.float32))

        self.tri_v0 = ti.Vector.field(3, dtype=ti.f32, shape=max(self.nface, 1))
        self.tri_v1 = ti.Vector.field(3, dtype=ti.f32, shape=max(self.nface, 1))
        self.tri_v2 = ti.Vector.field(3, dtype=ti.f32, shape=max(self.nface, 1))

        # build scene manager
        self.scene_aabb_manager = AABB(max_n_aabbs=self.ngeom+self.nface)
        self.scene_lbvh = LBVH(self.scene_aabb_manager, max_candidates=self.max_candidates, profiling=False)

        self._overflow = ti.field(dtype=ti.i32, shape=())
        self._hit_points = None
        self._distances = None

    def update(self, mj_data:mujoco.MjData):
        if np.allclose(self.geom_xpos, mj_data.geom_xpos, atol=1e-3) and np.allclose(self.geom_xmat, mj_data.geom_xmat, atol=1e-3):
            return
        self.geom_xpos = mj_data.geom_xpos.astype(np.float32)
        self.geom_xmat = mj_data.geom_xmat.astype(np.float32)
        self._update_geom_pose(self.geom_xpos, self.geom_xmat)
        self._update_geom_aabb()
        if self.nface:
            self._update_face_pose()
            self._update_face_aabb()
        ti.sync()
        self.scene_lbvh.build()

    def _update_geom_pose(self, geom_position, geom_xmat):
        self.geom_positions.from_numpy(geom_position.astype(np.float32))
        self.geom_rotations.from_numpy(geom_xmat.reshape((self.ngeom, 3, 3)).astype(np.float32))
    
    @ti.kernel
    def _update_geom_aabb(self):
        for i in ti.ndrange(self.ngeom):
            aabb_min, aabb_max = aabb_local2wolrd(self.geom_aabb_center[i], self.geom_aabb_size[i], self.geom_positions[i], self.geom_rotations[i])
            self.scene_aabb_manager.aabbs[i].min = aabb_min
            self.scene_aabb_manager.aabbs[i].max = aabb_max

    @ti.kernel
    def _update_face_pose(self):
        for i in ti.ndrange(self.nface):
            rot = self.geom_rotations[self.mesh_id[i]]
            arr = self.geom_positions[self.mesh_id[i]]
            self.tri_v0[i] = rot @ self.tri_static_v0[i] + arr
            self.tri_v1[i] = rot @ self.tri_static_v1[i] + arr
            self.tri_v2[i] = rot @ self.tri_static_v2[i] + arr

    @ti.kernel
    def _update_face_aabb(self):
        for i in ti.ndrange(self.nface):
            aabb_min = ti.Vector([
                ti.min(self.tri_v0[i][0], self.tri_v1[i][0], self.tri_v2[i][0]),
                ti.min(self.tri_v0[i][1], self.tri_v1[i][1], self.tri_v2[i][1]),
                ti.min(self.tri_v0[i][2], self.tri_v1[i][2], self.tri_v2[i][2]),
            ])
            aabb_max = ti.Vector([
                ti.max(self.tri_v0[i][0], self.tri_v1[i][0], self.tri_v2[i][0]),
                ti.max(self.tri_v0[i][1], self.tri_v1[i][1], self.tri_v2[i][1]),
                ti.max(self.tri_v0[i][2], self.tri_v1[i][2], self.tri_v2[i][2]),
            ])
            self.scene_aabb_manager.aabbs[self.ngeom + i].min = aabb_min
            self.scene_aabb_manager.aabbs[self.ngeom + i].max = aabb_max

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

    def get_hit_points(self):
        return self._hit_points.to_numpy()

    def get_distances(self):
        return self._distances.to_numpy()

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
                t_hit = -1.0
                if candidates[c] < self.ngeom:
                    geom_id = candidates[c]
                    geom_type = self.geom_types[geom_id]
                    geom_center = self.geom_positions[geom_id]
                    geom_size = self.geom_sizes[geom_id]
                    geom_rot = self.geom_rotations[geom_id]

                    if geom_type == -1:  # exclude
                        pass
                    elif geom_type == 0:  # PLANE
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
                        pass
                elif self.nface:
                    tri_id = candidates[c] - self.ngeom
                    v0 = self.tri_v0[tri_id]
                    v1 = self.tri_v1[tri_id]
                    v2 = self.tri_v2[tri_id]
                    t_hit = ray_triangle_distance(o, ray_dir, v0, v1, v2)

                if 0 <= t_hit and t_hit < best_t:
                    best_t = t_hit

            if best_t < self._cutoff:
                distances[i] = best_t
                hit_pts[i] = ti.Vector([best_t * dir_local.x,
                                        best_t * dir_local.y,
                                        best_t * dir_local.z])
            else:
                distances[i] = 0.0
                hit_pts[i] = ti.Vector([0.0, 0.0, 0.0])