import os
import numpy as np
import taichi as ti

from tibvh import AABB, LBVH
from tibvh.geometry import ray_triangle_distance

@ti.data_oriented
class StaticBVHLidar:
    """静态场景激光雷达扫描器"""
    def __init__(self, obj_path: str, max_candidates: int = 32):
        if not os.path.isfile(obj_path):
            raise FileNotFoundError(f"OBJ不存在: {obj_path}")
        self.obj_path = obj_path
        self.max_candidates = max_candidates
        self.vertices_np, self.faces_np = self._load_obj(obj_path)
        self.n_faces = self.faces_np.shape[0]
        v0 = self.vertices_np[self.faces_np[:, 0]]
        v1 = self.vertices_np[self.faces_np[:, 1]]
        v2 = self.vertices_np[self.faces_np[:, 2]]
        tri = np.stack([v0, v1, v2], axis=1).astype(np.float32)  # (n,3,3)
        aabb_mins_np = tri.min(axis=1).astype(np.float32)
        aabb_maxs_np = tri.max(axis=1).astype(np.float32)
        # SoA: 三个顶点字段
        self.tri_v0 = ti.Vector.field(3, dtype=ti.f32, shape=self.n_faces)
        self.tri_v1 = ti.Vector.field(3, dtype=ti.f32, shape=self.n_faces)
        self.tri_v2 = ti.Vector.field(3, dtype=ti.f32, shape=self.n_faces)
        self.aabb_mins = ti.Vector.field(3, dtype=ti.f32, shape=self.n_faces)
        self.aabb_maxs = ti.Vector.field(3, dtype=ti.f32, shape=self.n_faces)
        self.tri_v0.from_numpy(v0.astype(np.float32))
        self.tri_v1.from_numpy(v1.astype(np.float32))
        self.tri_v2.from_numpy(v2.astype(np.float32))
        self.aabb_mins.from_numpy(aabb_mins_np)
        self.aabb_maxs.from_numpy(aabb_maxs_np)
        self.aabb_manager = AABB(max_n_aabbs=self.n_faces)
        self._fill_aabb_manager()
        self.lbvh = LBVH(self.aabb_manager, max_candidates=self.max_candidates, profiling=False)
        self.lbvh.build()
        ti.sync()
        self._overflow = ti.field(dtype=ti.i32, shape=())
        self._hit_points = None
        self._distances = None
        self._theta_buf = None
        self._phi_buf = None
        self._static_theta = None  # 预置静态模式角度缓存
        self._static_phi = None
        self._use_static = False

    def _load_obj(self, path):
        verts = []
        faces = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    _, x, y, z = line.strip().split()[:4]
                    verts.append([float(x), float(y), float(z)])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    idx = []
                    for p in parts:
                        idx.append(int(p.split('/')[0]) - 1)
                    if len(idx) == 3:
                        faces.append(idx)
                    elif len(idx) == 4:
                        faces.append([idx[0], idx[1], idx[2]])
                        faces.append([idx[0], idx[2], idx[3]])
        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

    @ti.kernel
    def _fill_aabb_manager(self):
        for i in ti.ndrange(self.n_faces):
            self.aabb_manager.aabbs[i].min = self.aabb_mins[i]
            self.aabb_manager.aabbs[i].max = self.aabb_maxs[i]

    @ti.kernel
    def _reset_overflow(self):
        self._overflow[None] = 0

    def _ensure_capacity(self, n_rays: int):
        if self._hit_points is None or self._hit_points.shape[0] != n_rays:
            self._hit_points = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            self._distances = ti.field(dtype=ti.f32, shape=n_rays)
        if not self._use_static:
            if self._theta_buf is None or self._theta_buf.shape[0] != n_rays:
                self._theta_buf = ti.ndarray(dtype=ti.f32, shape=n_rays)
                self._phi_buf = ti.ndarray(dtype=ti.f32, shape=n_rays)

    def _decompose_pose(self, pose_4x4: np.ndarray):
        rot = pose_4x4[:3, :3].astype(np.float32)
        origin = pose_4x4[:3, 3].astype(np.float32)
        return rot, origin

    def trace_ti(self, pose_4x4: np.ndarray, theta_ti: ti.ndarray, phi_ti: ti.ndarray):
        # 直接使用已存在的 taichi ndarray，避免 numpy->device 拷贝
        if theta_ti.shape[0] != phi_ti.shape[0]:
            raise ValueError("theta/phi shape mismatch")
        n_rays = theta_ti.shape[0]
        self._ensure_capacity(n_rays)
        if pose_4x4.shape != (4, 4):
            raise ValueError("pose_4x4 必须为 (4,4)")
        rot_np, origin_np = self._decompose_pose(pose_4x4)
        rot_ti = ti.ndarray(dtype=ti.f32, shape=(3, 3))
        rot_ti.from_numpy(rot_np)
        origin_arr = ti.ndarray(dtype=ti.f32, shape=3)
        origin_arr.from_numpy(origin_np)
        self._reset_overflow()
        self._trace_kernel(rot_ti, origin_arr, theta_ti, phi_ti, n_rays, self._hit_points, self._distances)
        ti.sync()

    def get_hit_points(self):
        if self._hit_points is None:
            return None
        return self._hit_points.to_numpy()

    def get_hit_distances(self):
        if self._distances is None:
            return None
        return self._distances.to_numpy()

    @ti.kernel
    def _trace_kernel(self,
                      rot: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      origin_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                      theta_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                      phi_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
                      n_rays: ti.i32,
                      hit_pts: ti.template(),
                      distances: ti.template()
    ):
        o = ti.Vector([origin_arr[0], origin_arr[1], origin_arr[2]])
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
            candidates, candidates_count = self.lbvh.collect_intersecting_elements(o, ray_dir)
            if candidates_count >= self.max_candidates-1:
                ti.atomic_add(self._overflow[None], 1)
            best_t = 1e10
            for c in range(candidates_count):
                tri_id = candidates[c]
                v0 = self.tri_v0[tri_id]
                v1 = self.tri_v1[tri_id]
                v2 = self.tri_v2[tri_id]
                t_hit = ray_triangle_distance(o, ray_dir, v0, v1, v2)
                if t_hit >= 0 and t_hit < best_t:
                    best_t = t_hit
            if best_t < 1e9:
                distances[i] = best_t
                hit_pts[i] = ti.Vector([o.x + best_t * ray_dir.x,
                                        o.y + best_t * ray_dir.y,
                                        o.z + best_t * ray_dir.z])
            else:
                distances[i] = -1.0
                hit_pts[i] = ti.Vector([0.0, 0.0, 0.0])

    def get_stats(self):
        return {"overflow_count": int(self._overflow[None])}