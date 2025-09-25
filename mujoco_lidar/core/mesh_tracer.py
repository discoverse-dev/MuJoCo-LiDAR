import numpy as np
import taichi as ti

from tibvh import AABB, LBVH
from tibvh.geometry import ray_triangle_distance

@ti.data_oriented
class MeshTracer:
    def __init__(self, vertices_np, faces_np, max_candidates: int = 32):
        self.max_candidates = max_candidates
        
        n_faces = faces_np.shape[0]

        v0 = vertices_np[self.faces_np[:, 0]].astype(np.float32)
        v1 = vertices_np[self.faces_np[:, 1]].astype(np.float32)
        v2 = vertices_np[self.faces_np[:, 2]].astype(np.float32)
        self.tri_v0 = ti.Vector.field(3, dtype=ti.f32, shape=n_faces)
        self.tri_v1 = ti.Vector.field(3, dtype=ti.f32, shape=n_faces)
        self.tri_v2 = ti.Vector.field(3, dtype=ti.f32, shape=n_faces)
        self.tri_v0.from_numpy(v0.astype(np.float32))
        self.tri_v1.from_numpy(v1.astype(np.float32))
        self.tri_v2.from_numpy(v2.astype(np.float32))

        tri = np.stack([v0, v1, v2], axis=1).astype(np.float32)  # (n,3,3)
        aabb_mins = ti.Vector.field(3, dtype=ti.f32, shape=n_faces)
        aabb_maxs = ti.Vector.field(3, dtype=ti.f32, shape=n_faces)
        aabb_mins.from_numpy(tri.min(axis=1).astype(np.float32))
        aabb_maxs.from_numpy(tri.max(axis=1).astype(np.float32))

        self.aabb_manager = AABB(max_n_aabbs=n_faces)
        @ti.kernel
        def _fill_aabb_manager():
            for i in ti.ndrange(n_faces):
                self.aabb_manager.aabbs[i].min = aabb_mins[i]
                self.aabb_manager.aabbs[i].max = aabb_maxs[i]
        _fill_aabb_manager()

        self.lbvh = LBVH(self.aabb_manager, max_candidates=self.max_candidates, profiling=False)
        self.lbvh.build()
        ti.sync()

        self._overflow = ti.field(dtype=ti.i32, shape=())

    @ti.func
    def _reset_overflow(self):
        self._overflow[None] = 0

    @ti.func
    def trace(self,
              origin: ti.types.ndarray(dtype=ti.f32, ndim=1),
              ray_dir: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        self._reset_overflow()
        distances = -1.0
        candidates, candidates_count = self.lbvh.collect_intersecting_elements(origin, ray_dir)
        if candidates_count >= self.max_candidates-1:
            ti.atomic_add(self._overflow[None], 1)
        best_t = 1e10
        for c in range(candidates_count):
            tri_id = candidates[c]
            v0 = self.tri_v0[tri_id]
            v1 = self.tri_v1[tri_id]
            v2 = self.tri_v2[tri_id]
            t_hit = ray_triangle_distance(origin, ray_dir, v0, v1, v2)
            if t_hit >= 0 and t_hit < best_t:
                best_t = t_hit
        if best_t < 1e9:
            distances = best_t
        return distances
