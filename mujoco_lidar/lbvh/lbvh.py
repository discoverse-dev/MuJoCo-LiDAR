"""
原生Taichi Linear BVH实现

基于Karras算法的完全并行化Linear BVH实现

参考文献:
Karras, T. (2012). "Maximizing Parallelism in the Construction of BVHs, 
Octrees, and k-d Trees". NVIDIA Research.
"""
import time

import torch
import numpy as np

import taichi as ti
from .aabb import AABB

ti_bool = ti.u1

@ti.data_oriented
class LBVH:
    """
    原生Taichi Linear BVH实现
    
    使用莫顿码排序和并行树构建的高性能BVH，
    专为单场景优化，支持高效的空间查询。
    
    Attributes:
        aabb_manager (SingleSceneAABB): AABB管理器
        self.n_aabbs (int): AABB数量
        max_query_results (int): 最大查询结果数量
        max_stack_depth (int): 最大栈深度
    """
    
    def __init__(self, aabb_manager: AABB, max_query_results_per_aabb: int = 8, profiling: bool = False):
        """
        初始化Linear BVH
        
        Args:
            aabb_manager: AABB管理器
            max_query_results_per_aabb: 每个AABB的最大查询结果数量
            profiling: 是否启用性能统计
        """
        self.aabb_manager = aabb_manager
        self.max_aabbs = aabb_manager.max_n_aabbs
        self.n_aabbs = 0
        
        # 性能统计相关
        self.profiling = profiling
        self.timing_stats = {} if profiling else None
        
        # 查询相关配置
        self.max_query_results = min(self.max_aabbs * max_query_results_per_aabb, 0x7FFFFFFF)
        self.max_stack_depth = 64
        
        # AABB中心点和场景边界
        self.aabb_centers = ti.Vector.field(3, ti.f32, shape=self.max_aabbs)
        self.scene_min = ti.Vector.field(3, ti.f32, shape=())
        self.scene_max = ti.Vector.field(3, ti.f32, shape=())
        self.scene_scale = ti.Vector.field(3, ti.f32, shape=())
        
        # 莫顿码相关字段
        self.morton_codes = ti.Vector.field(2, ti.u32, shape=self.max_aabbs)  # [morton_code, original_index]

        # BVH节点定义
        @ti.dataclass
        class BVHNode:
            left: ti.i32        # 左子节点索引
            right: ti.i32       # 右子节点索引  
            parent: ti.i32      # 父节点索引
            aabb_min: ti.types.vector(3, ti.f32)  # 节点AABB最小坐标
            aabb_max: ti.types.vector(3, ti.f32)  # 节点AABB最大坐标
            element_id: ti.i32  # 叶子节点关联的元素ID (内部节点为-1)
        
        # BVH树结构：前self.n_aabbs-1个是内部节点，后self.n_aabbs个是叶子节点
        self.nodes = BVHNode.field(shape=self.max_aabbs * 2 - 1, layout=ti.Layout.SOA)
        
        # 树构建状态标记
        self.internal_node_active = ti.field(ti_bool, shape=self.max_aabbs)
        self.internal_node_ready = ti.field(ti_bool, shape=self.max_aabbs)
        
        # 查询结果存储
        self.query_result = ti.Vector.field(3, ti.i32, shape=self.max_query_results)  # [batch_id(0), elem_a, elem_b]
        self.query_result_count = ti.field(ti.i32, shape=())
       
        # 初始化
        self.reset()
    
    def _start_timing(self, step_name: str):
        """开始计时某个步骤"""
        if self.profiling:
            return time.time()
        return None
    
    def _end_timing(self, step_name: str, start_time):
        """结束计时并记录"""
        if self.profiling and start_time is not None:
            elapsed = time.time() - start_time
            if step_name not in self.timing_stats:
                self.timing_stats[step_name] = []
            self.timing_stats[step_name].append(elapsed)
    
    def get_timing_stats(self):
        """获取性能统计信息"""
        if not self.profiling or not self.timing_stats:
            return {}
        
        stats = {}
        for step, times in self.timing_stats.items():
            stats[step] = {
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'count': len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        return stats
    
    def print_timing_stats(self):
        """打印性能统计信息"""
        if not self.profiling:
            print("性能统计未启用")
            return
        
        stats = self.get_timing_stats()
        if not stats:
            print("暂无性能统计数据")
            return
        
        print("\n=== BVH构建性能统计 ===")
        
        # 过滤掉非时间统计项
        time_stats = {k: v for k, v in stats.items() if not k.endswith('_count')}
        total_time = sum(stat['total_time'] for stat in time_stats.values())
        
        for step, stat in time_stats.items():
            percentage = (stat['total_time'] / total_time * 100) if total_time > 0 else 0
            print(f"{step}:")
            print(f"  总时间: {stat['total_time']*1000:.2f}ms ({percentage:.1f}%)")
            print(f"  平均时间: {stat['avg_time']*1000:.2f}ms")
            print(f"  调用次数: {stat['count']}")
            print(f"  最小/最大时间: {stat['min_time']*1000:.2f}ms / {stat['max_time']*1000:.2f}ms")
        
        print(f"\n总耗时: {total_time*1000:.2f}ms")
        
        # 输出其他统计信息
        other_stats = {k: v for k, v in stats.items() if k.endswith('_count')}
        if other_stats:
            print("\n=== 其他统计信息 ===")
            for stat_name, stat in other_stats.items():
                if stat_name == 'bounds_layer_count':
                    avg_layers = stat['avg_time']  # 这里实际存储的是层数不是时间
                    print(f"边界计算层数: {int(avg_layers)} 层")
    
    def reset(self):
        """重置BVH状态"""
        self.query_result_count[None] = 0
        if self.profiling:
            self.timing_stats = {}
    
    def build(self):
        """
        构建BVH树
        
        完整的BVH构建流程：
        1. 计算AABB中心点和场景边界
        2. 计算莫顿码
        3. 基数排序
        4. 构建基数树
        5. 计算节点边界
        """
        # 更新当前AABB数量
        self.n_aabbs = self.aabb_manager.get_count()
       
        # 构建流程（带性能统计）
        start_time = self._start_timing("1_compute_centers_bounds")
        self.compute_aabb_centers_and_scene_bounds()
        self._end_timing("1_compute_centers_bounds", start_time)
        
        start_time = self._start_timing("2_compute_morton_codes")
        self.compute_morton_codes()
        self._end_timing("2_compute_morton_codes", start_time)

        start_time = self._start_timing("2_compute_morton_codes")
        value, idx = self.morton_codes.to_torch(device = 'cuda')[:,0].to(torch.int64).sort()
        morton_code = torch.concatenate([value.unsqueeze(1), idx.unsqueeze(1)], dim=1).to(torch.uint32)
        self.morton_codes.from_torch(morton_code)
        self._end_timing("3_sort", start_time)

        start_time = self._start_timing("4_build_tree")
        self.build_radix_tree()
        self._end_timing("4_build_tree", start_time)
        
        start_time = self._start_timing("5_compute_bounds")
        self.compute_bounds()
        self._end_timing("5_compute_bounds", start_time)
    
    def compute_aabb_centers_and_scene_bounds(self):
        """计算AABB中心点和场景边界"""
        # 更新当前AABB数量
        self.n_aabbs = self.aabb_manager.get_count()
        
        # 调用kernel进行计算
        self._kernel_compute_aabb_centers_and_scene_bounds()
    
    @ti.kernel
    def _kernel_compute_aabb_centers_and_scene_bounds(self):
        """kernel: 计算AABB中心点和场景边界"""
        # 计算所有AABB中心点
        for i in range(self.n_aabbs):
            aabb = self.aabb_manager.aabbs[ti.i32(i)]
            self.aabb_centers[ti.i32(i)] = (aabb.min + aabb.max) / 2
        
        # 初始化场景边界
        self.scene_min[None] = self.aabb_manager.aabbs[0].min
        self.scene_max[None] = self.aabb_manager.aabbs[0].max
        
        # 计算场景边界
        for i in range(self.n_aabbs):
            ti.atomic_min(self.scene_min[None], self.aabb_manager.aabbs[i].min)
            ti.atomic_max(self.scene_max[None], self.aabb_manager.aabbs[i].max)
        
        # 计算缩放因子
        scene_size = self.scene_max[None] - self.scene_min[None]
        EPS = 1e-6
        self.scene_scale[None] = ti.Vector([
            1.0 / scene_size[0] if scene_size[0] > EPS else 1.0,
            1.0 / scene_size[1] if scene_size[1] > EPS else 1.0,
            1.0 / scene_size[2] if scene_size[2] > EPS else 1.0
        ])
    
    @ti.func
    def expand_bits(self, v: ti.u32) -> ti.u32:
        """将10位整数扩展为30位（每位间插入2个0）"""
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)  
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v
    
    def compute_morton_codes(self):
        """计算莫顿码"""
        # 确保AABB数量已更新
        if self.n_aabbs == 0:
            self.n_aabbs = self.aabb_manager.get_count()
        
        # 调用kernel进行计算
        self._kernel_compute_morton_codes()
    
    @ti.kernel
    def _kernel_compute_morton_codes(self):
        """kernel: 计算莫顿码"""
        for i in range(self.n_aabbs):
            # 归一化中心点坐标
            center = self.aabb_centers[i] - self.scene_min[None]
            scaled_center = center * self.scene_scale[None]
            
            # 转换为10位整数坐标
            x = ti.u32(ti.max(0, ti.min(1023, scaled_center[0] * 1023.0)))
            y = ti.u32(ti.max(0, ti.min(1023, scaled_center[1] * 1023.0)))
            z = ti.u32(ti.max(0, ti.min(1023, scaled_center[2] * 1023.0)))
            
            # 扩展并交错位
            morton_x = self.expand_bits(x)
            morton_y = self.expand_bits(y)
            morton_z = self.expand_bits(z)
            
            # 组合为最终莫顿码
            morton_code = (morton_x << 2) | (morton_y << 1) | (morton_z)
            
            # 存储莫顿码和原始索引
            self.morton_codes[i] = ti.Vector([morton_code, i])
    
    @ti.kernel
    def build_radix_tree(self):
        """构建基数树结构 - 修复父子关系竞争条件"""
        # 首先初始化所有节点
        for i in range(self.n_aabbs * 2 - 1):
            self.nodes[i].parent = 0  # 临时设为0，后面会正确设置
            self.nodes[i].left = -1
            self.nodes[i].right = -1
            self.nodes[i].element_id = -1
        
        # 设置根节点
        self.nodes[0].parent = -1
        
        # 初始化叶子节点
        for i in range(self.n_aabbs):
            leaf_idx = i + self.n_aabbs - 1
            self.nodes[leaf_idx].element_id = ti.i32(self.morton_codes[i][1])
        
        # 构建内部节点 - 避免并行竞争
        for i in range(self.n_aabbs - 1):
            # 确定分裂方向
            d = ti.select(self.delta(i, i + 1) > self.delta(i, i - 1), 1, -1)
            
            # 找到范围的另一端
            delta_min = self.delta(i, i - d)
            l_max = ti.u32(2)
            while self.delta(i, i + ti.i32(l_max) * d) > delta_min:
                l_max *= 2
            l = ti.u32(0)
            
            # 二分查找确定范围的精确边界
            t = l_max // 2
            while t > 0:
                if self.delta(i, i + ti.i32(l + t) * d) > delta_min:
                    l += t
                t //= 2
            j = i + ti.i32(l) * d
            
            # 找到分裂点
            delta_node = self.delta(i, j)
            s = ti.u32(0)
            t = (l + 1) // 2
            while t > 0:
                if self.delta(i, i + ti.i32(s + t) * d) > delta_node:
                    s += t
                t = ti.select(t > 1, (t + 1) // 2, 0)
            
            gamma = i + ti.i32(s) * d + ti.min(d, 0)
            # 计算左右子节点索引 - 修正的Karras算法


            left = gamma  # 内部节点
            right = gamma + 1  # 内部节点

            if ti.min(i, j) == gamma:
                left = gamma + self.n_aabbs - 1  # 叶子节点
            else:
                left = gamma  # 内部节点
                
            if ti.max(i, j) == gamma + 1:
                right = gamma + 1 + self.n_aabbs - 1  # 叶子节点
            else:
                right = gamma + 1  # 内部节点
            
            # 设置当前节点的子节点
            self.nodes[i].left = ti.i32(left)
            self.nodes[i].right = ti.i32(right)
        # 单独设置父子关系，避免竞争条件
        for i in range(self.n_aabbs - 1):
            left_child = self.nodes[i].left
            right_child = self.nodes[i].right
            if left_child != -1:
                self.nodes[left_child].parent = i
            if right_child != -1:
                self.nodes[right_child].parent = i
            #debug test

    @ti.func
    def delta(self, i: ti.i32, j: ti.i32) -> ti.i32:
        """计算两个莫顿码的最长公共前缀"""
        result = -1
        if j >= 0 and j < self.n_aabbs:
            result = 64
            for i_bit in range(2):
                x = self.morton_codes[i][i_bit] ^ self.morton_codes[j][i_bit]
                for b in range(32):
                    if x & (ti.u32(1) << (31 - b)):
                        result = b + 32 * i_bit
                        break
                if result != 64:
                    break
        return result

    def compute_bounds(self):
        """自底向上计算节点边界"""
        init_start = self._start_timing("5a_bounds_init")
        self._kernel_compute_bounds_init()
        self._end_timing("5a_bounds_init", init_start)
        
        layer_start = self._start_timing("5b_bounds_layers")
        is_done = False
        layer_count = 0
        while not is_done:
            is_done = (self._kernel_compute_bounds_one_layer() == 1)
            layer_count += 1
            if layer_count > 100:  # 防止死循环
                break
        self._end_timing("5b_bounds_layers", layer_start)
        
        if self.profiling:
            if "bounds_layer_count" not in self.timing_stats:
                self.timing_stats["bounds_layer_count"] = []
            self.timing_stats["bounds_layer_count"].append(layer_count)
    
    @ti.kernel
    def _kernel_compute_bounds_init(self):
        """初始化叶子节点边界"""
        self.internal_node_active.fill(False)
        self.internal_node_ready.fill(False)
        for i in ti.ndrange(self.n_aabbs):
            idx = ti.i32(self.morton_codes[i][1])
            self.nodes[i + self.n_aabbs - 1].aabb_min = self.aabb_manager.aabbs[idx].min
            self.nodes[i + self.n_aabbs - 1].aabb_max = self.aabb_manager.aabbs[idx].max
            parent_idx = self.nodes[i + self.n_aabbs - 1].parent
            if parent_idx != -1:
                self.internal_node_active[parent_idx] = True
                self.internal_node_active[parent_idx] = True

    @ti.kernel  
    def _kernel_compute_bounds_one_layer(self) -> ti.i32:
        """计算一层节点边界"""
        # 批量处理当前活跃的内部节点
        for i in ti.ndrange(self.n_aabbs - 1):
            if self.internal_node_active[i]:
                left_child = self.nodes[i].left
                right_child = self.nodes[i].right
                
                # 从子节点获取边界
                left_min = self.nodes[left_child].aabb_min
                left_max = self.nodes[left_child].aabb_max
                right_min = self.nodes[right_child].aabb_min
                right_max = self.nodes[right_child].aabb_max
                
                # 更新当前节点边界
                self.nodes[i].aabb_min = ti.min(left_min, right_min)
                self.nodes[i].aabb_max = ti.max(left_max, right_max)
                
                # 标记父节点为下一层处理对象
                parent_idx = self.nodes[i].parent
                if parent_idx != -1:
                    self.internal_node_ready[parent_idx] = True
                
                # 当前节点处理完成，标记为非活跃
                self.internal_node_active[i] = False

        # 检查是否有更多层需要处理
        has_more_layers = 0
        for i in range(self.n_aabbs - 1):
            if self.internal_node_ready[i]:
                self.internal_node_active[i] = True
                self.internal_node_ready[i] = False
                has_more_layers = 1

        return has_more_layers == 0
    
    @ti.func
    def ray_bvh_traversal(self, ray_start: ti.types.vector(3, ti.f32),
                         ray_direction: ti.types.vector(3, ti.f32)) -> ti.i32:
        """
        射线-BVH遍历查找最近交点
        
        Args:
            ray_start: 射线起点
            ray_direction: 射线方向
            
        Returns:
            hit_element_id: 最近交点的元素ID，-1表示未命中
        """
        # 初始化结果
        hit_element_id = -1

        if self.n_aabbs > 0:
            # 使用栈进行深度优先遍历
            stack = ti.Vector.zero(ti.i32, self.max_stack_depth)
            stack_depth = 1
            stack[ti.i32(0)] = 0  # 从根节点开始
            
            min_distance = 1e10
            
            while stack_depth > 0:
                stack_depth -= 1
                node_idx = stack[ti.i32(stack_depth)]
                
                # 射线-AABB相交测试
                t = self.ray_node_intersect(ray_start, ray_direction, node_idx)
                
                if t >= 0 and t < min_distance:
                    # 检查是否为叶子节点
                    if self.nodes[node_idx].left == -1 and self.nodes[node_idx].right == -1:
                        # 叶子节点：返回element_id用于后续精确相交测试
                        element_id = self.nodes[node_idx].element_id
                        if element_id >= 0:
                            min_distance = t
                            hit_element_id = element_id
                    else:
                        # 内部节点：压栈子节点
                        if self.nodes[node_idx].right != -1 and stack_depth < self.max_stack_depth:
                            stack[ti.i32(stack_depth)] = self.nodes[node_idx].right
                            stack_depth += 1
                        if self.nodes[node_idx].left != -1 and stack_depth < self.max_stack_depth:
                            stack[ti.i32(stack_depth)] = self.nodes[node_idx].left
                            stack_depth += 1

        return hit_element_id
    
    @ti.func 
    def collect_intersecting_elements(self, 
                                    ray_start: ti.types.vector(3, ti.f32),
                                    ray_direction: ti.types.vector(3, ti.f32)
                                    ) -> ti.types.vector(32, ti.i32):
        """
        收集与射线相交的所有叶子节点元素ID
        
        Args:
            ray_start: 射线起点
            ray_direction: 射线方向
           
        Returns:
            (leaf_indices, leaf_distances, leaf_count): 候选节点ID数组和数量
        """
        MAX_CANDIDATES = 32

        candidates = ti.Vector([-1]*32, ti.i32)  # 叶节点索引数组 MAX_CANDIDATES
        candidate_count = 0
        
        if self.n_aabbs > 0:
            # 使用栈进行深度优先遍历
            stack = ti.Vector.zero(ti.i32, self.max_stack_depth)
            stack_depth = 1
            stack[0] = 0  # 从根节点开始
            
            while stack_depth > 0 and candidate_count < MAX_CANDIDATES:
                stack_depth -= 1
                node_idx = stack[stack_depth]
                
                # 射线-AABB相交测试
                t = self.ray_node_intersect(ray_start, ray_direction, node_idx)
                
                if t >= 0:  # 与AABB相交
                    # 检查是否为叶子节点
                    if self.nodes[node_idx].left == -1 and self.nodes[node_idx].right == -1:
                        # 叶子节点：收集element_id
                        element_id = self.nodes[node_idx].element_id
                        if element_id >= 0:
                            candidates[ti.i32(candidate_count)] = element_id
                            candidate_count += 1
                    else:
                        # 内部节点：压栈子节点
                        if self.nodes[node_idx].right != -1 and stack_depth < self.max_stack_depth:
                            stack[stack_depth] = self.nodes[node_idx].right
                            stack_depth += 1
                        if self.nodes[node_idx].left != -1 and stack_depth < self.max_stack_depth:
                            stack[stack_depth] = self.nodes[node_idx].left
                            stack_depth += 1
        
        return candidates, candidate_count
    
    @ti.func
    def ray_node_intersect(self, ray_start: ti.types.vector(3, ti.f32),
                          ray_direction: ti.types.vector(3, ti.f32),
                          node_idx: ti.i32) -> ti.f32:
        """射线与BVH节点的AABB相交测试"""
        node = self.nodes[node_idx]
        
        # slab方法
        inv_dir = ti.types.vector(3, ti.f32)([
            1.0 / ray_direction[0] if ti.abs(ray_direction[0]) > 1e-8 else 1e10,
            1.0 / ray_direction[1] if ti.abs(ray_direction[1]) > 1e-8 else 1e10,
            1.0 / ray_direction[2] if ti.abs(ray_direction[2]) > 1e-8 else 1e10
        ])
        
        t_min = (node.aabb_min - ray_start) * inv_dir
        t_max = (node.aabb_max - ray_start) * inv_dir
        
        t1 = ti.min(t_min, t_max)
        t2 = ti.max(t_min, t_max)
        
        t_near = ti.max(ti.max(t1[0], t1[1]), t1[2])
        t_far = ti.min(ti.min(t2[0], t2[1]), t2[2])
        
        result = -1.0
        if t_near <= t_far and t_far >= 0:
            if t_near >= 0:
                result = t_near
            else:
                result = t_far
        return result
    
    def validate_tree_structure(self):
        """
        验证BVH树结构的正确性
        
        Returns:
            dict: 验证结果，包含各种统计信息和错误报告
        """
        if self.n_aabbs == 0:
            return {"status": "empty", "message": "树为空"}
        
        # 同步数据到CPU
        ti.sync()
        
        # 获取节点数据
        parents_data = self.nodes.parent.to_numpy()
        left_data = self.nodes.left.to_numpy()
        right_data = self.nodes.right.to_numpy()
        aabb_min_data = self.nodes.aabb_min.to_numpy()
        aabb_max_data = self.nodes.aabb_max.to_numpy()
        element_id_data = self.nodes.element_id.to_numpy()
        
        total_nodes = self.n_aabbs * 2 - 1
        internal_nodes = self.n_aabbs - 1
        leaf_nodes = self.n_aabbs
        
        result = {
            "status": "valid",
            "n_aabbs": self.n_aabbs,
            "total_nodes": total_nodes,
            "internal_nodes": internal_nodes,
            "leaf_nodes": leaf_nodes,
            "errors": [],
            "warnings": []
        }
        
        # 1. 检查根节点
        root_nodes = []
        for i in range(total_nodes):
            if parents_data[i] == -1:
                root_nodes.append(i)
        
        if len(root_nodes) == 0:
            result["errors"].append("未找到根节点")
            result["status"] = "invalid"
        elif len(root_nodes) > 1:
            result["errors"].append(f"找到多个根节点: {root_nodes}")
            result["status"] = "invalid"
        else:
            root_idx = root_nodes[0]
            result["root_node"] = {
                "index": root_idx,
                "aabb_min": aabb_min_data[root_idx].tolist(),
                "aabb_max": aabb_max_data[root_idx].tolist(),
                "left_child": int(left_data[root_idx]),
                "right_child": int(right_data[root_idx]),
                "element_id": int(element_id_data[root_idx])
            }
            
            # 检查根节点AABB是否有效
            root_min = aabb_min_data[root_idx]
            root_max = aabb_max_data[root_idx]
            if np.any(root_min == 0) and np.any(root_max == 0):
                result["errors"].append("根节点AABB为零值")
            elif np.any(root_min > root_max):
                result["errors"].append("根节点AABB最小值大于最大值")
        
        # 2. 检查内部节点结构
        internal_node_issues = 0
        for i in range(internal_nodes):
            left_child = left_data[i]
            right_child = right_data[i]
            
            # 内部节点必须有两个子节点
            if left_child == -1 or right_child == -1:
                result["errors"].append(f"内部节点 {i} 缺少子节点: left={left_child}, right={right_child}")
                internal_node_issues += 1
            
            # 内部节点不应有element_id
            if element_id_data[i] != -1:
                result["warnings"].append(f"内部节点 {i} 有非-1的element_id: {element_id_data[i]}")
            
            # 检查子节点的父指针
            if left_child != -1 and left_child < total_nodes:
                if parents_data[left_child] != i:
                    result["errors"].append(f"节点 {i} 的左子节点 {left_child} 的父指针错误: {parents_data[left_child]} != {i}")
            
            if right_child != -1 and right_child < total_nodes:
                if parents_data[right_child] != i:
                    result["errors"].append(f"节点 {i} 的右子节点 {right_child} 的父指针错误: {parents_data[right_child]} != {i}")
        result["internal_nodes_valid"] = internal_nodes - internal_node_issues
        result["internal_nodes_invalid"] = internal_node_issues
        
        # 3. 检查叶子节点结构
        leaf_node_issues = 0
        valid_element_ids = set()
        for i in range(leaf_nodes):
            leaf_idx = i + internal_nodes
            left_child = left_data[leaf_idx]
            right_child = right_data[leaf_idx]
            element_id = element_id_data[leaf_idx]
            
            # 叶子节点不应有子节点
            if left_child != -1 or right_child != -1:
                result["errors"].append(f"叶子节点 {leaf_idx} 有子节点: left={left_child}, right={right_child}")
                leaf_node_issues += 1
            
            # 叶子节点必须有有效的element_id
            if element_id < 0 or element_id >= self.n_aabbs:
                result["errors"].append(f"叶子节点 {leaf_idx} 的element_id无效: {element_id}")
                leaf_node_issues += 1
            else:
                if element_id in valid_element_ids:
                    result["errors"].append(f"element_id {element_id} 被多个叶子节点引用")
                valid_element_ids.add(element_id)
        
        result["leaf_nodes_valid"] = leaf_nodes - leaf_node_issues
        result["leaf_nodes_invalid"] = leaf_node_issues
        result["unique_element_ids"] = len(valid_element_ids)
        
        # 4. 检查父子关系的一致性
        parent_child_errors = 0
        for i in range(total_nodes):
            parent_idx = parents_data[i]
            if parent_idx != -1:
                if parent_idx >= total_nodes:
                    result["errors"].append(f"节点 {i} 的父节点索引超出范围: {parent_idx}")
                    parent_child_errors += 1
                else:
                    # 检查父节点是否将当前节点作为子节点
                    left_child = left_data[parent_idx]
                    right_child = right_data[parent_idx]
                    if left_child != i and right_child != i:
                        result["errors"].append(f"节点 {i} 的父节点 {parent_idx} 未将其作为子节点")
                        parent_child_errors += 1
        
        result["parent_child_errors"] = parent_child_errors
        
        # 5. 检查AABB边界的有效性
        invalid_aabbs = 0
        for i in range(total_nodes):
            aabb_min = aabb_min_data[i]
            aabb_max = aabb_max_data[i]
            
            if np.any(aabb_min > aabb_max):
                result["warnings"].append(f"节点 {i} AABB最小值大于最大值")
                invalid_aabbs += 1
            
            if np.any(np.isnan(aabb_min)) or np.any(np.isnan(aabb_max)):
                result["errors"].append(f"节点 {i} AABB包含NaN值")
                invalid_aabbs += 1
        
        result["invalid_aabbs"] = invalid_aabbs
        
        # 6. 统计完成度
        if len(root_nodes) == 1:
            completed_internal_nodes = internal_nodes - internal_node_issues
            completion_rate = completed_internal_nodes / internal_nodes if internal_nodes > 0 else 1.0
            result["completion_rate"] = completion_rate
            result["completion_percentage"] = completion_rate * 100
        
        # 设置总体状态
        if len(result["errors"]) > 0:
            result["status"] = "invalid"
        elif len(result["warnings"]) > 0:
            result["status"] = "warning"
        
        return result
    
    def print_validation_results(self, validation_result=None):
        """
        打印验证结果
        
        Args:
            validation_result: 验证结果字典，如果为None则重新验证
        """
        if validation_result is None:
            validation_result = self.validate_tree_structure()
        
        result = validation_result
        print("\n=== BVH树结构验证结果 ===")
        print(f"状态: {result['status']}")
        print(f"AABB数量: {result['n_aabbs']}")
        print(f"总节点数: {result['total_nodes']}")
        print(f"内部节点数: {result['internal_nodes']}")
        print(f"叶子节点数: {result['leaf_nodes']}")
        
        if "root_node" in result:
            root = result["root_node"]
            print(f"\n根节点 (索引 {root['index']}):")
            print(f"  AABB: {root['aabb_min']} -> {root['aabb_max']}")
            print(f"  左子节点: {root['left_child']}")
            print(f"  右子节点: {root['right_child']}")
            print(f"  Element ID: {root['element_id']}")
        
        print(f"\n节点状态:")
        print(f"  有效内部节点: {result.get('internal_nodes_valid', 0)}/{result['internal_nodes']}")
        print(f"  有效叶子节点: {result.get('leaf_nodes_valid', 0)}/{result['leaf_nodes']}")
        print(f"  唯一element_id数量: {result.get('unique_element_ids', 0)}")
        
        if "completion_percentage" in result:
            print(f"  树构建完成度: {result['completion_percentage']:.1f}%")
        
        print(f"\n错误统计:")
        print(f"  父子关系错误: {result.get('parent_child_errors', 0)}")
        print(f"  无效AABB: {result.get('invalid_aabbs', 0)}")
        
        if result["errors"]:
            print(f"\n错误 ({len(result['errors'])}):")
            for i, error in enumerate(result["errors"][:10]):  # 只显示前10个错误
                print(f"  {i+1}. {error}")
            if len(result["errors"]) > 10:
                print(f"  ... 还有 {len(result['errors']) - 10} 个错误")
        
        if result["warnings"]:
            print(f"\n警告 ({len(result['warnings'])}):")
            for i, warning in enumerate(result["warnings"][:5]):  # 只显示前5个警告
                print(f"  {i+1}. {warning}")
            if len(result["warnings"]) > 5:
                print(f"  ... 还有 {len(result['warnings']) - 5} 个警告")
    