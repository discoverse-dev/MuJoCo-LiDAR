"""
原生Taichi AABB实现

基于标准Taichi库的轴对齐包围盒(AABB)数据结构，
"""

import taichi as ti

@ti.data_oriented
class AABB:
    """
    单场景AABB管理器，管理单个场景中的所有AABB
    
    Attributes:
        max_n_aabbs (int): 最大AABB数量
        aabbs (AABB.field): AABB存储字段
    """
    
    def __init__(self, max_n_aabbs: int):
        """
        初始化AABB管理器
        
        Args:
            max_n_aabbs: 支持的最大AABB数量
        """
        self.max_n_aabbs = max_n_aabbs
        
        # 定义AABB数据结构
        @ti.dataclass
        class ti_aabb:
            min: ti.types.vector(3, ti.f32)  # 最小坐标
            max: ti.types.vector(3, ti.f32)  # 最大坐标
            
            @ti.func
            def intersects(self, other) -> bool:
                """
                检查两个AABB是否相交
                
                Args:
                    other: 另一个AABB
                    
                Returns:
                    bool: 是否相交
                """
                return (self.min[0] <= other.max[0] and self.max[0] >= other.min[0] and
                        self.min[1] <= other.max[1] and self.max[1] >= other.min[1] and
                        self.min[2] <= other.max[2] and self.max[2] >= other.min[2])
            
            @ti.func
            def center(self) -> ti.types.vector(3, ti.f32):
                """计算AABB中心点"""
                return (self.min + self.max) * 0.5
            
            @ti.func
            def size(self) -> ti.types.vector(3, ti.f32):
                """计算AABB尺寸"""
                return self.max - self.min

        # 分配AABB存储空间，使用SOA布局优化内存访问
        self.aabbs = ti_aabb.field(shape=max_n_aabbs, layout=ti.Layout.SOA)
    
    def get_count(self) -> int:
        """获取当前AABB数量"""
        return self.max_n_aabbs
    