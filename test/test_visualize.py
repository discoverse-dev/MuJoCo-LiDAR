#!/usr/bin/env python3
"""
测试脚本：使用matplotlib 3D可视化MuJoCo模型中的mesh三角面片
"""

import mujoco
import matplotlib.pyplot as plt
from visualize_mesh import visualize_mesh_triangles, print_mesh_details, get_mesh_triangles_for_taichi

# 定义测试用的MuJoCo XML
xml = """
<mujoco>
  <default>
    <default class="collision">
        <geom group="0" type="mesh"/>
    </default>
  </default>

  <asset>
    <mesh name="eight" file="../models/eight.obj"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

    <!-- 平面 -->
    <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
    
    <!-- mesh几何体 -->
    <geom mesh="eight" pos="0 0 1" euler="0 0 0" rgba="0 1 0 1" class="collision"/>

  </worldbody>
</mujoco>
"""

def main():
    """主函数"""
    print("=" * 60)
    print("MuJoCo Mesh 三角面片可视化测试")
    print("=" * 60)
    
    try:
        # 创建MuJoCo模型和数据
        print("1. 加载MuJoCo模型...")
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_data = mujoco.MjData(mj_model)
        
        # 创建场景
        print("2. 创建和更新场景...")
        mj_scene = mujoco.MjvScene(mj_model, maxgeom=100)
        mujoco.mj_forward(mj_model, mj_data)
        mujoco.mjv_updateScene(
            mj_model, mj_data, mujoco.MjvOption(), 
            None, mujoco.MjvCamera(), 
            mujoco.mjtCatBit.mjCAT_ALL.value, mj_scene
        )
        
        print(f"模型加载成功！包含 {mj_model.nmesh} 个mesh和 {mj_scene.ngeom} 个几何体")
        
        # 显示场景中的几何体信息
        print("\n3. 场景几何体信息:")
        for i in range(mj_scene.ngeom):
            geom = mj_scene.geoms[i]
            geom_type_names = {
                0: "PLANE", 1: "HFIELD", 2: "SPHERE", 3: "CAPSULE", 
                4: "ELLIPSOID", 5: "CYLINDER", 6: "BOX", 7: "MESH"
            }
            type_name = geom_type_names.get(geom.type, f"UNKNOWN({geom.type})")
            print(f"  几何体{i}: type={type_name}, objtype={geom.objtype}, dataid={geom.dataid}")
            print(f"    位置: {geom.pos}")
        
        if mj_model.nmesh > 0:
            print("\n4. 打印第一个mesh的详细信息:")
            print_mesh_details(mj_model, 0)
            
            print("\n5. 开始可视化mesh三角面片...")
            fig, ax, triangles = visualize_mesh_triangles(mj_model, mj_scene)
            
            print("\n6. 获取适用于Taichi的三角形数据:")
            triangles_for_taichi = get_mesh_triangles_for_taichi(mj_model, mj_scene, 0)
            print(f"获取了 {len(triangles_for_taichi)} 个三角形")
            if triangles_for_taichi:
                print(f"第一个三角形的形状: {triangles_for_taichi[0].shape}")
                print(f"第一个三角形的顶点坐标:")
                for i, vertex in enumerate(triangles_for_taichi[0]):
                    print(f"  顶点{i}: [{vertex[0]:.3f}, {vertex[1]:.3f}, {vertex[2]:.3f}]")
                
                print(f"\n这些数据可以直接用于ray_triangle_intersection函数！")
                print(f"每个三角形都是3x3的numpy数组，每行代表一个顶点的(x,y,z)坐标")
        else:
            print("模型中没有mesh几何体")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 