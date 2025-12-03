import os
from etils import epath
import mujoco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mujoco_lidar import MjLidarWrapper, scan_gen

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 从文件加载MuJoCo模型
    mjcf_file = epath.Path(__file__).parent.parent / "models" / "demo.xml"
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # 生成网格扫描模式
    rays_theta, rays_phi = scan_gen.generate_grid_scan_pattern(num_ray_cols=64, num_ray_rows=16)

    # 获取需要排除的body id
    exclode_body_id = mj_model.body("your_robot_name").id
    print("exclude body id:", exclode_body_id)

    # 创建激光雷达传感器，使用GPU后端
    lidar_sensor = MjLidarWrapper(mj_model, site_name="lidar_site", backend="taichi", args={"bodyexclude": exclode_body_id})
    
    # 执行一次ray casting
    lidar_sensor.trace_rays(mj_data, rays_theta, rays_phi)
    points = lidar_sensor.get_hit_points()

    # 打印点云基本信息
    print("\nPoints basic info:")
    print("  .shape:", points.shape)
    print("  .dtype:", points.dtype)
    print("  x.min():", points[:, 0].min(), "x.max():", points[:, 0].max())
    print("  y.min():", points[:, 1].min(), "y.max():", points[:, 1].max())
    print("  z.min():", points[:, 2].min(), "z.max():", points[:, 2].max())

    # 使用matplotlib可视化点云
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 0.3])  # 设置三个轴的比例
    
    # 绘制点云，使用z坐标作为颜色映射
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=points[:, 2], cmap='viridis', s=3)
    
    ax.set_title('LiDAR Point Cloud (Single Ray Casting - GPU Backend)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax, label='Z coordinate')
    
    plt.show()
    print("\nVisualization completed.")

if __name__ == "__main__":
    main()
