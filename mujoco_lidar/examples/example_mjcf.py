import os
import time
import threading

import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from mujoco_lidar.scan_gen import generate_grid_scan_pattern


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 从文件加载MuJoCo模型
mj_model = mujoco.MjModel.from_xml_path("../../models/demo.xml")    
mj_data = mujoco.MjData(mj_model)

# 生成网格扫描模式
rays_theta, rays_phi = generate_grid_scan_pattern(num_ray_cols=64, num_ray_rows=16)

# 创建激光雷达传感器
lidar_sim = MjLidarWrapper(mj_model, mj_data, site_name="lidar_site")
points = lidar_sim.get_lidar_points(rays_phi, rays_theta, mj_data)

lidar_sim_rate = 10
lidar_sim_cnt = 0

# print help
print("在Mujoco Viewer视图，双击选中MoCap物体（带坐标系的红色透明方块 lidar_site）")
print("选中后，按住ctrl，按下鼠标右键拖动平移视角")
print("按住ctrl，按下鼠标左键拖动旋转视角")
# 主循环
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    # 设置视图模式为site
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value
    viewer.cam.distance = 5.

    def plot_points_thread():
        global points, lidar_sim_rate
        plt.ion()  # 开启交互模式
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 0.3])  # 设置三个轴的比例尺相同

        while viewer.is_running:
            ax.cla()  # 清除当前坐标轴
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=3)
            plt.draw()  # 更新绘图
            plt.pause(1./lidar_sim_rate)  # 暂停以更新图形

    plot_points_thread = threading.Thread(target=plot_points_thread)
    plot_points_thread.start()

    while viewer.is_running:
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        time.sleep(1./60.)

        if mj_data.time * lidar_sim_rate > lidar_sim_cnt:

            # 更新激光雷达位置
            lidar_sim.update_scene(mj_model, mj_data)

            # 执行光线追踪
            points = lidar_sim.get_lidar_points(rays_phi, rays_theta, mj_data)
            if lidar_sim_cnt == 0:
                print("points basic info:")
                print("  .shape:", points.shape)
                print("  .dtype:", points.dtype)
                print("  x.min():", points[:, 0].min(), "x.max():", points[:, 0].max())
                print("  y.min():", points[:, 1].min(), "y.max():", points[:, 1].max())
                print("  z.min():", points[:, 2].min(), "z.max():", points[:, 2].max())

            lidar_sim_cnt += 1

    plot_points_thread.join()
