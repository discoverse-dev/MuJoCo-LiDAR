import time
import numpy as np

import mujoco
import mujoco.viewer
from mujoco_lidar import MjLidarWrapper, scan_gen

if __name__ == "__main__":
    mj_model = mujoco.MjModel.from_xml_path("../models/demo.xml")    
    mj_data = mujoco.MjData(mj_model)

    update_rate = 12.0  # Hz
    n_substeps = int(round(1.0 / (mj_model.opt.timestep * update_rate)))
    print(f"n_substeps = {n_substeps}")

    lidar = MjLidarWrapper(mj_model, "lidar_site", args={"bodyexclude":mj_model.body("your_robot_name").id})
    livox_generator = scan_gen.LivoxGenerator("mid360")
    rays_theta, rays_phi = livox_generator.sample_ray_angles()

    with mujoco.viewer.launch_passive(
        mj_model, 
        mj_data
    ) as viewer:
        viewer.user_scn.ngeom = rays_theta.shape[0]
        for i in range(viewer.user_scn.ngeom):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.8])
            )

        print("Starting simulation...")
        print("Number of rays:", rays_theta.shape[0])

        _last_time = 1e6
        while viewer.is_running():
            mujoco.mj_step(mj_model, mj_data)

            if mj_data.time < _last_time:
                _counter = 0
                _start_time = time.time()
            _last_time = mj_data.time

            _counter += 1
            if _counter % n_substeps == 0:
                rays_theta, rays_phi = livox_generator.sample_ray_angles()
                lidar.trace_rays(mj_data, rays_theta, rays_phi)
                points = lidar.get_hit_points()
                world_points = points @ lidar.sensor_rotation.T + lidar.sensor_position
                for i in range(viewer.user_scn.ngeom):
                    viewer.user_scn.geoms[i].pos[:] = world_points[i]

            viewer.sync()
            run_time = time.time() - _start_time
            if run_time < mj_data.time:
                time.sleep(mj_data.time - run_time)
