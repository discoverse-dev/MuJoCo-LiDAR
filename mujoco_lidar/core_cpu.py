import mujoco
import numpy as np

try:
    import taichi as ti
except ImportError:
    ti = None

from typing import Union

class LidarSensor:
    def __init__(self, 
                 mj_model:mujoco.MjModel, site_name:str, 
                 cutoff_dist:float=100.0, geomgroup:int=None, 
                 bodyexclude:int=-1, backend:str="cpu", obj_path:str=None) -> None:
        """A LiDAR sensor class that simulates LiDAR behavior in MuJoCo environment using ray casting."""
        self.mj_model = mj_model
        try:
            # Find site in model. If not found, raise error
            self.site_name = site_name
            self.site_id = self.mj_model.site(self.site_name).id
        except ValueError:
            raise ValueError(f"Site '{site_name}' not found in model")        
        self.cutoff_dist = cutoff_dist
        self.geomgroup = geomgroup
        self.bodyexclude = bodyexclude
        self._sensorpose = np.eye(4, dtype=np.float64)
        assert backend in ["cpu", "gpu"], "backend must be 'cpu' or 'gpu'"
        self.backend = backend
        if self.backend == "gpu":
            from lidar_scanner import StaticBVHLidar
            self.ti_lidar = StaticBVHLidar(obj_path=obj_path)

    def update_sensor_pose(self, mj_data:mujoco.MjData) -> None:
        self._sensorpose[:3,:3] = mj_data.site(self.site_name).xmat.reshape(3,3)
        self._sensorpose[:3,3] = mj_data.site(self.site_name).xpos

    @property
    def sensor_pose(self) -> np.ndarray:
        posi = self._sensorpose[:3,3].copy()
        rmat = self._sensorpose[:3,:3].copy()
        return posi, rmat

    @property
    def sensor_position(self) -> np.ndarray:
        return self._sensorpose[:3,3].copy()

    @property
    def sensor_rotation(self) -> np.ndarray:
        return self._sensorpose[:3,:3].copy()

    def update(self, 
               mj_data:mujoco.MjData, 
               ray_phi:Union[np.ndarray, ti.ndarray], 
               ray_theta:Union[np.ndarray, ti.ndarray]) -> None:

        self.update_sensor_pose(mj_data)

        if self.backend == "gpu":
            self.ti_lidar.trace_ti(self._sensorpose, ray_theta, ray_phi)
            self.pcl_world = self.ti_lidar.get_hit_points()

        elif self.backend == "cpu":
            if ray_phi.shape[0] != ray_theta.shape[0]:
                raise ValueError("ray_phi and ray_theta must have the same shape")

            _nray = ray_phi.shape[0]

            # Initialize
            _dist = np.full(_nray, self.cutoff_dist, dtype=np.float64)
            _geomid = np.full(_nray, 0, dtype=np.int32)

            # Uniformly generate vec from site's pose and lidar settings
            # Note that all the vec are in the local frame.
            site_pos, site_mat = self.sensor_pose
            pnt = np.array([site_pos]).T
            x = np.cos(ray_phi) * np.cos(ray_theta)
            y = np.cos(ray_phi) * np.sin(ray_theta)
            z = np.sin(ray_phi)
            local_vecs = np.stack((x, y, z), axis=-1)
            world_vecs = (site_mat @ local_vecs.T).T
            world_vecs /= np.linalg.norm(world_vecs, axis=1, keepdims=True)
            world_vecs_flat = world_vecs.flatten()

            # Get the ray casting results
            mujoco.mj_multiRay(
                m=self.mj_model,
                d=mj_data,
                pnt=pnt,
                vec=world_vecs_flat,
                geomgroup=self.geomgroup,
                flg_static=1,
                bodyexclude=self.bodyexclude,
                geomid=_geomid,
                dist=_dist,
                nray=_nray,
                cutoff=self.cutoff_dist,
            )
            # Calculate the point's position in local frame from vec + dist
            _dist[_geomid == -1] = 0

            # Update the pcl frame with local frame data
            self.pcl_frame = local_vecs * _dist[:, np.newaxis]

    def get_data_in_local_frame(self) -> np.ndarray:
        if not hasattr(self, 'pcl_frame'):
            raise ValueError("No point cloud data available. Please call update() first.")
        return self.pcl_frame

    def get_data_in_world_frame(self) -> np.ndarray:
        if self.backend == "cpu":
            pcl_local = self.get_data_in_local_frame()
            site_pos, site_mat = self.sensor_pose()
            pcl_world = pcl_local @ site_mat.T + site_pos
            return pcl_world
        elif self.backend == "gpu":
            if not hasattr(self, 'pcl_world'):
                raise ValueError("No point cloud data available. Please call update() first.")
            return self.pcl_world