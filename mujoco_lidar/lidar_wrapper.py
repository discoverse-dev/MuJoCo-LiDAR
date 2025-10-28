import mujoco
import numpy as np

class MjLidarWrapper:
    """
    MuJoCo LiDAR wrapper that supports both CPU and GPU backends.
    
    Args:
        mj_model (mujoco.MjModel): MuJoCo model object
        site_name (str): Name of the LiDAR site in the MuJoCo model
        backend (str): Computation backend, either 'cpu' or 'gpu'. Default: 'gpu'
        cutoff_dist (float): Maximum ray tracing distance in meters. Default: 100.0
        args (dict): Additional backend-specific arguments. Default: {}
        
            CPU Backend Arguments:
                geomgroup (int | None): Geometry group filter (0-5, or None for all). Default: None
                    - None: Detect all geometries
                    - 0-5: Only detect geometries in the specified group
                bodyexclude (int): Body ID to exclude from detection. Default: -1
                    - -1: Don't exclude any body
                    - >= 0: Exclude all geometries of the specified body
                
            GPU Backend Arguments:
                max_candidates (int): Maximum number of BVH candidate nodes. Default: 32
                    - Larger values: More accurate but slower
                    - Smaller values: Faster but may miss collisions
                    - Recommended: 16-32 (simple), 32-64 (medium), 64-128 (complex)
                ti_init_args (dict): Arguments passed to taichi.init(). Default: {}
                    - device_memory_GB (float): GPU memory limit in GB
                    - debug (bool): Enable debug mode
                    - log_level (str): 'trace', 'debug', 'info', 'warn', 'error'
    
    Examples:
        >>> # CPU backend with body exclusion
        >>> lidar = MjLidarWrapper(
        ...     mj_model=model,
        ...     site_name="lidar_site",
        ...     backend="cpu",
        ...     cutoff_dist=50.0,
        ...     args={'bodyexclude': robot_body_id}
        ... )
        
        >>> # GPU backend for complex scenes
        >>> lidar = MjLidarWrapper(
        ...     mj_model=model,
        ...     site_name="lidar_site",
        ...     backend="gpu",
        ...     cutoff_dist=100.0,
        ...     args={
        ...         'max_candidates': 64,
        ...         'ti_init_args': {'device_memory_GB': 4.0}
        ...     }
        ... )
    
    See Also:
        ARGS_DOCUMENTATION.md for detailed parameter descriptions and usage examples
    """
    
    def __init__(self, mj_model, site_name:str,
                 backend:str="gpu", cutoff_dist:float=100.0, args:dict={}):
        self.backend = backend
        self.mj_model = mj_model
        self.cutoff_dist = cutoff_dist
        self.args = args
        
        # Lazy import backend modules based on the selected backend
        if backend == "gpu":
            self._init_gpu_backend()
        elif backend == "cpu":
            self._init_cpu_backend()
        else:
            raise ValueError(f"Unsupported backend: {backend}, choose from 'cpu' or 'gpu'")

        self.site_name = site_name
        self._sensor_pose = np.eye(4, dtype=np.float32)

    def _init_gpu_backend(self):
        """Initialize GPU backend with Taichi"""
        try:
            # Lazy import: only import when GPU backend is selected
            from mujoco_lidar.core_ti.mjlidar_ti import MjLidarTi
            import taichi as ti
            
            # Initialize Taichi if not already done
            if not hasattr(ti, '_is_initialized') or not ti._is_initialized:
                ti.init(arch=ti.gpu, **self.args.get('ti_init_args', {}))
            
            # Create GPU backend instance
            max_candidates = self.args.get('max_candidates', 32)
            self._backend_instance = MjLidarTi(
                self.mj_model, 
                cutoff_dist=self.cutoff_dist,
                max_candidates=max_candidates
            )
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import GPU backend dependencies. "
                f"Please install taichi: pip install taichi\n"
                f"Error: {e}"
            )
    
    def _init_cpu_backend(self):
        """Initialize CPU backend without Taichi dependencies"""
        try:
            # Lazy import: only import when CPU backend is selected
            from mujoco_lidar.core_cpu.mjlidar_cpu import MjLidarCPU
            
            # Create CPU backend instance
            geomgroup = self.args.get('geomgroup', None)
            bodyexclude = self.args.get('bodyexclude', -1)
            self._backend_instance = MjLidarCPU(
                self.mj_model,
                cutoff_dist=self.cutoff_dist,
                geomgroup=geomgroup,
                bodyexclude=bodyexclude
            )
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import CPU backend dependencies.\n"
                f"Error: {e}"
            )

    @property
    def sensor_position(self):
        return self._sensor_pose[:3,3].copy()

    @property
    def sensor_rotation(self):
        return self._sensor_pose[:3,:3].copy()

    def update_sensor_pose(self, mj_data:mujoco.MjData, site_name:str):
        self._sensor_pose[:3,:3] = mj_data.site(site_name).xmat.reshape(3,3)
        self._sensor_pose[:3,3] = mj_data.site(site_name).xpos

    def trace_rays(self, mj_data: mujoco.MjData, ray_theta: np.ndarray, ray_phi: np.ndarray, site_name:str=None):
        """
        Trace rays from a given pose with specified angles.
        
        Args:
            pose_4x4: 4x4 transformation matrix (position + rotation)
            ray_theta: Horizontal angles (azimuth)
            ray_phi: Vertical angles (elevation)
        """

        if site_name is None:
            self.update_sensor_pose(mj_data, self.site_name)
        else:
            self.update_sensor_pose(mj_data, site_name)

        self._backend_instance.update(mj_data)
        if self.backend == "gpu":
            # GPU backend expects Taichi ndarrays
            import taichi as ti
            
            # Convert numpy arrays to Taichi ndarrays if necessary
            if isinstance(ray_theta, np.ndarray):
                theta_ti = ti.ndarray(dtype=ti.f32, shape=ray_theta.shape[0])
                theta_ti.from_numpy(ray_theta.astype(np.float32))
            else:
                theta_ti = ray_theta
                
            if isinstance(ray_phi, np.ndarray):
                phi_ti = ti.ndarray(dtype=ti.f32, shape=ray_phi.shape[0])
                phi_ti.from_numpy(ray_phi.astype(np.float32))
            else:
                phi_ti = ray_phi
            
            self._backend_instance.trace_rays(self._sensor_pose, theta_ti, phi_ti)
        else:
            # CPU backend uses numpy arrays directly
            self._backend_instance.trace_rays(self._sensor_pose, ray_theta, ray_phi)
    
    def get_hit_points(self) -> np.ndarray:
        """Get hit points from the last ray tracing"""
        return self._backend_instance.get_hit_points()
    
    def get_distances(self) -> np.ndarray:
        """Get distances from the last ray tracing"""
        return self._backend_instance.get_distances()
