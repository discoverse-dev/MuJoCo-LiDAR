import mujoco
import numpy as np

class MjLidarWrapper:
    """
    MuJoCo LiDAR wrapper that supports CPU, Taichi, and JAX backends.
    
    Args:
        mj_model (mujoco.MjModel): MuJoCo model object
        site_name (str): Name of the LiDAR site in the MuJoCo model
        backend (str): Computation backend, 'cpu', 'taichi', or 'jax'. Default: 'taichi'
        cutoff_dist (float): Maximum ray tracing distance in meters. Default: 100.0
        args (dict): Additional backend-specific arguments. Default: {}
        
            CPU Backend Arguments:
                geomgroup (np.ndarray | None): Geometry group filter (0-5, or None for all). Default: None
                    - None: Detect all geometries
                    - geomgroup is an array of length mjNGROUP, where 1 means the group should be included. Pass geomgroup=None to skip group exclusion.
                bodyexclude (int): Body ID to exclude from detection. Default: -1
                    - -1: Don't exclude any body
                    - >= 0: Exclude all geometries of the specified body
                
            Taichi Backend Arguments:
                max_candidates (int): Maximum number of BVH candidate nodes. Default: 32
                    - Larger values: More accurate but slower
                    - Smaller values: Faster but may miss collisions
                    - Recommended: 16-32 (simple), 32-64 (medium), 64-128 (complex)
                ti_init_args (dict): Arguments passed to taichi.init(). Default: {}
                    - device_memory_GB (float): GPU memory limit in GB
                    - debug (bool): Enable debug mode
                    - log_level (str): 'trace', 'debug', 'info', 'warn', 'error'

            JAX Backend Arguments:
                geom_ids (list | None): List of geometry IDs to include. Default: None (all)    

    Examples:
        >>> # CPU backend with body exclusion
        >>> lidar = MjLidarWrapper(
        ...     mj_model=model,
        ...     site_name="lidar_site",
        ...     backend="cpu",
        ...     cutoff_dist=50.0,
        ...     args={
        ...         'bodyexclude': robot_body_id,
        ...         'geomgroup': np.array([1, 1, 1, 0, 0, 0], np.dtype=np.uint8)
        ...     }   
        ... )
        
        >>> # GPU backend for complex scenes
        >>> lidar = MjLidarWrapper(
        ...     mj_model=model,
        ...     site_name="lidar_site",
        ...     backend="gpu",
        ...     cutoff_dist=100.0,
        ...     args={
        ...         'bodyexclude': robot_body_id,
        ...         'geomgroup': np.array([1, 1, 1, 0, 0, 0], np.dtype=np.uint8),
        ...         'max_candidates': 64,
        ...         'ti_init_args': {'device_memory_GB': 4.0}
        ...     }
        ... )
    """
    
    def __init__(self, mj_model, site_name:str,
                 backend:str="taichi", cutoff_dist:float=100.0, args:dict={}):
        if backend == "gpu":
            backend = "taichi"
        self.backend = backend
        self.mj_model = mj_model
        self.cutoff_dist = cutoff_dist
        self.args = args
        
        if backend == "taichi":
            self._init_taichi_backend()
        elif backend == "jax":
            self._init_jax_backend()
        elif backend == "cpu":
            self._init_cpu_backend()
        else:
            raise ValueError(f"Unsupported backend: {backend}, choose from 'cpu', 'taichi', or 'jax'")

        self.site_name = site_name
        self._sensor_pose = np.eye(4, dtype=np.float32)
        self._distances = None
        self._hit_points = None

    def _init_taichi_backend(self):
        """Initialize Taichi backend"""
        try:
            # Lazy import: only import when Taichi backend is selected
            from mujoco_lidar.core_ti.mjlidar_ti import MjLidarTi
            import taichi as ti
            
            # Initialize Taichi if not already done
            if not hasattr(ti, '_is_initialized') or not ti._is_initialized:
                ti.init(arch=ti.gpu, **self.args.get('ti_init_args', {}))
            
            # Create Taichi backend instance
            geomgroup = self.args.get('geomgroup', None)
            bodyexclude = self.args.get('bodyexclude', -1)
            max_candidates = self.args.get('max_candidates', 32)
            self._backend_instance = MjLidarTi(
                self.mj_model, 
                cutoff_dist=self.cutoff_dist,
                geomgroup=geomgroup,
                bodyexclude=bodyexclude,
                max_candidates=max_candidates
            )
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import Taichi backend dependencies. "
                f"Please install taichi: pip install taichi\n"
                f"Error: {e}"
            )

    def _init_jax_backend(self):
        """Initialize JAX backend"""
        try:
            from mujoco_lidar.core_jax.mjlidar_jax import MjLidarJax
            
            # Pass mj_model directly. MjLidarJax will extract what it needs.
            self._backend_instance = MjLidarJax(self.mj_model, geom_ids=self.args.get('geom_ids'))
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import JAX backend dependencies.\n"
                f"Error: {e}"
            )
    
    def _init_cpu_backend(self):
        """Initialize CPU backend"""
        try:
            from mujoco_lidar.core_cpu.mjlidar_cpu import MjLidarCPU
            
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

    def update_sensor_pose(self, mj_data, site_name:str):
        # For CPU/Taichi backend, mj_data is mujoco.MjData
        if self.backend in ['cpu', 'taichi']:
            self._sensor_pose[:3,:3] = mj_data.site(site_name).xmat.reshape(3,3)
            self._sensor_pose[:3,3] = mj_data.site(site_name).xpos

    def trace_rays(self, mj_data, ray_theta, ray_phi, site_name:str=None):
        """
        Trace rays.
        For JAX backend, mj_data can be mujoco.MjData.
        """
        target_site = self.site_name if site_name is None else site_name

        if self.backend == "jax":
            import jax.numpy as jnp
            
            # Get site ID
            site_id = self.mj_model.site(target_site).id
            
            # Assume single environment (CPU MjData)
            # Extract data needed for rendering
            sensor_pos = jnp.array(mj_data.site(target_site).xpos)
            sensor_mat = jnp.array(mj_data.site(target_site).xmat.reshape(3, 3))
            
            geom_xpos = jnp.array(mj_data.geom_xpos)
            geom_xmat = jnp.array(mj_data.geom_xmat)

            # Convert angles to local rays
            theta = jnp.array(ray_theta)
            phi = jnp.array(ray_phi)
            
            x = jnp.cos(phi) * jnp.cos(theta)
            y = jnp.cos(phi) * jnp.sin(theta)
            z = jnp.sin(phi)
            local_rays = jnp.stack([x, y, z], axis=-1) # (Nrays, 3)
            
            # Transform to world rays
            world_rays = local_rays @ sensor_mat.T
                
            # Render
            self._distances = self._backend_instance.render(geom_xpos, geom_xmat, sensor_pos, world_rays)
            
            return self._distances
            
        elif self.backend == "taichi":
            # Taichi Backend
            self.update_sensor_pose(mj_data, target_site)
            self._backend_instance.update(mj_data)
            
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
            return self._backend_instance.get_distances()

        else:
            # CPU Backend
            self.update_sensor_pose(mj_data, target_site)
            self._backend_instance.update(mj_data)
            self._backend_instance.trace_rays(self._sensor_pose, ray_theta, ray_phi)
            return self._backend_instance.get_distances()
    
    def get_hit_points(self):
        if self.backend == "jax":
            return None 
        return self._backend_instance.get_hit_points()
    
    def get_distances(self):
        if self.backend == "jax":
            return self._distances
        return self._backend_instance.get_distances()

