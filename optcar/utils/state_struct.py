"""State structures for batched simulation."""

import numpy as np
import torch


class State_struct:
    """NumPy-based state structure."""
    
    def __init__(self, pos=np.zeros(3), 
                       vel=np.zeros(3),
                       acc=np.zeros(3),
                       quat=np.array([1.0, 0.0, 0.0, 0.0]),
                       dead=np.zeros(1),
                       n_envs=1):
        
        self.pos = np.ones((n_envs, 3)) * pos
        self.vel = np.ones((n_envs, 3)) * vel 
        self.acc = np.ones((n_envs, 3)) * acc
        self.quat = np.ones((n_envs, 4)) * quat
        self.dead = np.ones((n_envs, 1)) * dead
        self.n_envs = n_envs


class State_struct_torch:
    """PyTorch-based state structure for batched environments."""
    
    pos: torch.Tensor
    pos_local: torch.Tensor
    pos_global: torch.Tensor
    vel: torch.Tensor
    acc: torch.Tensor
    quat: torch.Tensor
    dead: torch.Tensor
    ang_vel: torch.Tensor
    wheel_vel: torch.Tensor

    def __init__(self, pos=torch.zeros(3), 
                       vel=torch.zeros(3),
                       acc=torch.zeros(3),
                       quat=torch.tensor([1.0, 0.0, 0.0, 0.0]), 
                       dead=torch.zeros(1),
                       ang_vel=torch.zeros(3),
                       wheel_vel=torch.zeros(4),
                       n_envs=1,
                       dtype=torch.float32,
                       device='cpu'):

        self.device = device
        self.dtype = dtype

        # Dynamically process each state attribute
        for attr_name, default_value in {
            "pos": pos,
            "vel": vel,
            "acc": acc,
            "quat": quat,
            "dead": dead,
            "ang_vel": ang_vel,
            "wheel_vel": wheel_vel,
        }.items():
            value = default_value
            if len(value.shape) == 2:
                if value.shape[0] != n_envs:
                    raise ValueError(f"Number of environments in {attr_name} not equal to n_envs")
                setattr(self, 
                        attr_name, 
                        value.to(self.device))
            else:
                setattr(self, 
                        attr_name, 
                        torch.ones((n_envs, *value.shape),
                                    dtype=dtype,
                                    device=self.device) * value.to(device=self.device, 
                                                                   dtype=self.dtype)
                        )

        self.n_envs = n_envs
        self.dtype = dtype
        # Track both local (env-relative) and global positions.
        self.pos_local = self.pos.clone()
        self.pos_global = self.pos.clone()

    def to(self, device):
        """Move all tensor attributes to the given device."""
        self.device = device
        for attr_name in ('pos', 'vel', 'acc', 'quat', 'dead',
                          'ang_vel', 'wheel_vel', 'pos_local', 'pos_global'):
            val = getattr(self, attr_name, None)
            if isinstance(val, torch.Tensor):
                setattr(self, attr_name, val.to(device))
        return self

    @classmethod
    def from_obs(cls, obs, device='cpu'):
        """
        Create State_struct_torch from observation tensor.
        
        Observation format (13-dim policy state): [pos(3), quat(4), vel(3), ang_vel(3)]
        
        Args:
            obs: Observation tensor of shape (n_envs, 13) or dict with 'policy' key.
                Extra dict entries, such as sensor costmaps, are ignored here.
            device: Device to place tensors on
            
        Returns:
            State_struct_torch instance
        """
        # Handle dict observations (Isaac Lab format)
        if isinstance(obs, dict):
            obs_tensor = obs['policy']
        else:
            obs_tensor = obs
        
        # Ensure it's a tensor
        if not isinstance(obs_tensor, torch.Tensor):
            obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32, device=device)
        
        # Get number of environments
        if obs_tensor.ndim == 1:
            n_envs = 1
            obs_tensor = obs_tensor.unsqueeze(0)
        else:
            n_envs = obs_tensor.shape[0]
        
        # Extract state components from policy observation.
        # Format: [pos(3), quat(4), vel(3), ang_vel(3)]
        pos = obs_tensor[:, 0:3]
        quat = obs_tensor[:, 3:7]
        vel = obs_tensor[:, 7:10]
        ang_vel = obs_tensor[:, 10:13]
        # wheel_vel = obs_tensor[:, 13:17]
        # action = obs_tensor[:, 17:19]  # Not stored in state struct
        
        # Create instance
        state = cls(
            pos=pos[0] if n_envs == 1 else pos,
            vel=vel[0] if n_envs == 1 else vel,
            quat=quat[0] if n_envs == 1 else quat,
            ang_vel=ang_vel[0] if n_envs == 1 else ang_vel,
            wheel_vel=torch.zeros((n_envs, 4), device=device) if n_envs > 1 else torch.zeros(4, device=device),
            acc=torch.zeros(3, device=device),
            dead=torch.zeros(1, device=device),
            n_envs=n_envs,
            dtype=obs_tensor.dtype,
            device=device
        )
        
        return state

    def getNumpyStates(self):
        """Convert to NumPy state structure."""
        return State_struct(
            self.pos.detach().cpu().numpy(),
            self.vel.detach().cpu().numpy(),
            self.acc.detach().cpu().numpy(),
            self.quat.detach().cpu().numpy(),
            self.dead.detach().cpu().numpy(),
        )
    
    def get_vector_state(self, req_states=["pos", "quat", "dead"], **kwargs):
        """
        Get concatenated vector of requested state components.
        
        Args:
            req_states: List of state names to include
            **kwargs: Additional arguments (e.g., debug=True)
            
        Returns:
            Concatenated tensor of requested states
        """
        states = []
        if "pos" in req_states: 
            states.append(self.pos)
        if "pos_global" in req_states:
            states.append(self.pos_global)
        if "pos_local" in req_states:
            states.append(self.pos_local)
        if "quat" in req_states:
            states.append(self.quat)
        if "dead" in req_states:
            states.append(self.dead)
        if "vel" in req_states:
            states.append(self.vel)
        if "acc" in req_states:
            states.append(self.acc)
        if "ang_vel" in req_states:
            states.append(self.ang_vel)
        if "wheel_vel" in req_states:
            states.append(self.wheel_vel)
        
        state_vec = torch.concatenate(states, dim=1)
        return state_vec
    
    def copy_to_n_envs(self, n_envs):
        """Create a copy with different number of environments."""
        return State_struct_torch(
            pos=self.pos[0],
            vel=self.vel[0],
            acc=self.acc[0],
            ang_vel=self.ang_vel[0],
            quat=self.quat[0],
            dead=self.dead[0],
            n_envs=n_envs,
            dtype=self.dtype,
            device=self.device
        )

    def set_global_positions(self, env_origins=None):
        """Update state to include global/world positions.

        Args:
            env_origins: Tensor/array of shape (n_envs, 3) giving each
                environment's world origin in meters. If None, global
                positions stay equal to local positions.
        """
        self.pos_local = self.pos.clone()

        if env_origins is None:
            self.pos_global = self.pos.clone()
            return

        if isinstance(env_origins, torch.Tensor):
            origins = env_origins.to(self.device, dtype=self.dtype)
        else:
            origins = torch.as_tensor(env_origins, dtype=self.dtype, device=self.device)

        if origins.ndim == 1:
            origins = origins.unsqueeze(0)

        if origins.shape[0] != self.n_envs:
            raise ValueError(
                f"env_origins has {origins.shape[0]} rows but state has {self.n_envs} envs"
            )

        if origins.shape[1] < self.pos.shape[1]:
            raise ValueError(
                "env_origins must provide at least as many position components as the state"
            )

        self.pos = self.pos + origins[:, : self.pos.shape[1]]
        self.pos_global = self.pos.clone()
