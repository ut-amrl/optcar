"""Base controller class for all WheeledLab controllers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, Any
import torch

if TYPE_CHECKING:
    from wheeledlab_op.utils.state_struct import State_struct_torch


class BaseController(ABC):
    """Abstract base class for all controllers."""
    
    def __init__(self, config, **kwargs):
        """
        Initialize base controller.
        
        Args:
            config: Controller configuration object
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self.cntrl_config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        self.dtype = config.dtype if hasattr(config, 'dtype') else torch.float32
        
        # Task can be set later
        self.task = None
        
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset controller state.

        Args:
            seed: Optional seed to reset any controller RNG. When None, the
                controller should keep its previously configured seed behavior.
        """
        pass
    
    @abstractmethod
    def compute_control(
        self, 
        curr_states: State_struct_torch,
        **kwargs
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute control action based on current state.
        
        Args:
            curr_states: Current state of the robot(s)
            **kwargs: Additional arguments (past_states, past_actions, etc.)
            
        Returns:
            actions: (batch_size, action_dim) control actions
            info: Dictionary with additional information
        """
        pass
    
    def set_task(self, task):
        """Set the current task for the controller."""
        self.task = task
    
    def get_task(self):
        """Get the current task."""
        return self.task
