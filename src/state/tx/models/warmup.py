"""
Independent warm-up component for learning rate scheduling.

This module provides a clean separation between warm-up and main learning rate schedulers,
allowing for flexible configuration and independent testing.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupConfig:
    """Configuration for warm-up phase."""
    
    def __init__(
        self,
        enabled: bool = True,
        steps: int = 1000,
        type: str = "linear",  # linear, cosine, exponential, constant
        start_factor: float = 0.1,
        end_factor: float = 1.0,
        **kwargs
    ):
        """
        Initialize warm-up configuration.
        
        Args:
            enabled: Whether warm-up is enabled
            steps: Number of warm-up steps
            type: Type of warm-up schedule (linear, cosine, exponential, constant)
            start_factor: Starting learning rate factor (relative to base LR)
            end_factor: Ending learning rate factor (relative to base LR)
            **kwargs: Additional warm-up specific parameters
        """
        self.enabled = enabled
        self.steps = steps
        self.type = type
        self.start_factor = start_factor
        self.end_factor = end_factor
        
        # Additional parameters for specific warm-up types
        self.exponential_gamma = kwargs.get('exponential_gamma', 0.9)
        self.cosine_eta_min = kwargs.get('cosine_eta_min', 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'steps': self.steps,
            'type': self.type,
            'start_factor': self.start_factor,
            'end_factor': self.end_factor,
            'exponential_gamma': self.exponential_gamma,
            'cosine_eta_min': self.cosine_eta_min
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WarmupConfig':
        """Create WarmupConfig from dictionary."""
        return cls(**config_dict)


class WarmupScheduler(_LRScheduler):
    """
    Independent warm-up scheduler that can be used with any main scheduler.
    
    This scheduler handles only the warm-up phase and can be chained with
    other schedulers using SequentialLR.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: WarmupConfig,
        last_epoch: int = -1
    ):
        """
        Initialize warm-up scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            config: Warm-up configuration
            last_epoch: The index of last epoch (default: -1)
        """
        self.config = config
        self.step_count = 0  # Initialize before super().__init__()
        super().__init__(optimizer, last_epoch)
        # Initialize _last_lr for compatibility with SequentialLR
        self._last_lr = [group['initial_lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self) -> list[float]:
        """Get the learning rate for the current step."""
        if not self.config.enabled or self.step_count > self.config.steps:
            # Warm-up is disabled or completed, return base learning rates
            return [group['initial_lr'] for group in self.optimizer.param_groups]
        
        # Calculate warm-up factor based on type
        if self.config.type == "linear":
            factor = self._linear_warmup()
        elif self.config.type == "cosine":
            factor = self._cosine_warmup()
        elif self.config.type == "exponential":
            factor = self._exponential_warmup()
        elif self.config.type == "constant":
            factor = self._constant_warmup()
        else:
            raise ValueError(f"Unknown warm-up type: {self.config.type}")
        
        # Apply factor to base learning rates
        return [group['initial_lr'] * factor for group in self.optimizer.param_groups]
    
    def _linear_warmup(self) -> float:
        """Linear warm-up from start_factor to end_factor."""
        progress = self.step_count / self.config.steps
        return self.config.start_factor + \
               (self.config.end_factor - self.config.start_factor) * progress
    
    def _cosine_warmup(self) -> float:
        """Cosine warm-up from start_factor to end_factor."""
        progress = self.step_count / self.config.steps
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return self.config.start_factor + \
               (self.config.end_factor - self.config.start_factor) * cosine_factor
    
    def _exponential_warmup(self) -> float:
        """Exponential warm-up from start_factor to end_factor."""
        progress = self.step_count / self.config.steps
        exp_factor = math.pow(self.config.exponential_gamma, self.config.steps - self.step_count)
        return self.config.start_factor + \
               (self.config.end_factor - self.config.start_factor) * (1 - exp_factor)
    
    def _constant_warmup(self) -> float:
        """Constant warm-up at start_factor."""
        return self.config.start_factor
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        # Only increment step_count if this is not the initial step
        if hasattr(self, '_initial_step_done'):
            self.step_count += 1
        else:
            self._initial_step_done = True
        
        # Update learning rates
        lrs = self.get_lr()
        self._last_lr = lrs  # Update _last_lr for compatibility
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


class WarmupManager:
    """
    Manager class for handling warm-up configuration and creation.
    
    This class provides a clean interface for creating warm-up schedulers
    and integrating them with main schedulers.
    """
    
    @staticmethod
    def create_warmup_scheduler(
        optimizer: Optimizer,
        config: Union[WarmupConfig, Dict[str, Any]],
        last_epoch: int = -1
    ) -> WarmupScheduler:
        """
        Create a warm-up scheduler from configuration.
        
        Args:
            optimizer: The optimizer to schedule
            config: Warm-up configuration (WarmupConfig or dict)
            last_epoch: The index of last epoch (default: -1)
            
        Returns:
            WarmupScheduler instance
        """
        if isinstance(config, dict):
            config = WarmupConfig.from_dict(config)
        
        return WarmupScheduler(optimizer, config, last_epoch)
    
    @staticmethod
    def create_combined_scheduler(
        optimizer: Optimizer,
        warmup_config: Union[WarmupConfig, Dict[str, Any], None],
        main_scheduler: _LRScheduler,
        total_steps: int
    ) -> _LRScheduler:
        """
        Create a combined scheduler with warm-up and main scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            warmup_config: Warm-up configuration (None to disable warm-up)
            main_scheduler: The main learning rate scheduler
            total_steps: Total number of training steps
            
        Returns:
            Combined scheduler (SequentialLR if warm-up enabled, main_scheduler otherwise)
        """
        from torch.optim.lr_scheduler import SequentialLR
        
        if warmup_config is None or (isinstance(warmup_config, dict) and not warmup_config.get('enabled', True)):
            return main_scheduler
        
        if isinstance(warmup_config, dict):
            warmup_config = WarmupConfig.from_dict(warmup_config)
        
        if not warmup_config.enabled:
            return main_scheduler
        
        # Create warm-up scheduler
        warmup_scheduler = WarmupManager.create_warmup_scheduler(optimizer, warmup_config)
        
        # Create combined scheduler
        combined_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_config.steps]
        )
        
        return combined_scheduler
    
    @staticmethod
    def get_warmup_config_from_training_config(training_config: Dict[str, Any]) -> Optional[WarmupConfig]:
        """
        Extract warm-up configuration from training configuration.
        
        Args:
            training_config: Training configuration dictionary
            
        Returns:
            WarmupConfig if warm-up is enabled, None otherwise
        """
        warmup_config = training_config.get('warmup', {})
        
        if not warmup_config.get('enabled', False):
            return None
        
        return WarmupConfig.from_dict(warmup_config)


def create_warmup_from_yaml(yaml_path: str) -> WarmupConfig:
    """
    Create WarmupConfig from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        WarmupConfig instance
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return WarmupConfig.from_dict(config)


def visualize_warmup_schedule(
    config: WarmupConfig,
    total_steps: int = 10000,
    base_lr: float = 1e-4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize warm-up schedule.
    
    Args:
        config: Warm-up configuration
        total_steps: Total number of steps to visualize
        base_lr: Base learning rate
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create dummy optimizer for visualization
    dummy_params = [torch.tensor([1.0], requires_grad=True)]
    dummy_optimizer = torch.optim.Adam(dummy_params, lr=base_lr)
    
    # Create warm-up scheduler
    warmup_scheduler = WarmupScheduler(dummy_optimizer, config)
    
    # Collect learning rates
    steps = []
    lrs = []
    
    for step in range(min(total_steps, config.steps + 1000)):
        steps.append(step)
        lrs.append(warmup_scheduler.get_lr()[0])
        warmup_scheduler.step()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, label=f'Warm-up ({config.type})')
    plt.axvline(x=config.steps, color='r', linestyle='--', alpha=0.7, label='Warm-up end')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title(f'Warm-up Schedule: {config.type} ({config.steps} steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Warm-up schedule saved to {save_path}")
    else:
        plt.show()
