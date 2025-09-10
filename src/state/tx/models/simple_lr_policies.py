"""
Simple learning rate policy system that directly uses PyTorch schedulers.

This module provides a clean interface for configuring learning rate schedules
using PyTorch's built-in schedulers without unnecessary re-implementation.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    ChainedScheduler,
    SequentialLR,
    LambdaLR,
    ConstantLR,
    PolynomialLR,
)

import lightning as L
from lightning.pytorch.callbacks import Callback


class SimpleLearningRatePolicy:
    """
    Simple learning rate policy that directly uses PyTorch schedulers.
    
    This class provides a clean interface for configuring learning rate schedules
    without re-implementing PyTorch's built-in functionality.
    """
    
    def __init__(self, scheduler_name: str, **scheduler_kwargs):
        """
        Initialize learning rate policy.
        
        Args:
            scheduler_name: Name of the PyTorch scheduler to use
            **scheduler_kwargs: Arguments to pass to the scheduler
        """
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        self._scheduler_map = {
            'cosine_annealing': CosineAnnealingLR,
            'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts,
            'exponential': ExponentialLR,
            'linear': LinearLR,
            'one_cycle': OneCycleLR,
            'reduce_on_plateau': ReduceLROnPlateau,
            'step': StepLR,
            'multistep': MultiStepLR,
            'lambda': LambdaLR,
            'constant': ConstantLR,
            'polynomial': PolynomialLR,
        }
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a learning rate scheduler configuration.
        
        Args:
            optimizer: The optimizer to schedule
            total_steps: Total number of training steps (if needed)
            
        Returns:
            Dictionary with scheduler configuration for PyTorch Lightning
        """
        if self.scheduler_name not in self._scheduler_map:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}. Available: {list(self._scheduler_map.keys())}")
        
        scheduler_class = self._scheduler_map[self.scheduler_name]
        
        # Handle special cases that need total_steps
        if self.scheduler_name == 'one_cycle' and total_steps is not None:
            self.scheduler_kwargs['total_steps'] = total_steps
        
        # Create the scheduler
        scheduler = scheduler_class(optimizer, **self.scheduler_kwargs)
        
        # Determine monitoring and interval based on scheduler type
        if self.scheduler_name == 'reduce_on_plateau':
            return {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.scheduler_kwargs.get("monitor", "val_loss")
            }
        else:
            return {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
    
    def get_name(self) -> str:
        """Return the name of this learning rate policy."""
        return self.scheduler_name


class LearningRateMonitor(Callback):
    """
    Callback to monitor and log learning rate changes.
    """
    
    def __init__(self, logging_interval: str = "epoch"):
        self.logging_interval = logging_interval
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.logging_interval == "step":
            self._log_lr(trainer, pl_module)
    
    def on_train_epoch_start(self, trainer, pl_module):
        if self.logging_interval == "epoch":
            self._log_lr(trainer, pl_module)
    
    def _log_lr(self, trainer, pl_module):
        """Log the current learning rate."""
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
            pl_module.log("learning_rate", current_lr, on_step=True, on_epoch=False)


def should_use_scheduler(scheduler_name: Optional[str]) -> bool:
    """
    Check if a learning rate scheduler should be used.
    
    Args:
        scheduler_name: Name of the scheduler or None
        
    Returns:
        True if a scheduler should be used, False otherwise
    """
    if scheduler_name is None:
        return False
    
    # Strip whitespace and check against known "none" values
    cleaned = scheduler_name.strip().lower()
    return cleaned not in ["none", "null", ""]


def create_lr_policy(scheduler_name: str, **kwargs) -> SimpleLearningRatePolicy:
    """
    Create a learning rate policy with the specified scheduler.
    
    Args:
        scheduler_name: Name of the PyTorch scheduler to use
        **kwargs: Arguments for the scheduler
        
    Returns:
        SimpleLearningRatePolicy instance
    """
    # Define valid parameters for each scheduler
    valid_params = {
        "cosine_annealing": ["T_max", "eta_min"],
        "cosine_annealing_warm_restarts": ["T_0", "eta_min"],
        "one_cycle": ["max_lr", "pct_start", "div_factor", "final_div_factor"],
        "reduce_on_plateau": ["mode", "factor", "patience", "threshold", "cooldown", "min_lr"],
        "polynomial": ["total_iters", "power"],
        "exponential": ["gamma"],
        "step": ["step_size", "gamma"],
        "multistep": ["milestones", "gamma"],
        "lambda": ["lr_lambda"],
        "constant": [],
    }
    
    # Set default parameters for each scheduler
    default_params = {
        "cosine_annealing": {
            "T_max": 10000,
            "eta_min": 1e-6
        },
        "cosine_annealing_warm_restarts": {
            "T_0": 10000,
            "eta_min": 1e-6
        },
        "one_cycle": {
            "max_lr": 1e-4,
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 10000.0
        },
        "reduce_on_plateau": {
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "threshold": 1e-4,
            "cooldown": 0,
            "min_lr": 1e-6
        },
        "polynomial": {
            "total_iters": 10000,
            "power": 1.0
        },
        "exponential": {
            "gamma": 0.95
        },
        "step": {
            "step_size": 1000,
            "gamma": 0.9
        },
        "multistep": {
            "milestones": [1000, 2000, 3000],
            "gamma": 0.9
        },
        "lambda": {
            "lr_lambda": lambda epoch: 0.95 ** epoch
        },
        "constant": {},
    }
    
    # Get valid parameters for this scheduler
    valid_keys = valid_params.get(scheduler_name, [])
    
    # Filter kwargs to only include valid parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    # Get default parameters for this scheduler
    scheduler_params = default_params.get(scheduler_name, {})
    
    # Override with any provided valid kwargs
    scheduler_params.update(filtered_kwargs)
    
    return SimpleLearningRatePolicy(scheduler_name, **scheduler_params)


