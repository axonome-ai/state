"""
Learning rate policy system with separated warm-up component.

This module provides learning rate scheduling strategies with clean separation
between warm-up and main schedulers, allowing for flexible configuration.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    SequentialLR,
    LambdaLR,
    ConstantLR,
    PolynomialLR,
)
from abc import ABC, abstractmethod

import lightning as L
from lightning.pytorch.callbacks import Callback

from .warmup import WarmupConfig, WarmupManager


class LearningRatePolicy(ABC):
    """
    Abstract base class for learning rate policies.
    
    This class defines the interface for learning rate scheduling strategies
    without built-in warm-up functionality.
    """
    
    def __init__(self, **kwargs):
        """Initialize the learning rate policy."""
        pass
    
    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        """
        Create a learning rate scheduler configuration.
        
        Args:
            optimizer: The optimizer to schedule
            total_steps: Total number of training steps
            
        Returns:
            Dictionary with scheduler configuration for PyTorch Lightning
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this learning rate policy."""
        pass


class CosineAnnealingPolicy(LearningRatePolicy):
    """
    Pure cosine annealing learning rate policy without warm-up.
    
    This policy implements cosine annealing, which is often effective
    for transformer-based models and state transition models.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        T_max: Optional[int] = None,
        eta_min_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max = T_max
        self.eta_min_ratio = eta_min_ratio
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        T_max = self.T_max or total_steps
        eta_min = self.max_lr * self.eta_min_ratio
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "train_loss"
        }
    
    def get_name(self) -> str:
        return "cosine_annealing"


class OneCyclePolicy(LearningRatePolicy):
    """
    One cycle learning rate policy without warm-up.
    
    This policy implements the 1cycle learning rate schedule for super-convergence.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        total_steps: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        actual_total_steps = self.total_steps or total_steps
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=actual_total_steps,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor
        )
        
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
    
    def get_name(self) -> str:
        return "one_cycle"


class WarmupCosineRestartsPolicy(LearningRatePolicy):
    """
    Cosine annealing with warm restarts without built-in warm-up.
    
    This policy can help escape local minima and improve convergence.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        T_0: int = 10000,
        T_mult: int = 1,
        eta_min_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_ratio = eta_min_ratio
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        eta_min = self.max_lr * self.eta_min_ratio
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=eta_min
        )
        
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "train_loss"
        }
    
    def get_name(self) -> str:
        return "warmup_cosine_restarts"


class ReduceOnPlateauPolicy(LearningRatePolicy):
    """
    Reduce learning rate on plateau without built-in warm-up.
    
    This policy reduces learning rate when validation loss plateaus.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        cooldown: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            cooldown=self.cooldown,
            min_lr=self.min_lr
        )
        
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
    
    def get_name(self) -> str:
        return "reduce_on_plateau"


class PolynomialDecayPolicy(LearningRatePolicy):
    """
    Polynomial decay learning rate policy without warm-up.
    
    This policy implements polynomial decay which can be effective for
    long training runs.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        power: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.power = power
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        scheduler = PolynomialLR(
            optimizer,
            total_iters=total_steps,
            power=self.power
        )
        
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
    
    def get_name(self) -> str:
        return "polynomial_decay"


class CustomLambdaPolicy(LearningRatePolicy):
    """
    Custom lambda-based learning rate policy without warm-up.
    
    This policy allows for custom learning rate functions.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        lr_lambda: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.lr_lambda = lr_lambda or (lambda epoch: 1.0)
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        scheduler = LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
    
    def get_name(self) -> str:
        return "custom_lambda"


class LearningRatePolicyFactory:
    """
    Factory class for creating learning rate policies.
    
    This factory supports both the new separated warm-up system and
    backward compatibility with the old system.
    """
    
    _policy_map = {
        'cosine_annealing': CosineAnnealingPolicy,
        'one_cycle': OneCyclePolicy,
        'warmup_cosine_restarts': WarmupCosineRestartsPolicy,
        'reduce_on_plateau': ReduceOnPlateauPolicy,
        'polynomial_decay': PolynomialDecayPolicy,
        'custom_lambda': CustomLambdaPolicy,
    }
    
    @classmethod
    def create_policy(cls, policy_name: str, **kwargs) -> LearningRatePolicy:
        """
        Create a learning rate policy by name.
        
        Args:
            policy_name: Name of the policy to create
            **kwargs: Arguments to pass to the policy constructor
            
        Returns:
            LearningRatePolicy instance
        """
        if policy_name not in cls._policy_map:
            raise ValueError(f"Unknown policy: {policy_name}. Available: {list(cls._policy_map.keys())}")
        
        policy_class = cls._policy_map[policy_name]
        return policy_class(**kwargs)
    
    @classmethod
    def create_combined_scheduler(
        cls,
        optimizer: Optimizer,
        policy_config: Dict[str, Any],
        warmup_config: Optional[Union[WarmupConfig, Dict[str, Any]]] = None,
        total_steps: int = 10000
    ) -> Dict[str, Any]:
        """
        Create a combined scheduler with warm-up and main scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            policy_config: Learning rate policy configuration
            warmup_config: Warm-up configuration (optional)
            total_steps: Total number of training steps
            
        Returns:
            Dictionary with scheduler configuration for PyTorch Lightning
        """
        # Create main scheduler
        policy_name = policy_config.pop('name')
        policy = cls.create_policy(policy_name, **policy_config)
        main_scheduler_config = policy.get_scheduler(optimizer, total_steps)
        main_scheduler = main_scheduler_config['scheduler']
        
        # Create combined scheduler with warm-up if specified
        if warmup_config is not None:
            combined_scheduler = WarmupManager.create_combined_scheduler(
                optimizer, warmup_config, main_scheduler, total_steps
            )
            
            return {
                "scheduler": combined_scheduler,
                "interval": main_scheduler_config.get("interval", "step"),
                "frequency": main_scheduler_config.get("frequency", 1),
                "monitor": main_scheduler_config.get("monitor")
            }
        else:
            return main_scheduler_config
    
    @classmethod
    def get_available_policies(cls) -> list[str]:
        """Get list of available policy names."""
        return list(cls._policy_map.keys())
    
    @classmethod
    def list_policies(cls) -> list[str]:
        """Get list of available policy names (alias for get_available_policies)."""
        return cls.get_available_policies()


def create_lr_policy_from_config(
    config: Dict[str, Any],
    warmup_config: Optional[Union[WarmupConfig, Dict[str, Any]]] = None
) -> LearningRatePolicy:
    """
    Create a learning rate policy from configuration.
    
    Args:
        config: Learning rate policy configuration
        warmup_config: Warm-up configuration (optional)
        
    Returns:
        LearningRatePolicy instance
    """
    policy_name = config.get('name')
    if not policy_name:
        raise ValueError("Policy configuration must include 'name' field")
    
    policy_kwargs = {k: v for k, v in config.items() if k != 'name'}
    return LearningRatePolicyFactory.create_policy(policy_name, **policy_kwargs)


def create_combined_scheduler_from_configs(
    optimizer: Optimizer,
    policy_config: Dict[str, Any],
    warmup_config: Optional[Union[WarmupConfig, Dict[str, Any]]] = None,
    total_steps: int = 10000
) -> Dict[str, Any]:
    """
    Create a combined scheduler from separate policy and warm-up configurations.
    
    Args:
        optimizer: The optimizer to schedule
        policy_config: Learning rate policy configuration
        warmup_config: Warm-up configuration (optional)
        total_steps: Total number of training steps
        
    Returns:
        Dictionary with scheduler configuration for PyTorch Lightning
    """
    return LearningRatePolicyFactory.create_combined_scheduler(
        optimizer, policy_config, warmup_config, total_steps
    )


# Backward compatibility: Legacy classes with built-in warm-up
class LegacyCosineAnnealingPolicy(LearningRatePolicy):
    """
    Legacy cosine annealing learning rate policy with built-in warm-up.
    
    This is kept for backward compatibility. New code should use
    CosineAnnealingPolicy + WarmupConfig instead.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 1000,
        warmup_ratio: float = 0.1,
        T_max: Optional[int] = None,
        eta_min_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.T_max = T_max
        self.eta_min_ratio = eta_min_ratio
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        T_max = self.T_max or total_steps
        eta_min = self.max_lr * self.eta_min_ratio
        
        if self.warmup_steps > 0:
            # Linear warmup followed by cosine annealing
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=T_max - self.warmup_steps,
                eta_min=eta_min
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "train_loss"
        }
    
    def get_name(self) -> str:
        return "legacy_cosine_annealing"


# Learning rate monitoring callback
class LearningRateMonitor(Callback):
    """
    Callback to monitor and log learning rate changes.
    """
    
    def __init__(self, logging_interval: str = "step"):
        """
        Initialize the learning rate monitor.
        
        Args:
            logging_interval: When to log learning rates ("step" or "epoch")
        """
        self.logging_interval = logging_interval
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.logging_interval == "step":
            self._log_lr(trainer, pl_module)
    
    def on_train_epoch_start(self, trainer, pl_module):
        if self.logging_interval == "epoch":
            self._log_lr(trainer, pl_module)
    
    def _log_lr(self, trainer, pl_module):
        """Log current learning rates."""
        if trainer.optimizers:
            for i, optimizer in enumerate(trainer.optimizers):
                for j, param_group in enumerate(optimizer.param_groups):
                    lr = param_group['lr']
                    pl_module.log(f"learning_rate/group_{i}_{j}", lr, on_step=True, on_epoch=False)