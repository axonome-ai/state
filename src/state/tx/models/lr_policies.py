"""
Learning rate policies for state transition models.

This module provides various learning rate scheduling strategies that can be used
with the state transition model to improve training stability and performance.
"""

import math
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

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
    ChainedScheduler,
    SequentialLR,
    LambdaLR,
    ConstantLR,
    PolynomialLR,
)

import lightning as L
from lightning.pytorch.callbacks import Callback


class LearningRatePolicy(ABC):
    """
    Abstract base class for learning rate policies.
    
    A learning rate policy defines how the learning rate should be scheduled
    during training, including warmup, decay, and restart strategies.
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Any:
        """
        Create a learning rate scheduler for the given optimizer.
        
        Args:
            optimizer: The optimizer to schedule
            total_steps: Total number of training steps
            
        Returns:
            A learning rate scheduler or configuration dict
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this learning rate policy."""
        pass


class CosineAnnealingPolicy(LearningRatePolicy):
    """
    Cosine annealing learning rate policy with optional warmup.
    
    This policy implements cosine annealing with warmup, which is often effective
    for transformer-based models and state transition models.
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
        return "cosine_annealing"


class OneCyclePolicy(LearningRatePolicy):
    """
    OneCycle learning rate policy with super-convergence.
    
    This policy implements the 1cycle learning rate schedule which can lead to
    faster convergence and better generalization.
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
        steps = self.total_steps or total_steps
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=steps,
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
    Cosine annealing with warm restarts and optional warmup.
    
    This policy can help escape local minima and improve convergence.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 1000,
        T_0: int = 10000,
        T_mult: int = 1,
        eta_min_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_ratio = eta_min_ratio
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        eta_min = self.max_lr * self.eta_min_ratio
        
        if self.warmup_steps > 0:
            # Linear warmup followed by cosine annealing with restarts
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=eta_min
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
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
    Reduce learning rate on plateau with optional warmup.
    
    This policy reduces the learning rate when the validation loss plateaus,
    which can help with fine-tuning and avoiding overfitting.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 1000,
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
        self.warmup_steps = warmup_steps
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        if self.warmup_steps > 0:
            # For warmup + plateau, we need to use a custom approach
            # since SequentialLR doesn't support ReduceLROnPlateau
            # We'll use a custom lambda scheduler that combines both
            def combined_lr_lambda(epoch):
                if epoch < self.warmup_steps:
                    # Warmup phase
                    return 0.1 + 0.9 * (epoch / self.warmup_steps)
                else:
                    # After warmup, return 1.0 (plateau scheduler will handle reduction)
                    return 1.0
            
            scheduler = LambdaLR(optimizer, lr_lambda=combined_lr_lambda)
            
            # Note: This is a simplified approach. In practice, you might want to
            # implement a more sophisticated warmup + plateau combination
        else:
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
    Polynomial decay learning rate policy with optional warmup.
    
    This policy implements polynomial decay which can be effective for
    long training runs.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 1000,
        power: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.power = power
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        if self.warmup_steps > 0:
            # Linear warmup followed by polynomial decay
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            poly_scheduler = PolynomialLR(
                optimizer,
                total_iters=total_steps - self.warmup_steps,
                power=self.power
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, poly_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
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
    Custom lambda-based learning rate policy.
    
    This policy allows for custom learning rate functions defined as lambda functions.
    """
    
    def __init__(
        self,
        max_lr: float = 1e-4,
        lr_lambda: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.lr_lambda = lr_lambda or self._default_lr_lambda
    
    def _default_lr_lambda(self, epoch: int) -> float:
        """Default lambda function for learning rate decay."""
        return 0.95 ** epoch
    
    def get_scheduler(self, optimizer: Optimizer, total_steps: int) -> Dict[str, Any]:
        scheduler = LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1
        }
    
    def get_name(self) -> str:
        return "custom_lambda"


class LearningRatePolicyFactory:
    """
    Factory class for creating learning rate policies.
    """
    
    _policies = {
        "cosine_annealing": CosineAnnealingPolicy,
        "one_cycle": OneCyclePolicy,
        "warmup_cosine_restarts": WarmupCosineRestartsPolicy,
        "reduce_on_plateau": ReduceOnPlateauPolicy,
        "polynomial_decay": PolynomialDecayPolicy,
        "custom_lambda": CustomLambdaPolicy,
    }
    
    @classmethod
    def create_policy(cls, policy_name: str, **kwargs) -> LearningRatePolicy:
        """
        Create a learning rate policy by name.
        
        Args:
            policy_name: Name of the policy to create
            **kwargs: Additional arguments for the policy
            
        Returns:
            A learning rate policy instance
            
        Raises:
            ValueError: If the policy name is not recognized
        """
        if policy_name not in cls._policies:
            available = ", ".join(cls._policies.keys())
            raise ValueError(f"Unknown policy '{policy_name}'. Available policies: {available}")
        
        return cls._policies[policy_name](**kwargs)
    
    @classmethod
    def list_policies(cls) -> List[str]:
        """Return a list of available policy names."""
        return list(cls._policies.keys())


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


def create_lr_policy_from_config(config: Dict[str, Any]) -> LearningRatePolicy:
    """
    Create a learning rate policy from a configuration dictionary.
    
    Args:
        config: Configuration dictionary containing policy settings
        
    Returns:
        A learning rate policy instance
    """
    policy_name = config.pop("name", "cosine_annealing")
    return LearningRatePolicyFactory.create_policy(policy_name, **config)
