"""
Utility functions for working with learning rate policies.

This module provides helper functions for creating, configuring, and visualizing
learning rate policies for state transition models.
"""

import yaml
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import torch
import numpy as np

from .lr_policies import LearningRatePolicyFactory, LearningRateMonitor


def load_lr_policy_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load a learning rate policy configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the learning rate policy configuration
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_lr_policy_from_yaml(yaml_path: str):
    """
    Create a learning rate policy from a YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        A learning rate policy instance
    """
    config = load_lr_policy_from_yaml(yaml_path)
    return LearningRatePolicyFactory.create_policy(config["name"], **config)


def visualize_lr_schedule(
    policy,
    total_steps: int = 10000,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Visualize the learning rate schedule for a given policy.
    
    Args:
        policy: Learning rate policy instance
        total_steps: Total number of training steps
        save_path: Optional path to save the plot
        title: Optional title for the plot
    """
    # Create a dummy optimizer
    dummy_params = [torch.nn.Parameter(torch.randn(1))]
    optimizer = torch.optim.AdamW(dummy_params, lr=1e-4)
    
    # Get scheduler configuration
    scheduler_config = policy.get_scheduler(optimizer, total_steps)
    
    # Create scheduler
    if isinstance(scheduler_config, dict):
        scheduler = scheduler_config["scheduler"]
    else:
        scheduler = scheduler_config
    
    # Simulate learning rate changes
    lrs = []
    steps = []
    
    for step in range(total_steps):
        if hasattr(scheduler, 'get_last_lr'):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        lrs.append(current_lr)
        steps.append(step)
        
        # Step the scheduler
        if isinstance(scheduler_config, dict):
            if scheduler_config.get("interval") == "step":
                scheduler.step()
        else:
            scheduler.step()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title(title or f'Learning Rate Schedule: {policy.get_name()}')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_lr_policies(
    policy_configs: List[Dict[str, Any]],
    total_steps: int = 10000,
    save_path: Optional[str] = None
):
    """
    Compare multiple learning rate policies on the same plot.
    
    Args:
        policy_configs: List of policy configuration dictionaries
        total_steps: Total number of training steps
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for config in policy_configs:
        policy = LearningRatePolicyFactory.create_policy(config["name"], **config)
        
        # Create dummy optimizer
        dummy_params = [torch.nn.Parameter(torch.randn(1))]
        optimizer = torch.optim.AdamW(dummy_params, lr=config.get("max_lr", 1e-4))
        
        # Get scheduler configuration
        scheduler_config = policy.get_scheduler(optimizer, total_steps)
        
        # Create scheduler
        if isinstance(scheduler_config, dict):
            scheduler = scheduler_config["scheduler"]
        else:
            scheduler = scheduler_config
        
        # Simulate learning rate changes
        lrs = []
        steps = []
        
        for step in range(total_steps):
            if hasattr(scheduler, 'get_last_lr'):
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            lrs.append(current_lr)
            steps.append(step)
            
            # Step the scheduler
            if isinstance(scheduler_config, dict):
                if scheduler_config.get("interval") == "step":
                    scheduler.step()
            else:
                scheduler.step()
        
        plt.plot(steps, lrs, linewidth=2, label=policy.get_name())
    
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Policy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def get_recommended_lr_policy(
    model_type: str = "state_transition",
    training_steps: int = 10000,
    dataset_size: str = "medium"
) -> Dict[str, Any]:
    """
    Get recommended learning rate policy based on model type and training setup.
    
    Args:
        model_type: Type of model ("state_transition", "transformer", etc.)
        training_steps: Expected number of training steps
        dataset_size: Size of dataset ("small", "medium", "large")
        
    Returns:
        Recommended learning rate policy configuration
    """
    recommendations = {
        "state_transition": {
            "small": {
                "name": "cosine_annealing",
                "max_lr": 1e-4,
                "min_lr": 1e-6,
                "warmup_steps": 500,
                "eta_min_ratio": 0.1
            },
            "medium": {
                "name": "cosine_annealing",
                "max_lr": 1e-4,
                "min_lr": 1e-6,
                "warmup_steps": 1000,
                "eta_min_ratio": 0.1
            },
            "large": {
                "name": "warmup_cosine_restarts",
                "max_lr": 1e-4,
                "min_lr": 1e-6,
                "warmup_steps": 2000,
                "T_0": training_steps // 4,
                "eta_min_ratio": 0.1
            }
        },
        "transformer": {
            "small": {
                "name": "one_cycle",
                "max_lr": 1e-4,
                "pct_start": 0.3,
                "div_factor": 25.0,
                "final_div_factor": 10000.0
            },
            "medium": {
                "name": "cosine_annealing",
                "max_lr": 1e-4,
                "min_lr": 1e-6,
                "warmup_steps": 1000,
                "eta_min_ratio": 0.1
            },
            "large": {
                "name": "warmup_cosine_restarts",
                "max_lr": 1e-4,
                "min_lr": 1e-6,
                "warmup_steps": 2000,
                "T_0": training_steps // 3,
                "eta_min_ratio": 0.1
            }
        }
    }
    
    if model_type not in recommendations:
        model_type = "state_transition"
    
    if dataset_size not in recommendations[model_type]:
        dataset_size = "medium"
    
    return recommendations[model_type][dataset_size]


def create_lr_policy_callback(logging_interval: str = "epoch") -> LearningRateMonitor:
    """
    Create a learning rate monitoring callback.
    
    Args:
        logging_interval: How often to log learning rate ("step" or "epoch")
        
    Returns:
        LearningRateMonitor callback instance
    """
    return LearningRateMonitor(logging_interval=logging_interval)


def print_available_policies():
    """Print all available learning rate policies."""
    policies = LearningRatePolicyFactory.list_policies()
    print("Available Learning Rate Policies:")
    print("-" * 40)
    for policy in policies:
        print(f"  - {policy}")
    print()


def print_policy_info(policy_name: str):
    """
    Print detailed information about a specific learning rate policy.
    
    Args:
        policy_name: Name of the policy to get information about
    """
    try:
        policy_class = LearningRatePolicyFactory._policies[policy_name]
        print(f"Policy: {policy_name}")
        print("-" * 40)
        print(f"Class: {policy_class.__name__}")
        print(f"Description: {policy_class.__doc__}")
        
        # Get default parameters
        import inspect
        sig = inspect.signature(policy_class.__init__)
        print("\nParameters:")
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                default = param.default if param.default != inspect.Parameter.empty else "Required"
                print(f"  - {param_name}: {default}")
        
    except KeyError:
        print(f"Policy '{policy_name}' not found.")
        print_available_policies()
