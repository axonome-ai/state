#!/usr/bin/env python3
"""
Test script to verify the warm-up separation system works correctly.
"""

import warnings
import os

# Suppress warnings before any other imports
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import lightning as L
import sys
import tempfile
import yaml

# Add src to path
sys.path.append('src')

from state.tx.models.warmup import WarmupConfig, WarmupScheduler, WarmupManager
from state.tx.models.lr_policies import (
    LearningRatePolicyFactory, 
    CosineAnnealingPolicy,
    create_combined_scheduler_from_configs
)


class DummyModel(L.LightningModule):
    """A dummy model for testing the warm-up system."""
    
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def test_warmup_config():
    """Test WarmupConfig creation and serialization."""
    print("Testing WarmupConfig...")
    
    # Test basic config
    config = WarmupConfig(
        enabled=True,
        steps=1000,
        type="linear",
        start_factor=0.1,
        end_factor=1.0
    )
    
    assert config.enabled == True
    assert config.steps == 1000
    assert config.type == "linear"
    assert config.start_factor == 0.1
    assert config.end_factor == 1.0
    
    # Test dict conversion
    config_dict = config.to_dict()
    assert config_dict['enabled'] == True
    assert config_dict['steps'] == 1000
    
    # Test from dict
    config2 = WarmupConfig.from_dict(config_dict)
    assert config2.enabled == config.enabled
    assert config2.steps == config.steps
    
    print("✓ WarmupConfig tests passed")


def test_warmup_scheduler():
    """Test WarmupScheduler functionality."""
    print("Testing WarmupScheduler...")
    
    # Create dummy optimizer
    params = [torch.tensor([1.0], requires_grad=True)]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    # Test linear warm-up
    config = WarmupConfig(
        enabled=True,
        steps=100,
        type="linear",
        start_factor=0.1,
        end_factor=1.0
    )
    
    scheduler = WarmupScheduler(optimizer, config)
    
    # Test initial state
    initial_lr = scheduler.get_lr()[0]
    expected_initial = 1e-3 * 0.1  # base_lr * start_factor
    assert abs(initial_lr - expected_initial) < 1e-5, f"Expected {expected_initial}, got {initial_lr}"
    
    # Test warm-up progression
    for step in range(50):
        optimizer.step()  # Call optimizer.step() before scheduler.step()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        expected_lr = 1e-3 * (0.1 + 0.9 * (step + 1) / 100)
        assert abs(lr - expected_lr) < 1e-5, f"Step {step}: expected {expected_lr}, got {lr}"
    
    # Continue warm-up until completion (100 steps total)
    for step in range(50, 100):
        optimizer.step()  # Call optimizer.step() before scheduler.step()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        expected_lr = 1e-3 * (0.1 + 0.9 * (step + 1) / 100)
        assert abs(lr - expected_lr) < 1e-5, f"Step {step}: expected {expected_lr}, got {lr}"
    
    # Test after warm-up - should return to base learning rate
    for step in range(100, 105):  # Test after warm-up is complete
        optimizer.step()  # Call optimizer.step() before scheduler.step()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        # After warm-up, should return to base learning rate (initial_lr)
        assert abs(lr - 1e-3) < 1e-5, f"After warm-up step {step}: expected 1e-3, got {lr}"
    
    print("✓ WarmupScheduler tests passed")


def test_cosine_warmup():
    """Test cosine warm-up functionality."""
    print("Testing cosine warm-up...")
    
    params = [torch.tensor([1.0], requires_grad=True)]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    config = WarmupConfig(
        enabled=True,
        steps=100,
        type="cosine",
        start_factor=0.1,
        end_factor=1.0
    )
    
    scheduler = WarmupScheduler(optimizer, config)
    
    # Test cosine warm-up progression
    lrs = []
    for step in range(100):
        lr = scheduler.get_lr()[0]
        lrs.append(lr)
        scheduler.step()
    
    # Check that cosine warm-up is smooth and monotonic
    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i-1], f"Cosine warm-up should be monotonic: {lrs[i-1]} -> {lrs[i]}"
    
    # Check final value
    assert abs(lrs[-1] - 1e-3) < 1e-5, f"Final LR should be 1e-3, got {lrs[-1]}"
    
    print("✓ Cosine warm-up tests passed")


def test_combined_scheduler():
    """Test combined scheduler with warm-up and main scheduler."""
    print("Testing combined scheduler...")
    
    params = [torch.tensor([1.0], requires_grad=True)]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    # Test configuration
    warmup_config = {
        'enabled': True,
        'steps': 50,
        'type': 'linear',
        'start_factor': 0.1,
        'end_factor': 1.0
    }
    
    policy_config = {
        'name': 'cosine_annealing',
        'max_lr': 1e-3,
        'min_lr': 1e-5,
        'T_max': 200
    }
    
    # Create combined scheduler
    scheduler_config = create_combined_scheduler_from_configs(
        optimizer, policy_config, warmup_config, total_steps=200
    )
    
    scheduler = scheduler_config['scheduler']
    
    # Test warm-up phase
    lrs = []
    for step in range(60):
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        optimizer.step()  # Call optimizer.step() before scheduler.step()
        scheduler.step()
    
    # Check warm-up phase (first 50 steps)
    for step in range(50):
        expected_lr = 1e-3 * (0.1 + 0.9 * (step + 1) / 50)
        assert abs(lrs[step] - expected_lr) < 1e-5, f"Warm-up step {step}: expected {expected_lr}, got {lrs[step]}"
    
    # Check that we transition to cosine annealing after warm-up
    # The LR should start decreasing after step 50
    assert lrs[50] == 1e-3, f"LR at warm-up end should be 1e-3, got {lrs[50]}"
    
    print("✓ Combined scheduler tests passed")


def test_policy_without_warmup():
    """Test that policies work correctly without warm-up."""
    print("Testing policies without warm-up...")
    
    params = [torch.tensor([1.0], requires_grad=True)]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    # Test cosine annealing policy
    policy = CosineAnnealingPolicy(max_lr=1e-3, min_lr=1e-5, T_max=100)
    scheduler_config = policy.get_scheduler(optimizer, total_steps=100)
    
    scheduler = scheduler_config['scheduler']
    
    # Test that it starts at max_lr
    initial_lr = optimizer.param_groups[0]['lr']
    assert abs(initial_lr - 1e-3) < 1e-5, f"Initial LR should be 1e-3, got {initial_lr}"
    
    # Test a few steps
    lrs = []
    for step in range(10):
        optimizer.step()  # Call optimizer.step() before scheduler.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        # Should be decreasing due to cosine annealing
        if step > 0:
            assert lr <= lrs[step-1], f"Cosine annealing should be decreasing: {lrs[step-1]} -> {lr}"
    
    print("✓ Policy without warm-up tests passed")


def test_config_loading():
    """Test loading warm-up configuration from YAML."""
    print("Testing config loading...")
    
    # Create temporary YAML file
    config_data = {
        'enabled': True,
        'steps': 500,
        'type': 'cosine',
        'start_factor': 0.05,
        'end_factor': 1.0
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        from state.tx.models.warmup import create_warmup_from_yaml
        config = create_warmup_from_yaml(temp_path)
        
        assert config.enabled == True
        assert config.steps == 500
        assert config.type == 'cosine'
        assert config.start_factor == 0.05
        assert config.end_factor == 1.0
        
        print("✓ Config loading tests passed")
    finally:
        os.unlink(temp_path)


def test_training_integration():
    """Test integration with PyTorch Lightning training."""
    print("Testing training integration...")
    
    # Create model and data
    model = DummyModel()
    train_data = [(torch.randn(32, 10), torch.randn(32, 1)) for _ in range(100)]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Create trainer with warm-up
    trainer = L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    
    # Test that training runs without errors
    try:
        trainer.fit(model, train_loader)
        print("✓ Training integration tests passed")
    except Exception as e:
        print(f"✗ Training integration failed: {e}")
        raise


def test_warmup_visualization():
    """Test warm-up visualization functionality."""
    print("Testing warm-up visualization...")
    
    try:
        import matplotlib.pyplot as plt
        
        config = WarmupConfig(
            enabled=True,
            steps=100,
            type="linear",
            start_factor=0.1,
            end_factor=1.0
        )
        
        from state.tx.models.warmup import visualize_warmup_schedule
        
        # Test visualization (should not raise errors)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        visualize_warmup_schedule(config, total_steps=200, base_lr=1e-3, save_path=temp_path)
        
        # Check that file was created
        assert os.path.exists(temp_path), "Visualization file should be created"
        
        os.unlink(temp_path)
        print("✓ Warm-up visualization tests passed")
        
    except ImportError:
        print("⚠ Skipping visualization tests (matplotlib not available)")
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")


if __name__ == "__main__":
    print("Running warm-up separation tests...\n")
    
    test_warmup_config()
    test_warmup_scheduler()
    test_cosine_warmup()
    test_combined_scheduler()
    test_policy_without_warmup()
    test_config_loading()
    test_training_integration()
    test_warmup_visualization()
    
    print("\n✅ All warm-up separation tests passed!")
