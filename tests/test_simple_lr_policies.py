"""
Tests for simple learning rate policies.

This module tests the simplified learning rate policy system that directly uses
PyTorch's built-in schedulers.
"""

import os
import sys
import pytest
import torch
import lightning as L
from unittest.mock import Mock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from state.tx.models.simple_lr_policies import (
    SimpleLearningRatePolicy,
    create_lr_policy,
    should_use_scheduler
)
from state.tx.models.state_transition import StateTransitionPerturbationModel


class TestSimpleLearningRatePolicies:
    """Test suite for simple learning rate policies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
        self.optimizer = torch.optim.AdamW(self.dummy_params, lr=1e-4)
        self.total_steps = 1000
    
    def test_policy_creation(self):
        """Test creating learning rate policies."""
        # Test cosine annealing
        policy = create_lr_policy("cosine_annealing", T_max=1000, eta_min=1e-6)
        assert isinstance(policy, SimpleLearningRatePolicy)
        assert policy.get_name() == "cosine_annealing"
        
        # Test one cycle
        policy = create_lr_policy("one_cycle", max_lr=1e-4, pct_start=0.3)
        assert isinstance(policy, SimpleLearningRatePolicy)
        assert policy.get_name() == "one_cycle"
    
    def test_scheduler_generation(self):
        """Test that schedulers are generated correctly."""
        # Test cosine annealing
        policy = create_lr_policy("cosine_annealing", T_max=1000, eta_min=1e-6)
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        assert "frequency" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            scheduler.step()
    
    def test_reduce_on_plateau(self):
        """Test reduce on plateau scheduler."""
        policy = create_lr_policy(
            "reduce_on_plateau",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "monitor" in scheduler_config
        assert scheduler_config["interval"] == "epoch"
    
    def test_one_cycle_with_total_steps(self):
        """Test one cycle scheduler with total steps."""
        policy = create_lr_policy("one_cycle", max_lr=1e-4, pct_start=0.3)
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            scheduler.step()
    
    def test_invalid_scheduler(self):
        """Test that invalid scheduler names raise ValueError."""
        policy = create_lr_policy("invalid_scheduler")
        with pytest.raises(ValueError, match="Unknown scheduler"):
            policy.get_scheduler(self.optimizer, self.total_steps)
    
    def test_should_use_scheduler(self):
        """Test the should_use_scheduler helper function."""
        # Test cases that should return False (no scheduler)
        assert not should_use_scheduler(None)
        assert not should_use_scheduler("none")
        assert not should_use_scheduler("None")
        assert not should_use_scheduler("NULL")
        assert not should_use_scheduler("null")
        assert not should_use_scheduler("")
        assert not should_use_scheduler("   ")
        
        # Test cases that should return True (use scheduler)
        assert should_use_scheduler("cosine_annealing")
        assert should_use_scheduler("one_cycle")
        assert should_use_scheduler("reduce_on_plateau")
        assert should_use_scheduler("step")
        assert should_use_scheduler("exponential")
    
    


class TestStateTransitionModelWithSimpleLRPolicies:
    """Test state transition model integration with simple learning rate policies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_config = {
            "input_dim": 5120,
            "hidden_dim": 512,
            "output_dim": 512,
            "pert_dim": 512,
            "batch_dim": 10,
            "predict_residual": True,
            "distributional_loss": "energy",
            "transformer_backbone_key": "GPT2",
            "transformer_backbone_kwargs": {
                "max_position_embeddings": 256,
                "hidden_size": 512,
                "intermediate_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "use_cache": False,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "layer_norm_eps": 1e-6,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "tie_word_embeddings": False,
                "rotary_dim": 0,
                "use_rotary_embeddings": False,
            },
            "output_space": "gene",
            "gene_dim": 2000,
            "embed_key": "X_hvg",
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "cell_set_len": 256,
            "dropout": 0.1,
            "lr": 1e-4,
            "weight_decay": 0.01
        }
    
    def test_model_without_lr_policy(self):
        """Test model without learning rate policy."""
        model = StateTransitionPerturbationModel(**self.model_config)
        
        # Test that lr_policy is None
        assert model.lr_policy is None
        
        # Test configure_optimizers returns just optimizer
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, torch.optim.AdamW)
    
    def test_model_with_none_lr_scheduler(self):
        """Test model with None lr_scheduler."""
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_scheduler=None
        )
        
        # Test that lr_policy is None
        assert model.lr_policy is None
        
        # Test configure_optimizers returns just optimizer
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, torch.optim.AdamW)
    
    def test_model_with_various_none_values(self):
        """Test model with various 'none' string values."""
        none_values = ["none", "None", "NULL", "null", ""]
        
        for none_val in none_values:
            model = StateTransitionPerturbationModel(
                **self.model_config,
                lr_scheduler=none_val
            )
            
            # Test that lr_policy is None
            assert model.lr_policy is None, f"Failed for lr_scheduler='{none_val}'"
            
            # Test configure_optimizers returns just optimizer
            optimizer_config = model.configure_optimizers()
            assert isinstance(optimizer_config, torch.optim.AdamW), f"Failed for lr_scheduler='{none_val}'"
    
    def test_model_with_cosine_annealing(self):
        """Test model with cosine annealing policy."""
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_scheduler="cosine_annealing",
            T_max=1000,
            eta_min=1e-6
        )

        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000

        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_model_with_one_cycle(self):
        """Test model with one cycle policy."""
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_scheduler="one_cycle",
            max_lr=1e-4,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )

        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000

        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_model_with_reduce_on_plateau(self):
        """Test model with reduce on plateau policy."""
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_scheduler="reduce_on_plateau",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000

        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_all_common_schedulers(self):
        """Test all common schedulers with the model."""
        schedulers = [
            "cosine_annealing", "cosine_annealing_warm_restarts", "one_cycle",
            "reduce_on_plateau", "polynomial", "exponential", "step", 
            "multistep", "lambda", "constant"
        ]
        
        for scheduler_name in schedulers:
            model = StateTransitionPerturbationModel(
                **self.model_config,
                lr_scheduler=scheduler_name
            )
            
            # Mock trainer
            model.trainer = Mock()
            model.trainer.estimated_stepping_batches = 1000
            
            # Test that configure_optimizers works
            optimizer_config = model.configure_optimizers()
            assert isinstance(optimizer_config, dict)
            assert "optimizer" in optimizer_config
            assert "lr_scheduler" in optimizer_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
