"""
Tests for learning rate policies.

This module tests all learning rate policy implementations to ensure they work correctly
with the state transition model and PyTorch Lightning.
"""

import os
import sys
import pytest
import torch
import lightning as L
from unittest.mock import Mock
import warnings

# Suppress known harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*loss_fn.*already saved during checkpointing.*")

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from state.tx.models.lr_policies import (
    LearningRatePolicyFactory,
    CosineAnnealingPolicy,
    OneCyclePolicy,
    WarmupCosineRestartsPolicy,
    ReduceOnPlateauPolicy,
    PolynomialDecayPolicy,
    CustomLambdaPolicy,
    LearningRateMonitor,
    create_lr_policy_from_config
)
from state.tx.models.state_transition import StateTransitionPerturbationModel
from state.tx.models.lr_policy_utils import (
    get_recommended_lr_policy,
    print_available_policies,
    print_policy_info
)


class TestLearningRatePolicies:
    """Test suite for learning rate policies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
        self.optimizer = torch.optim.AdamW(self.dummy_params, lr=1e-4)
        self.total_steps = 1000
    
    def test_policy_factory_creation(self):
        """Test that all policies can be created through the factory."""
        policy_names = LearningRatePolicyFactory.list_policies()
        
        for policy_name in policy_names:
            policy = LearningRatePolicyFactory.create_policy(policy_name)
            assert policy is not None
            assert policy.get_name() == policy_name
    
    def test_policy_factory_invalid_name(self):
        """Test that invalid policy names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown policy"):
            LearningRatePolicyFactory.create_policy("invalid_policy")
    
    def test_cosine_annealing_policy(self):
        """Test cosine annealing policy."""
        policy = CosineAnnealingPolicy(
            max_lr=1e-4,
            min_lr=1e-6,
            warmup_steps=100,
            eta_min_ratio=0.1
        )
        
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        assert "frequency" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            self.optimizer.step()  # Call optimizer.step() before scheduler.step()
            scheduler.step()
        
        assert policy.get_name() == "cosine_annealing"
    
    def test_one_cycle_policy(self):
        """Test one cycle policy."""
        policy = OneCyclePolicy(
            max_lr=1e-4,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            self.optimizer.step()  # Call optimizer.step() before scheduler.step()
            scheduler.step()
        
        assert policy.get_name() == "one_cycle"
    
    def test_warmup_cosine_restarts_policy(self):
        """Test warmup cosine restarts policy."""
        policy = WarmupCosineRestartsPolicy(
            max_lr=1e-4,
            min_lr=1e-6,
            warmup_steps=100,
            T_0=500,
            T_mult=1,
            eta_min_ratio=0.1
        )
        
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            self.optimizer.step()  # Call optimizer.step() before scheduler.step()
            scheduler.step()
        
        assert policy.get_name() == "warmup_cosine_restarts"
    
    def test_reduce_on_plateau_policy(self):
        """Test reduce on plateau policy."""
        policy = ReduceOnPlateauPolicy(
            max_lr=1e-4,
            min_lr=1e-6,
            warmup_steps=100,
            mode="min",
            factor=0.5,
            patience=10
        )
        
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        assert "monitor" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            self.optimizer.step()  # Call optimizer.step() before scheduler.step()
            scheduler.step(0.5)  # Provide metrics parameter for ReduceLROnPlateau
        
        assert policy.get_name() == "reduce_on_plateau"
    
    def test_polynomial_decay_policy(self):
        """Test polynomial decay policy."""
        policy = PolynomialDecayPolicy(
            max_lr=1e-4,
            min_lr=1e-6,
            warmup_steps=100,
            power=1.0
        )
        
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            self.optimizer.step()  # Call optimizer.step() before scheduler.step()
            scheduler.step()
        
        assert policy.get_name() == "polynomial_decay"
    
    def test_custom_lambda_policy(self):
        """Test custom lambda policy."""
        def custom_lr_lambda(epoch):
            return 0.95 ** epoch
        
        policy = CustomLambdaPolicy(
            max_lr=1e-4,
            lr_lambda=custom_lr_lambda
        )
        
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        
        # Test that scheduler can be created and stepped
        scheduler = scheduler_config["scheduler"]
        for _ in range(10):
            self.optimizer.step()  # Call optimizer.step() before scheduler.step()
            scheduler.step()
        
        assert policy.get_name() == "custom_lambda"
    
    def test_create_policy_from_config(self):
        """Test creating policy from configuration dictionary."""
        config = {
            "name": "cosine_annealing",
            "max_lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_steps": 100,
            "eta_min_ratio": 0.1
        }
        
        policy = create_lr_policy_from_config(config)
        
        assert policy is not None
        assert policy.get_name() == "cosine_annealing"
        
        # Test that it works with the optimizer
        scheduler_config = policy.get_scheduler(self.optimizer, self.total_steps)
        assert isinstance(scheduler_config, dict)
    
    def test_learning_rate_monitor_callback(self):
        """Test learning rate monitor callback."""
        callback = LearningRateMonitor(logging_interval="step")
        
        assert callback.logging_interval == "step"
        
        # Test that callback can be instantiated without errors
        assert isinstance(callback, L.Callback)
    
    def test_recommended_policies(self):
        """Test getting recommended policies."""
        # Test for state transition model
        for dataset_size in ["small", "medium", "large"]:
            recommended = get_recommended_lr_policy(
                model_type="state_transition",
                training_steps=10000,
                dataset_size=dataset_size
            )
            
            assert isinstance(recommended, dict)
            assert "name" in recommended
            assert "max_lr" in recommended
        
        # Test for transformer model
        for dataset_size in ["small", "medium", "large"]:
            recommended = get_recommended_lr_policy(
                model_type="transformer",
                training_steps=10000,
                dataset_size=dataset_size
            )
            
            assert isinstance(recommended, dict)
            assert "name" in recommended
            assert "max_lr" in recommended
    
    def test_utility_functions(self):
        """Test utility functions."""
        # Test listing policies
        policies = LearningRatePolicyFactory.list_policies()
        assert isinstance(policies, list)
        assert len(policies) > 0
        
        # Test printing functions (should not raise errors)
        print_available_policies()
        print_policy_info("cosine_annealing")


class TestStateTransitionModelWithLRPolicies:
    """Test state transition model integration with learning rate policies."""
    
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
            "embed_key": "X_hvg",  # Add missing embed_key
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
        
        # Test configure_optimizers returns just optimizer
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, torch.optim.AdamW)
    
    def test_model_with_cosine_annealing_policy(self):
        """Test model with cosine annealing policy."""
        lr_policy_config = {
            "name": "cosine_annealing",
            "max_lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_steps": 100,
            "eta_min_ratio": 0.1
        }
        
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_policy_config=lr_policy_config
        )
        
        # Mock trainer to provide estimated_stepping_batches
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000
        
        # Test configure_optimizers returns dict with optimizer and scheduler
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        
        # Test that scheduler can be created
        scheduler_config = optimizer_config["lr_scheduler"]
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
    
    def test_model_with_one_cycle_policy(self):
        """Test model with one cycle policy."""
        lr_policy_config = {
            "name": "one_cycle",
            "max_lr": 1e-4,
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 10000.0
        }
        
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_policy_config=lr_policy_config
        )
        
        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000
        
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_model_with_warmup_cosine_restarts_policy(self):
        """Test model with warmup cosine restarts policy."""
        lr_policy_config = {
            "name": "warmup_cosine_restarts",
            "max_lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_steps": 100,
            "T_0": 500,
            "eta_min_ratio": 0.1
        }
        
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_policy_config=lr_policy_config
        )
        
        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000
        
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_model_with_reduce_on_plateau_policy(self):
        """Test model with reduce on plateau policy."""
        lr_policy_config = {
            "name": "reduce_on_plateau",
            "max_lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_steps": 100,
            "mode": "min",
            "factor": 0.5,
            "patience": 10
        }
        
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_policy_config=lr_policy_config
        )
        
        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000
        
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_model_with_polynomial_decay_policy(self):
        """Test model with polynomial decay policy."""
        lr_policy_config = {
            "name": "polynomial_decay",
            "max_lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_steps": 100,
            "power": 1.0
        }
        
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_policy_config=lr_policy_config
        )
        
        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000
        
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_model_with_custom_lambda_policy(self):
        """Test model with custom lambda policy."""
        def custom_lr_lambda(epoch):
            return 0.95 ** epoch
        
        lr_policy_config = {
            "name": "custom_lambda",
            "max_lr": 1e-4,
            "lr_lambda": custom_lr_lambda
        }
        
        model = StateTransitionPerturbationModel(
            **self.model_config,
            lr_policy_config=lr_policy_config
        )
        
        # Mock trainer
        model.trainer = Mock()
        model.trainer.estimated_stepping_batches = 1000
        
        optimizer_config = model.configure_optimizers()
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
    
    def test_all_policies_with_model(self):
        """Test all available policies with the state transition model."""
        policy_names = LearningRatePolicyFactory.list_policies()
        
        for policy_name in policy_names:
            # Create a basic config for each policy
            lr_policy_config = {
                "name": policy_name,
                "max_lr": 1e-4,
                "min_lr": 1e-6,
                "warmup_steps": 100
            }
            
            # Add policy-specific parameters
            if policy_name == "one_cycle":
                lr_policy_config.update({
                    "pct_start": 0.3,
                    "div_factor": 25.0,
                    "final_div_factor": 10000.0
                })
            elif policy_name == "warmup_cosine_restarts":
                lr_policy_config.update({
                    "T_0": 500,
                    "T_mult": 1,
                    "eta_min_ratio": 0.1
                })
            elif policy_name == "reduce_on_plateau":
                lr_policy_config.update({
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 10
                })
            elif policy_name == "polynomial_decay":
                lr_policy_config.update({
                    "power": 1.0
                })
            elif policy_name == "custom_lambda":
                lr_policy_config.update({
                    "lr_lambda": lambda epoch: 0.95 ** epoch
                })
            else:  # cosine_annealing
                lr_policy_config.update({
                    "eta_min_ratio": 0.1
                })
            
            model = StateTransitionPerturbationModel(
                **self.model_config,
                lr_policy_config=lr_policy_config
            )
            
            # Mock trainer
            model.trainer = Mock()
            model.trainer.estimated_stepping_batches = 1000
            
            # Test that configure_optimizers works
            optimizer_config = model.configure_optimizers()
            assert isinstance(optimizer_config, dict)
            assert "optimizer" in optimizer_config
            assert "lr_scheduler" in optimizer_config
            
            # Test that scheduler can be created and stepped
            scheduler_config = optimizer_config["lr_scheduler"]
            scheduler = scheduler_config["scheduler"]
            
            # Step the scheduler a few times to ensure it works
            optimizer = optimizer_config["optimizer"]
            for _ in range(5):
                optimizer.step()  # Call optimizer.step() before scheduler.step()
                if policy_name == "reduce_on_plateau":
                    scheduler.step(0.5)  # Provide metrics parameter for ReduceLROnPlateau
                else:
                    scheduler.step()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
