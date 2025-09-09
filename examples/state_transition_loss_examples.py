#!/usr/bin/env python3
"""
Example configurations for StateTransitionPerturbationModel with different loss functions.

This file demonstrates how to configure the StateTransitionPerturbationModel
with various loss functions that are now available.
"""

# Example 1: Using KL Divergence Loss
kl_divergence_config = {
    "loss": "kl_divergence",
    "apply_normalization": True,  # Convert to probabilities before KL divergence
    "input_dim": 512,
    "hidden_dim": 256,
    "output_dim": 512,
    "pert_dim": 128,
    "predict_residual": True,
    "transformer_backbone_key": "GPT2",
    "output_space": "gene",
}

# Example 2: Using Wasserstein Loss
wasserstein_config = {
    "loss": "wasserstein",
    "input_dim": 512,
    "hidden_dim": 256,
    "output_dim": 512,
    "pert_dim": 128,
    "predict_residual": True,
    "transformer_backbone_key": "GPT2",
    "output_space": "gene",
}

# Example 3: Using MMD Loss with custom kernel
mmd_config = {
    "loss": "mmd",
    "kernel": "energy",  # or "gaussian", "laplacian"
    "num_downsample": 1,
    "input_dim": 512,
    "hidden_dim": 256,
    "output_dim": 512,
    "pert_dim": 128,
    "predict_residual": True,
    "transformer_backbone_key": "GPT2",
    "output_space": "gene",
}

# Example 4: Using Tabular Loss
tabular_config = {
    "loss": "tabular",
    "shared": 128,  # Number of shared genes
    "num_downsample": 1,
    "input_dim": 512,
    "hidden_dim": 256,
    "output_dim": 512,
    "pert_dim": 128,
    "predict_residual": True,
    "transformer_backbone_key": "GPT2",
    "output_space": "gene",
}

# Example 5: Using Cross Entropy Loss
cross_entropy_config = {
    "loss": "cross_entropy",
    "input_dim": 512,
    "hidden_dim": 256,
    "output_dim": 512,
    "pert_dim": 128,
    "predict_residual": True,
    "transformer_backbone_key": "GPT2",
    "output_space": "gene",
}

# Example 6: Using Combined Sinkhorn + Energy Loss
combined_config = {
    "loss": "se",
    "sinkhorn_weight": 0.01,
    "energy_weight": 1.0,
    "blur": 0.05,
    "input_dim": 512,
    "hidden_dim": 256,
    "output_dim": 512,
    "pert_dim": 128,
    "predict_residual": True,
    "transformer_backbone_key": "GPT2",
    "output_space": "gene",
}

# Available loss functions:
AVAILABLE_LOSSES = [
    "energy",           # Energy distance (default)
    "mse",             # Mean Squared Error
    "se",              # Combined Sinkhorn + Energy
    "sinkhorn",        # Sinkhorn loss
    "cross_entropy",   # Binary Cross Entropy with Logits
    "wasserstein",     # Wasserstein distance
    "kl_divergence",   # KL Divergence
    "mmd",             # Maximum Mean Discrepancy
    "tabular",         # Tabular loss (gene + cell level)
]

def create_model_with_loss(loss_name: str, **kwargs):
    """
    Create a StateTransitionPerturbationModel with the specified loss function.
    
    Args:
        loss_name: Name of the loss function to use
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured StateTransitionPerturbationModel instance
    """
    from src.state.tx.models.state_transition import StateTransitionPerturbationModel
    
    config = {
        "loss": loss_name,
        "input_dim": 512,
        "hidden_dim": 256,
        "output_dim": 512,
        "pert_dim": 128,
        "predict_residual": True,
        "transformer_backbone_key": "GPT2",
        "output_space": "gene",
        **kwargs
    }
    
    return StateTransitionPerturbationModel(**config)

if __name__ == "__main__":
    print("Available loss functions:")
    for loss in AVAILABLE_LOSSES:
        print(f"  - {loss}")
    
    print("\nExample usage:")
    print("model = create_model_with_loss('kl_divergence', apply_normalization=True)")
    print("model = create_model_with_loss('wasserstein')")
    print("model = create_model_with_loss('mmd', kernel='energy')")
