#!/usr/bin/env python3
"""
Test script that runs the actual training command with very short parameters.

This uses your exact command but with minimal training steps.
"""

import subprocess
import sys
import os
from pathlib import Path

# Disable wandb completely for testing
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

def run_short_training(loss_name="energy"):
    """Run training with very short parameters."""
    
    print(f"Testing training with loss: {loss_name}")
    
    # Your exact command but with very short training
    cmd = [
        "python", "-m", "state._cli._tx._train",
        "data.kwargs.toml_config_path=../state/competition_support_set/overfit_hepg2/overfit_data.toml",
        "data.kwargs.num_workers=2",  # Reduced from 8
        "data.kwargs.batch_col=batch_var",
        "data.kwargs.pert_col=target_gene",
        "data.kwargs.cell_type_key=cell_type",
        "data.kwargs.control_pert=non-targeting",
        "data.kwargs.perturbation_features_file=/home/hackerman/Github/state/competition_support_set/ESM2_pert_features.pt",
        "training.max_steps=20",  # Very short - was 60000
        "training.ckpt_every_n_steps=10",  # Was 2000
        "training.val_freq=10",  # Was 2000
        "model=state_sm",
        f"model.loss={loss_name}",  # Set the loss function
        "output_dir=test_short_training",
        "validations.diff_exp.enable=false",  # Disable for speed
        "validations.perturbation.enable=false",  # Disable for speed
        "training.devices=1",  # Single device
        "training.strategy=auto",
        "use_wandb=false",  # Disable wandb for testing
        "wandb.enable=false"  # Double disable wandb
    ]
    
    # Add loss-specific parameters
    if loss_name == "kl_divergence":
        cmd.append("model.apply_normalization=false")
    elif loss_name == "se":
        cmd.extend(["model.sinkhorn_weight=0.01", "model.energy_weight=1.0"])
    elif loss_name == "mmd":
        cmd.extend(["model.kernel=energy", "model.num_downsample=1"])
    elif loss_name == "tabular":
        cmd.extend(["model.shared=128", "model.num_downsample=1"])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            timeout=300,  # 5 minute timeout
            cwd=Path(__file__).parent,
            env=os.environ.copy()  # Pass environment with wandb disabled
        )
        
        if result.returncode == 0:
            print(f"✅ {loss_name}: Training completed successfully")
            return True
        else:
            print(f"❌ {loss_name}: Training failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {loss_name}: Training timed out")
        return False
    except Exception as e:
        print(f"❌ {loss_name}: Error - {e}")
        return False

def test_all_losses():
    """Test all loss functions."""
    losses = [
        "energy", "mse", "se", "sinkhorn", "cross_entropy",
        "wasserstein", "kl_divergence", "mmd", "tabular"
    ]
    
    results = {}
    for loss in losses:
        results[loss] = run_short_training(loss)
        print()  # Empty line between tests
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"Summary: {passed}/{total} loss functions worked")
    
    if passed < total:
        print("Failed losses:", [k for k, v in results.items() if not v])
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific loss function
        loss_name = sys.argv[1]
        success = run_short_training(loss_name)
        sys.exit(0 if success else 1)
    else:
        # Test all loss functions
        results = test_all_losses()
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
