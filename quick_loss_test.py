#!/usr/bin/env python3
"""
Quick test script to verify loss functions work with minimal training.

Usage: python quick_loss_test.py [loss_name]
If no loss_name provided, tests all losses.
"""

import os
import sys
import subprocess
from pathlib import Path

# Disable wandb completely for testing
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

def test_loss_function(loss_name: str = "energy"):
    """Test a single loss function with very minimal training."""
    
    print(f"Testing loss function: {loss_name}")
    
    # Minimal training command
    cmd = [
        "python", "-m", "state._cli._tx._train",
        "data.kwargs.toml_config_path=../state/competition_support_set/overfit_hepg2/overfit_data.toml",
        "data.kwargs.num_workers=1",
        "data.kwargs.batch_col=batch_var",
        "data.kwargs.pert_col=target_gene",
        "data.kwargs.cell_type_key=cell_type", 
        "data.kwargs.control_pert=non-targeting",
        "data.kwargs.perturbation_features_file=/home/hackerman/Github/state/competition_support_set/ESM2_pert_features.pt",
        "training.max_steps=5",  # Very short
        "training.ckpt_every_n_steps=3",
        "training.val_freq=3",
        "model=state_sm",
        f"model.loss={loss_name}",
        "output_dir=quick_test",
        "validations.diff_exp.enable=false",
        "validations.perturbation.enable=false",
        "training.devices=1",
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
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=Path(__file__).parent,
            env=os.environ.copy()  # Pass environment with wandb disabled
        )
        
        if result.returncode == 0:
            print(f"✅ {loss_name}: SUCCESS")
            return True
        else:
            print(f"❌ {loss_name}: FAILED")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {loss_name}: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {loss_name}: ERROR - {e}")
        return False

def test_all_losses():
    """Test all available loss functions."""
    losses = [
        "energy", "mse", "se", "sinkhorn", "cross_entropy",
        "wasserstein", "kl_divergence", "mmd", "tabular"
    ]
    
    results = {}
    for loss in losses:
        results[loss] = test_loss_function(loss)
        print()  # Empty line between tests
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed < total:
        print("Failed losses:", [k for k, v in results.items() if not v])
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific loss function
        loss_name = sys.argv[1]
        success = test_loss_function(loss_name)
        sys.exit(0 if success else 1)
    else:
        # Test all loss functions
        results = test_all_losses()
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
