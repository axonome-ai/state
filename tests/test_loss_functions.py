#!/usr/bin/env python3
"""
Test script to verify all loss functions work correctly with StateTransitionPerturbationModel.

This script runs a short training session for each loss function to ensure they don't crash.
"""

import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path

# Disable wandb completely for testing
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_short_training_test(loss_name: str, additional_kwargs: str = ""):
    """
    Run a short training test for a specific loss function.
    
    Args:
        loss_name: Name of the loss function to test
        additional_kwargs: Additional configuration parameters
    
    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    print(f"\n{'='*60}")
    print(f"Testing loss function: {loss_name}")
    print(f"{'='*60}")
    
    # Base command with very short training
    cmd = [
        "python", "-m", "state._cli._tx._train",
        "data.kwargs.toml_config_path=../state/competition_support_set/overfit_hepg2/overfit_data.toml",
        "data.kwargs.num_workers=2",  # Reduced workers
        "data.kwargs.batch_col=batch_var",
        "data.kwargs.pert_col=target_gene", 
        "data.kwargs.cell_type_key=cell_type",
        "data.kwargs.control_pert=non-targeting",
        "data.kwargs.perturbation_features_file=/home/hackerman/Github/state/competition_support_set/ESM2_pert_features.pt",
        "training.max_steps=10",  # Very short training
        "training.ckpt_every_n_steps=5",
        "training.val_freq=5",
        "model=state_sm",
        f"model.loss={loss_name}",  # Set the loss function
        "output_dir=test_outputs",
        "validations.diff_exp.enable=false",  # Disable validations for speed
        "validations.perturbation.enable=false",
        "training.devices=1",  # Single device
        "training.strategy=auto",
        "use_wandb=false",  # Disable wandb for testing
        "wandb.enable=false",  # Double disable wandb
        additional_kwargs
    ]
    
    # Remove empty strings
    cmd = [arg for arg in cmd if arg]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path(__file__).parent,
            env=os.environ.copy()  # Pass environment with wandb disabled
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Training timed out after 5 minutes"
    except Exception as e:
        return False, "", f"Error running training: {str(e)}"

def test_all_loss_functions():
    """Test all available loss functions."""
    
    # Define loss functions with their specific parameters
    loss_configs = {
        "energy": "",  # Default
        "mse": "",
        "se": "model.sinkhorn_weight=0.01 model.energy_weight=1.0",
        "sinkhorn": "",
        "cross_entropy": "",
        "wasserstein": "",
        "kl_divergence": "model.apply_normalization=false",
        "kl_divergence_normalized": "model.apply_normalization=true",
        "mmd": "model.kernel=energy model.num_downsample=1",
        "tabular": "model.shared=128 model.num_downsample=1"
    }
    
    results = {}
    
    print("Starting loss function tests...")
    print(f"Will test {len(loss_configs)} loss configurations")
    
    for loss_name, additional_kwargs in loss_configs.items():
        start_time = time.time()
        
        success, stdout, stderr = run_short_training_test(loss_name, additional_kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        
        results[loss_name] = {
            "success": success,
            "duration": duration,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if success:
            print(f"✅ {loss_name}: PASSED ({duration:.1f}s)")
        else:
            print(f"❌ {loss_name}: FAILED ({duration:.1f}s)")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
    
    return results

def print_summary(results):
    """Print a summary of test results."""
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("\nFailed tests:")
        for loss_name, result in results.items():
            if not result["success"]:
                print(f"  - {loss_name}")
                if result["stderr"]:
                    print(f"    Error: {result['stderr'][:100]}...")
    
    print(f"\nDetailed results:")
    for loss_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {loss_name}: {status} ({result['duration']:.1f}s)")

def cleanup_test_outputs():
    """Clean up test output directories."""
    test_dirs = ["test_outputs", "wandb_logs"]
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"Cleaned up {dir_name}")

if __name__ == "__main__":
    print("State Transition Model Loss Function Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src/state"):
        print("Error: Please run this script from the axonome-state root directory")
        sys.exit(1)
    
    # Check if the data file exists
    data_file = "../state/competition_support_set/overfit_hepg2/overfit_data.toml"
    if not os.path.exists(data_file):
        print(f"Warning: Data file {data_file} not found. Some tests may fail.")
    
    try:
        # Run tests
        results = test_all_loss_functions()
        
        # Print summary
        print_summary(results)
        
        # Cleanup
        print("\nCleaning up test outputs...")
        cleanup_test_outputs()
        
        # Exit with appropriate code
        all_passed = all(r["success"] for r in results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        cleanup_test_outputs()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        cleanup_test_outputs()
        sys.exit(1)
