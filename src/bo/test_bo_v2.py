#!/usr/bin/env python3
"""
Test script for BO V2 system.

This script demonstrates the config-based BO system with:
- Config validation
- Run continuation
- State management
"""

import os
import yaml
import tempfile
from pathlib import Path
from .bo_config_manager import BOConfigManager
from .bayesian_optimization_v2 import BayesianOptimizerV2


def create_test_config():
    """Create a minimal test config."""
    config = {
        'dataset_dir_path': '/tmp/test_dataset',
        'base_config_path': '/tmp/test_config.toml',
        'max_steps': 10,  # Very short for testing
        'n_calls': 5,
        'n_initial_points': 2,
        'random_state': 42,
        'search_space': {
            'input_dropout': {
                'options': [0.0, 0.1, 0.2],
                'description': 'Input dropout rate'
            },
            'loss_fn': {
                'options': ['mse', 'energy'],
                'description': 'Loss function'
            }
        },
        'target_metric': 'discrimination_score_l1',
        'training_params': {
            'data': {
                'num_workers': 1,
                'perturbation_features_file': '/tmp/test_features.pt'
            }
        },
        'evaluation': {
            'holdout_data_path': '/tmp/test_holdout.h5'
        }
    }
    
    # Create temporary config file
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, config_file)
    config_file.close()
    
    return config_file.name


def test_config_manager():
    """Test the BOConfigManager."""
    print("Testing BOConfigManager...")
    
    # Create test config
    config_path = create_test_config()
    
    try:
        # Test config manager
        manager = BOConfigManager("test_bo_runs")
        
        # Test getting/creating run directory
        run_dir, is_continuation = manager.get_or_create_run_dir(config_path)
        print(f"  Run directory: {run_dir}")
        print(f"  Is continuation: {is_continuation}")
        
        # Test listing runs
        runs = manager.list_active_runs()
        print(f"  Active runs: {len(runs)}")
        
        # Test config validation (should be able to continue)
        run_dir2, is_continuation2 = manager.get_or_create_run_dir(config_path)
        print(f"  Second call - continuation: {is_continuation2}")
        
        # Test with modified config (should create new run)
        config_path2 = create_test_config()
        run_dir3, is_continuation3 = manager.get_or_create_run_dir(config_path2)
        print(f"  Modified config - continuation: {is_continuation3}")
        
        print("  ✓ BOConfigManager tests passed")
        
    finally:
        # Cleanup
        os.unlink(config_path)
        if 'config_path2' in locals():
            os.unlink(config_path2)


def test_bo_optimizer():
    """Test the BayesianOptimizerV2 (without actual training)."""
    print("\nTesting BayesianOptimizerV2...")
    
    # Create test config
    config_path = create_test_config()
    
    try:
        # Test optimizer initialization
        optimizer = BayesianOptimizerV2(
            config_path=config_path,
            base_dir="test_bo_runs",
            max_steps=1,  # Very short
            n_calls=3,
            n_initial_points=1,
            random_state=42
        )
        
        print(f"  Config loaded: {len(optimizer.config)} parameters")
        print(f"  Search space: {len(optimizer.space)} dimensions")
        print(f"  Run directory: {optimizer.run_dir}")
        print(f"  Is continuation: {optimizer.is_continuation}")
        
        # Test state saving/loading
        test_state = {
            'test_data': 'test_value',
            'results': [{'params': {'test': 1}, 'score': 0.5}],
            'best_score': 0.5
        }
        
        optimizer.config_manager.save_bo_state(optimizer.run_dir, test_state)
        loaded_state = optimizer.config_manager.load_bo_state(optimizer.run_dir)
        
        assert loaded_state['test_data'] == 'test_value'
        assert len(loaded_state['results']) == 1
        print("  ✓ State save/load tests passed")
        
        print("  ✓ BayesianOptimizerV2 tests passed")
        
    finally:
        # Cleanup
        os.unlink(config_path)


def test_folder_structure():
    """Test the folder structure creation."""
    print("\nTesting folder structure...")
    
    config_path = create_test_config()
    
    try:
        manager = BOConfigManager("test_bo_runs")
        run_dir, _ = manager.get_or_create_run_dir(config_path)
        
        # Check folder structure
        expected_files = ['config.yaml', 'config_hash.txt']
        expected_dirs = ['runs', 'checkpoints']
        
        for file in expected_files:
            assert (run_dir / file).exists(), f"Missing file: {file}"
        
        for dir_name in expected_dirs:
            assert (run_dir / dir_name).exists(), f"Missing directory: {dir_name}"
        
        print("  ✓ Folder structure created correctly")
        
    finally:
        os.unlink(config_path)


def main():
    """Run all tests."""
    print("BO V2 System Test")
    print("="*40)
    
    try:
        test_config_manager()
        test_bo_optimizer()
        test_folder_structure()
        
        print("\n" + "="*40)
        print("All tests passed! ✓")
        print("="*40)
        
        # Show example usage
        print("\nExample usage:")
        print("  python src/bo/run_bo_v2.py src/state/configs/bo/bo_config_example.yaml")
        print("  python src/bo/bo_manager.py list")
        print("  python src/bo/bo_manager.py show bo_config_example")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup test directories
        import shutil
        if Path("test_bo_runs").exists():
            shutil.rmtree("test_bo_runs")
    
    return 0


if __name__ == "__main__":
    exit(main())

