#!/usr/bin/env python3
"""
Test script to verify the first Bayesian optimization run works correctly.
This runs a single training iteration with default parameters to ensure
the pipeline works end-to-end.
"""

import logging
import numpy as np
import time
from datetime import datetime
from training_runner import TrainingRunner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_bo_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_single_run():
    """Test a single training run with default parameters."""
    
    start_time = time.time()
    logger.info("Testing single training run with NON-DEFAULT parameters...")
    
    # Non-default parameters to test all functionality
    params = {
        'input_dropout': 0.15,  # Non-default: enabled with moderate dropout
        'loss_fn': 'energy',  # Non-default: energy instead of mse
        'lr_scheduler': 'one_cycle',  # Non-default: one_cycle instead of cosine_annealing
        'warmup_enabled': True,  # Non-default: enabled (since default is now False)
        'basal_mapping_strategy': 'phase',  # Non-default: phase instead of batch
        'learning_rate': 0.001,  # Non-default: higher learning rate
        'phase_enabled': True  # Non-default: enable phase functionality
    }
    
    # Initialize training runner with very short training
    runner = TrainingRunner(
        base_config_path="/home/hackerman/Github/state/competition_support_set/withPhases/selected_h1_holdout.toml",
        perturbation_features_file="/home/hackerman/Github/state/competition_support_set/ESM2_pert_features.pt",  # Always the same file
        max_steps=50,  # Very short for testing
        num_workers=8,
        holdout_data_path="/home/hackerman/Github/state/competition_support_set/withPhases/h1_holdout/competition_train_h1_holdout_phase_augmented.h5"
    )
    
    # Create output directory with timestamp (relative to HDD base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_run_{timestamp}"
    
    # Run the full pipeline
    run_name = f"test_run_{timestamp}"
    score, message = runner.run_full_pipeline(params, output_dir, run_name)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if score == -np.inf:
        logger.error(f"Pipeline failed: {message}")
        logger.error(f"Total test time: {total_time:.2f} seconds")
        return False
    
    logger.info(f"\nSUCCESS! Extracted score: {score}")
    logger.info(f"Parameters used (all NON-DEFAULT): {params}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Message: {message}")
    logger.info(f"Total test time: {total_time:.2f} seconds")
    
    # Log which settings were used
    logger.info(f"\nNON-DEFAULT SETTINGS USED:")
    logger.info(f"  - input_dropout: {params['input_dropout']} (enabled, not 0.0)")
    logger.info(f"  - loss_fn: {params['loss_fn']} (energy, not mse)")
    logger.info(f"  - lr_scheduler: {params['lr_scheduler']} (one_cycle, not cosine_annealing)")
    logger.info(f"  - warmup_enabled: {params['warmup_enabled']} (enabled, not disabled)")
    logger.info(f"  - basal_mapping_strategy: {params['basal_mapping_strategy']} (phase, not batch)")
    logger.info(f"  - learning_rate: {params['learning_rate']} (0.001, not 0.0001)")
    logger.info(f"  - phase_enabled: {params['phase_enabled']} (enabled, not disabled)")
    
    return True


if __name__ == "__main__":
    success = test_single_run()
    if success:
        logger.info("\n✅ Test passed! The pipeline works correctly.")
    else:
        logger.error("\n❌ Test failed! Check the errors above.")