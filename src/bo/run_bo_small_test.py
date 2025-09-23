#!/usr/bin/env python3
"""
Small Bayesian Optimization test to verify both random and guided phase functionality.
This runs a quick test with only 6 iterations and 100 steps each.
"""

import logging
import time
from datetime import datetime
from .bayesian_optimization import BayesianOptimizer, load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_small_bo_test():
    """Run a small Bayesian Optimization test."""
    
    start_time = time.time()
    logger.info("Starting small Bayesian Optimization test...")
    logger.info("This will test both random phase (n_initial_points=5) and guided phase (n_calls=10)")
    logger.info("All runs will share the same wandb tag for easy filtering")
    
    # Load configuration
    config = load_config("src/state/configs/bo/bo_config_small_test.yaml")
    
    # Debug: Check if interpolation was resolved
    logger.info(f"Dataset dir path: {config['dataset_dir_path']}")
    logger.info(f"Resolved base_config_path: {config['base_config_path']}")
    logger.info(f"Config keys: {list(config.keys())}")
    
    # Initialize Bayesian Optimizer with small test config
    optimizer = BayesianOptimizer(
        base_config_path=config['base_config_path'],
        output_dir=config['output_dir'],
        max_steps=config['max_steps'],
        n_calls=config['n_calls'],
        n_initial_points=config['n_initial_points'],
        random_state=config['random_state']
    )
    
    logger.info("Running Bayesian Optimization...")
    logger.info(f"BO Session ID: {optimizer.bo_session_id}")
    logger.info(f"Total iterations: {optimizer.n_calls}")
    logger.info(f"Random exploration: {optimizer.n_initial_points}")
    logger.info(f"Guided optimization: {optimizer.n_calls - optimizer.n_initial_points}")
    logger.info(f"Steps per iteration: {optimizer.max_steps}")
    logger.info(f"Wandb tag for filtering: {optimizer.bo_session_id}")
    
    # Run optimization
    try:
        best_params, best_score, results = optimizer.optimize()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("BAYESIAN OPTIMIZATION TEST COMPLETED!")
        logger.info("="*60)
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per iteration: {total_time/optimizer.n_calls:.2f} seconds")
        
        # Show results summary
        logger.info("\nResults Summary:")
        for i, (params, score) in enumerate(results):
            phase_type = "RANDOM" if i < optimizer.n_initial_points else "GUIDED"
            logger.info(f"  Iteration {i+1} ({phase_type}): Score = {score:.4f}")
            logger.info(f"    - basal_mapping_strategy: {params.get('basal_mapping_strategy', 'N/A')}")
            logger.info(f"    - phase_enabled: {params.get('phase_enabled', 'N/A')}")
            logger.info(f"    - loss_fn: {params.get('loss_fn', 'N/A')}")
            logger.info(f"    - learning_rate: {params.get('learning_rate', 'N/A')}")
        
        logger.info("\n✅ Test completed successfully!")
        logger.info("Both random and guided phase functionality have been tested.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = run_small_bo_test()
    if success:
        logger.info("\n🎉 Small BO test passed! The system is ready for full optimization.")
    else:
        logger.error("\n💥 Small BO test failed! Check the logs for details.")
