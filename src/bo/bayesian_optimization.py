#!/usr/bin/env python3
"""
Bayesian Optimization setup for training hyperparameter tuning.

This script optimizes the following hyperparameters:
- training.input_dropout (0.0-0.5, 0 = disabled)
- training.loss_fn (categorical: energy, mse, se, sinkhorn, etc.)
- training.lr_scheduler.name (categorical: cosine_annealing, one_cycle, etc.)
- training.warmup.enabled (boolean)

Target metric: discrimination_score_l1 (perturbation_rank) from agg_results.csv
"""

import os
import tempfile
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import time
from datetime import datetime
from .training_runner import TrainingRunner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei
from skopt import Optimizer


class BayesianOptimizer:
    def __init__(
        self,
        base_config_path: str,
        output_dir: str = "competition",
        max_steps: int = 1000,  # Short training for testing
        n_calls: int = 20,
        n_initial_points: int = 5,  # Number of random exploration points
        random_state: int = 42
    ):
        """
        Initialize Bayesian Optimizer for hyperparameter tuning.
        
        Args:
            base_config_path: Path to base TOML config file
            output_dir: Directory for training outputs
            max_steps: Maximum training steps
            n_calls: Number of optimization iterations
            random_state: Random seed for reproducibility
        """
        self.base_config_path = base_config_path
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.n_calls = n_calls
        self.random_state = random_state
        
        # Generate a unique BO session ID for consistent wandb tagging
        self.bo_session_id = f"bo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Define search space
        self.space = [
            Categorical([0.0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], name='input_dropout'),
            Categorical(['energy', 'mse', 'se', 'sinkhorn', 'cross_entropy', 
                        'wasserstein', 'kl_divergence', 'mmd', 'tabular'], name='loss_fn'),
            Categorical(['cosine_annealing', 'one_cycle', 'warmup_cosine_restarts', 
                        'reduce_on_plateau', 'polynomial_decay'], name='lr_scheduler'),
            Categorical([True, False], name='warmup_enabled'),
            Categorical(['batch', 'random', 'phase'], name='basal_mapping_strategy'),
            Categorical([0.0001, 0.001, 0.01], name='learning_rate'),
            Categorical([True, False], name='phase_enabled')  # Both model and data phase will use this value
        ]
        
        # Initialize optimizer
        self.optimizer = Optimizer(
            dimensions=self.space,
            base_estimator='GP',
            acq_func='EI',
            acq_optimizer='sampling',  # Use sampling for categorical variables
            n_initial_points=n_initial_points,  # Number of random points for exploration
            random_state=random_state
        )
        
        # Track phases
        self.n_initial_points = n_initial_points
        
        # Results tracking
        self.results = []
        self.best_score = -np.inf
        self.best_params = None
        
        # Set up dataset paths - these will be loaded from config in main()
        # For now, use default paths that can be overridden
        dataset_dir_path = "/home/hackerman/Github/state/competition_support_set/withPhases"
        holdout_data_path = f"{dataset_dir_path}/h1_holdout/competition_train_h1_holdout_phase_augmented.h5"
        
        # Initialize training runner
        self.training_runner = TrainingRunner(
            base_config_path=base_config_path,
            perturbation_features_file="/home/hackerman/Github/state/competition_support_set/ESM2_pert_features.pt",  # Always the same file
            max_steps=max_steps,
            num_workers=8,
            holdout_data_path=holdout_data_path
        )
        
        logger.info(f"Initialized BayesianOptimizer with output_dir={output_dir}, max_steps={max_steps}, n_calls={n_calls}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def update_dataset_paths(self, dataset_dir_path: str):
        """Update dataset paths with new base directory."""
        holdout_data_path = f"{dataset_dir_path}/h1_holdout/competition_train_h1_holdout_phase_augmented.h5"
        perturbation_features_file = f"{dataset_dir_path}/ESM2_pert_features.pt"
        
        # Update training runner paths
        self.training_runner.holdout_data_path = holdout_data_path
        self.training_runner.perturbation_features_file = perturbation_features_file
        
        logger.info(f"Updated dataset paths:")
        logger.info(f"  Dataset dir: {dataset_dir_path}")
        logger.info(f"  Holdout data: {holdout_data_path}")
        logger.info(f"  Perturbation features: {perturbation_features_file}")
        
    
    def run_training(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Run training with given parameters and return score."""
        
        # Add timestamp to output directory to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"bo_run_{timestamp}"
        output_dir = f"bo_runs/{run_name}"
        
        # Use the shared training runner with BO session ID for consistent wandb tagging
        score, message = self.training_runner.run_full_pipeline(params, output_dir, run_name, self.bo_session_id)
        
        return score, output_dir
    
    
    def objective(self, params: Dict[str, Any]) -> float:
        """Objective function for optimization (maximize discrimination_score_l1)."""
        
        # Convert parameters to dict
        param_dict = {
            'input_dropout': params[0],
            'loss_fn': params[1],
            'lr_scheduler': params[2],
            'warmup_enabled': params[3],
            'basal_mapping_strategy': params[4],
            'learning_rate': params[5],
            'phase_enabled': params[6]
        }
        
        # Run training and evaluation
        score, eval_dir = self.run_training(param_dict)
        
        # Store results
        result = {
            'params': param_dict.copy(),
            'score': score,
            'eval_dir': eval_dir,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        
        # Check for invalid scores and handle gracefully
        if np.isinf(score) or np.isnan(score):
            error_msg = f"Invalid score detected: {score}. This indicates a problem with the evaluation pipeline."
            print(f"WARNING: {error_msg}")
            # Return a very low score instead of crashing
            score = -1e6
        
        # Update best if this is better
        if score > self.best_score:
            self.best_score = score
            self.best_params = param_dict.copy()
            print(f"New best score: {score}")
            print(f"Best parameters: {self.best_params}")
        
        # Save results to file
        self.save_results()
        
        # Return negative score for minimization
        return -score
    
    def optimize(self):
        """Run Bayesian optimization."""
        
        logger.info(f"Starting Bayesian optimization with {self.n_calls} iterations...")
        logger.info(f"Search space:")
        logger.info(f"  input_dropout: {self.space[0].categories}")
        logger.info(f"  loss_fn: {self.space[1].categories}")
        logger.info(f"  lr_scheduler: {self.space[2].categories}")
        logger.info(f"  warmup_enabled: {self.space[3].categories}")
        logger.info(f"  basal_mapping_strategy: {self.space[4].categories}")
        logger.info(f"  learning_rate: {self.space[5].categories}")
        logger.info(f"  phase_enabled: {self.space[6].categories}")
        
        # Run optimization
        for i in range(self.n_calls):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {i+1}/{self.n_calls}")
            
            # Determine phase
            if i < self.n_initial_points:
                phase = "RANDOM EXPLORATION"
                logger.info(f"PHASE: {phase} (iteration {i+1}/{self.n_initial_points})")
            else:
                phase = "BAYESIAN OPTIMIZATION"
                logger.info(f"PHASE: {phase} (iteration {i+1-self.n_initial_points}/{self.n_calls-self.n_initial_points})")
            
            logger.info(f"{'='*60}")
            
            # Get next parameters to try
            # The optimizer automatically handles random vs. acquisition-based sampling
            params = self.optimizer.ask()
            
            # Convert to dict for easier handling
            param_dict = {
                'input_dropout': params[0],
                'loss_fn': params[1],
                'lr_scheduler': params[2],
                'warmup_enabled': params[3],
                'basal_mapping_strategy': params[4],
                'learning_rate': params[5],
                'phase_enabled': params[6]
            }
            
            # Run objective function
            score = self.objective(params)
            
            # Tell optimizer about the result
            self.optimizer.tell(params, score)
            
            logger.info(f"Iteration {i+1} completed. Score: {-score}")
            logger.info(f"Current best score: {self.best_score}")
        
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"Phase breakdown:")
        logger.info(f"  Random exploration: {self.n_initial_points} iterations")
        logger.info(f"  Bayesian optimization: {self.n_calls - self.n_initial_points} iterations")
        logger.info(f"  Total iterations: {self.n_calls}")
        logger.info(f"Best score: {self.best_score}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def save_results(self):
        """Save optimization results to file."""
        
        results_file = f"{self.output_dir}/bo_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            # Convert numpy types
            for key, value in serializable_result['params'].items():
                if isinstance(value, np.bool_):
                    serializable_result['params'][key] = bool(value)
                elif isinstance(value, np.floating):
                    serializable_result['params'][key] = float(value)
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump({
                'best_score': float(self.best_score),
                'best_params': self.best_params,
                'all_results': serializable_results
            }, f, indent=2)
        
        print(f"Results saved to {results_file}")


def load_config(config_path: str = "src/state/configs/bo/bo_config.yaml"):
    """Load configuration with variable substitution."""
    import os
    import re
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Simple variable substitution
    def substitute_vars(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    
    # Replace ${var} with environment variables or keep as is
    content = re.sub(r'\$\{([^}]+)\}', substitute_vars, content)
    
    # Parse YAML
    config = yaml.safe_load(content)
    
    # Handle dataset_dir_path substitution
    dataset_dir_path = config.get('dataset_dir_path', '/home/hackerman/Github/state/competition_support_set')
    
    # Simple substitution - only replace ${dataset_dir_path} once
    result = config.copy()
    for key, value in result.items():
        if key != 'dataset_dir_path' and isinstance(value, str) and '${dataset_dir_path}' in value:
            result[key] = value.replace('${dataset_dir_path}', dataset_dir_path)
    
    return result

def main():
    """Main function to run Bayesian optimization."""
    
    # Load configuration
    config = load_config()
    
    # Extract configuration values
    base_config = config['base_config_path']
    output_dir = config['output_dir']
    max_steps = config['max_steps']
    n_calls = config['n_calls']
    n_initial_points = config['n_initial_points']
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        base_config_path=base_config,
        output_dir=output_dir,
        max_steps=max_steps,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42
    )
    
    # Update dataset paths from config
    dataset_dir_path = config['dataset_dir_path']
    optimizer.update_dataset_paths(dataset_dir_path)
    
    # Run optimization
    best_params, best_score = optimizer.optimize()
    
    print(f"\nFinal Results:")
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()
