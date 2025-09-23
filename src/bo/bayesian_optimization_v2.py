#!/usr/bin/env python3
"""
Enhanced Bayesian Optimization setup with config-based run management and continuation.

This script provides:
- Config-based folder management
- Automatic run continuation
- State persistence and recovery
- Better organization of results and logs
"""

import os
import subprocess
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
from training_runner import TrainingRunner
from bo_config_manager import BOConfigManager

# Set up logging
logger = logging.getLogger(__name__)

try:
    from skopt import gp_minimize
    from skopt.space import Real, Categorical, Integer
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    from skopt import Optimizer
except ImportError:
    print("Installing scikit-optimize...")
    subprocess.run(["pip", "install", "scikit-optimize"], check=True)
    from skopt import gp_minimize
    from skopt.space import Real, Categorical, Integer
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    from skopt import Optimizer


class BayesianOptimizerV2:
    def __init__(
        self,
        config_path: str,
        base_dir: str = "bo_runs",
        max_steps: int = 1000,
        n_calls: int = 20,
        n_initial_points: int = 5,
        random_state: int = 42,
        force_new: bool = False,
        force_continue: bool = False
    ):
        """
        Initialize Enhanced Bayesian Optimizer with config-based management.
        
        Args:
            config_path: Path to BO config YAML file
            base_dir: Base directory for all BO runs
            max_steps: Maximum training steps
            n_calls: Number of optimization iterations
            n_initial_points: Number of random exploration points
            random_state: Random seed for reproducibility
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.base_dir = base_dir
        self.max_steps = max_steps
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        
        # Initialize config manager
        self.config_manager = BOConfigManager(base_dir)
        
        # Handle force flags
        if force_new:
            # Remove existing run directory if it exists
            config_name = self.config_manager._get_config_name(config_path)
            existing_dir = self.config_manager._get_run_dir(config_name)
            if existing_dir.exists():
                import shutil
                shutil.rmtree(existing_dir)
                logger.info(f"Removed existing run directory: {existing_dir}")
            
            # Also clean up wandb runs for this config
            self._cleanup_wandb_runs(config_name)
        
        # Get or create run directory
        if force_continue:
            # Skip config validation for --continue flag
            config_name = self.config_manager._get_config_name(config_path)
            self.run_dir = self.config_manager._get_run_dir(config_name)
            self.is_continuation = self.run_dir.exists()
            if not self.is_continuation:
                self.run_dir = self.config_manager._create_new_run_dir(config_name, config_path)
        else:
            self.run_dir, self.is_continuation = self.config_manager.get_or_create_run_dir(config_path)
        
        # Set up logging for this specific run
        self._setup_logging()
        
        # Generate BO session ID
        from state.utils.naming import generate_run_name
        self.bo_session_id = generate_run_name("bo")
        
        # Define search space from config
        self.space = self._create_search_space()
        
        # Initialize optimizer - will be properly configured after state loading
        self.optimizer = None
        
        # Results tracking
        self.results = []
        self.best_score = -np.inf
        self.best_params = None
        
        # Add state persistence methods
        self.state_file = self.run_dir / "bo_state.pkl"
        
        # Load existing state if continuing
        if self.is_continuation:
            self._load_existing_state()
        else:
            # Load state from pickle file
            self._load_state()
        
        # Initialize optimizer with proper random state after loading results
        self._initialize_optimizer()
        
        # Initialize training runner
        self.training_runner = TrainingRunner(
            base_config_path=self.config['base_config_path'],
            perturbation_features_file=self.config['training_params']['data']['perturbation_features_file'],
            max_steps=max_steps,
            num_workers=self.config['training_params']['data']['num_workers'],
            base_output_dir=str(self.run_dir),  # Use the BO run directory as base output
            holdout_data_path=self.config['evaluation']['holdout_data_path']
        )
        
        logger.info(f"Initialized BayesianOptimizerV2")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Run dir: {self.run_dir}")
        logger.info(f"  Continuation: {self.is_continuation}")
        logger.info(f"  Max steps: {max_steps}, N calls: {n_calls}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and process configuration file."""
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Simple variable substitution
        import re
        def substitute_vars(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        content = re.sub(r'\$\{([^}]+)\}', substitute_vars, content)
        config = yaml.safe_load(content)
        
        # Handle dataset_dir_path substitution recursively
        dataset_dir_path = config.get('dataset_dir_path', '/home/hackerman/Github/state/competition_support_set')
        
        def substitute_dataset_path(obj):
            if isinstance(obj, str):
                return obj.replace('${dataset_dir_path}', dataset_dir_path)
            elif isinstance(obj, dict):
                return {k: substitute_dataset_path(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_dataset_path(item) for item in obj]
            else:
                return obj
        
        config = substitute_dataset_path(config)
        
        return config
    
    def _initialize_optimizer(self):
        """Initialize optimizer with proper random state based on current state."""
        if self.is_continuation:
            # For resumption, advance the random state to the correct position
            # based on how many iterations have already been completed
            completed_iterations = len(self.results)
            
            # Create a deterministic but unique seed for this run
            # Use run directory hash to ensure consistency across restarts
            run_dir_hash = hash(str(self.run_dir)) % 2**16
            
            # Advance the random state by the number of completed iterations
            # This ensures we continue from the correct position in the random sequence
            dynamic_random_state = (run_dir_hash + completed_iterations) % 2**32
            
            logger.info(f"Resuming run - using advanced random state: {dynamic_random_state}")
            logger.info(f"  Base seed: {run_dir_hash}, completed iterations: {completed_iterations}")
        else:
            dynamic_random_state = self.random_state
            logger.info(f"New run - using fixed random state: {self.random_state}")
        
        self.optimizer = Optimizer(
            dimensions=self.space,
            base_estimator='GP',
            acq_func='EI',
            acq_optimizer='sampling',
            n_initial_points=self.n_initial_points,
            random_state=dynamic_random_state
        )
        
        # Restore optimizer with previous evaluations if resuming
        if self.is_continuation and self.results:
            logger.info(f"Restoring {len(self.results)} previous evaluations to optimizer")
            
            # Rebuild optimizer with previous evaluations
            X = []
            y = []
            for result in self.results:
                params = [result['params'][dim.name] for dim in self.space]
                X.append(params)
                y.append(-result['score'])  # Negative because we minimize
            
            # Tell optimizer about previous evaluations
            if X and y:
                self.optimizer.tell(X, y)
                logger.info(f"Restored {len(X)} previous evaluations to optimizer")
    
    def _setup_logging(self):
        """Set up logging for this specific run."""
        log_file = self.run_dir / "bo_log.txt"
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure logger
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _create_search_space(self):
        """Create search space from config."""
        search_space = self.config.get('search_space', {})
        
        space = []
        for param_name, param_config in search_space.items():
            options = param_config['options']
            
            if param_name in ['warmup_enabled', 'phase_enabled']:
                # Boolean parameters
                space.append(Categorical([True, False], name=param_name))
            elif param_name == 'learning_rate':
                # Learning rate as categorical
                space.append(Categorical(options, name=param_name))
            else:
                # Other categorical parameters
                space.append(Categorical(options, name=param_name))
        
        return space
    
    def _evaluate_completed_runs(self):
        """Check for completed training runs that need evaluation and retry crashed runs."""
        runs_dir = self.run_dir / "runs"
        if not runs_dir.exists():
            return
        
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                # Check if training completed (has final.ckpt)
                training_dir = run_dir / "training_output"
                if not training_dir.exists():
                    # Check double-nested structure
                    run_id = run_dir.name
                    # With single timestamp format, the run_id should match the directory name
                    actual_run_id = run_id
                    training_dir = self.run_dir / "bo_runs" / self.config_manager._get_config_name(self.config_path) / "runs" / run_id / actual_run_id
                
                if training_dir.exists():
                    # Check for final.ckpt (training completed)
                    final_ckpt = training_dir / "final.ckpt"
                    checkpoints_dir = training_dir / "checkpoints"
                    if (final_ckpt.exists() or (checkpoints_dir.exists() and (checkpoints_dir / "final.ckpt").exists())):
                        # Check if evaluation already done
                        eval_dir = run_dir / "eval_results"
                        if not eval_dir.exists():
                            eval_dir = self.run_dir / "bo_runs" / self.config_manager._get_config_name(self.config_path) / "runs" / run_id / "eval_results"
                        
                        if not eval_dir.exists() or not (eval_dir / "agg_results.csv").exists():
                            logger.info(f"Found completed training run that needs evaluation: {run_dir.name}")
                            # Run evaluation for this completed run
                            self._run_evaluation_for_completed_run(run_dir, training_dir, eval_dir)
                    else:
                        # Training didn't complete - this run crashed, ignore it
                        logger.info(f"Found crashed training run, ignoring: {run_dir.name}")
                else:
                    # No training directory - this run never started properly, ignore it
                    logger.info(f"Found incomplete run, ignoring: {run_dir.name}")
    
    def _run_evaluation_for_completed_run(self, run_dir, training_dir, eval_dir):
        """Run evaluation for a completed training run."""
        try:
            # Create eval results directory
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Run evaluation
            success, msg = self.training_runner.run_evaluation(str(training_dir), str(eval_dir))
            if success:
                logger.info(f"Evaluation completed for {run_dir.name}")
                # Extract score and add to results
                try:
                    score = self.training_runner.extract_score(str(eval_dir))
                    # Add to results
                    result = {
                        'params': {},  # We don't have the original params easily accessible
                        'score': score,
                        'eval_dir': str(eval_dir),
                        'timestamp': datetime.now().isoformat(),
                        'run_id': run_dir.name
                    }
                    self.results.append(result)
                    logger.info(f"Added evaluation result: score = {score}")
                except Exception as e:
                    logger.error(f"Failed to extract score for {run_dir.name}: {e}")
                    raise RuntimeError(f"Failed to extract score for {run_dir.name}: {e}")
            else:
                logger.error(f"Evaluation failed for {run_dir.name}: {msg}")
                raise RuntimeError(f"Evaluation failed for {run_dir.name}: {msg}")
        except Exception as e:
            logger.error(f"Error running evaluation for {run_dir.name}: {e}")
            raise RuntimeError(f"Error running evaluation for {run_dir.name}: {e}")
    
    def _load_state(self):
        """Load BO state from file."""
        import pickle
        if self.state_file.exists():
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            self.results = state.get('results', [])
            self.best_score = state.get('best_score', -np.inf)
            self.best_params = state.get('best_params')
            completed = state.get('completed_iterations', 0)
            logger.info(f"Loaded BO state: {completed} completed iterations")
            return completed
        return 0
    
    def _load_existing_state(self):
        """Load existing BO state for continuation."""
        state = self.config_manager.load_bo_state(self.run_dir)
        if state is None:
            logger.info("No existing state found, starting fresh")
            return
        
        # Restore results
        self.results = state.get('results', [])
        self.best_score = state.get('best_score', -np.inf)
        self.best_params = state.get('best_params')
        
        logger.info(f"Restored state: {len(self.results)} evaluations, best score: {self.best_score}")
    
    def _save_state(self):
        """Save current BO state."""
        state = {
            'config_path': self.config_path,
            'run_dir': str(self.run_dir),
            'bo_session_id': self.bo_session_id,
            'max_steps': self.max_steps,
            'n_calls': self.n_calls,
            'n_initial_points': self.n_initial_points,
            'random_state': self.random_state,
            'results': self.results,
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'space_dimensions': [dim.name for dim in self.space],
            'saved_at': datetime.now().isoformat()
        }
        
        self.config_manager.save_bo_state(self.run_dir, state)
    
    def run_training(self, params: Dict[str, Any]) -> Tuple[float, str]:
        """Run training with given parameters and return score."""
        
        # Get next run ID and create output directory
        config_name = self.config_manager._get_config_name(self.config_path)
        run_id = self.config_manager.get_next_run_id(self.run_dir, config_name)
        output_dir = self.config_manager.get_run_output_dir(self.run_dir, run_id)
        output_dir.mkdir(parents=True)
        
        # Use the shared training runner
        score, message = self.training_runner.run_full_pipeline(
            params, run_id, run_id, config_name
        )
        
        return score, str(output_dir)
    
    def objective(self, params: Dict[str, Any]) -> float:
        """Objective function for optimization (maximize discrimination_score_l1)."""
        
        # Run training and evaluation
        score, eval_dir = self.run_training(params)
        
        # Check for invalid scores and crash if found
        if np.isinf(score) or np.isnan(score):
            error_msg = f"Invalid score detected: {score}. This indicates a problem with the training/evaluation pipeline."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Store results
        result = {
            'params': params.copy(),
            'score': score,
            'eval_dir': eval_dir,
            'timestamp': datetime.now().isoformat(),
            'run_id': Path(eval_dir).name
        }
        self.results.append(result)
        
        # Update best if this is better
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            logger.info(f"New best score: {score}")
            logger.info(f"Best parameters: {self.best_params}")
        
        # Save state after each evaluation
        self._save_state()
        
        # Return negative score for minimization
        return -score
    
    def optimize(self):
        """Run Bayesian optimization."""
        
        logger.info(f"Starting Bayesian optimization with {self.n_calls} iterations...")
        logger.info(f"Search space dimensions: {[dim.name for dim in self.space]}")
        
        # Calculate remaining iterations
        completed_iterations = len(self.results)
        remaining_iterations = self.n_calls - completed_iterations
        
        if remaining_iterations <= 0:
            logger.info("All iterations already completed!")
            return self.best_params, self.best_score
        
        logger.info(f"Completed iterations: {completed_iterations}")
        logger.info(f"Remaining iterations: {remaining_iterations}")
        
        # Check for completed training runs that need evaluation
        self._evaluate_completed_runs()
        
        # Run optimization
        for i in range(remaining_iterations):
            iteration = completed_iterations + i + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{self.n_calls}")
            
            # Determine phase
            if iteration <= self.n_initial_points:
                phase = "RANDOM EXPLORATION"
                logger.info(f"PHASE: {phase} (iteration {iteration}/{self.n_initial_points})")
            else:
                phase = "BAYESIAN OPTIMIZATION"
                logger.info(f"PHASE: {phase} (iteration {iteration-self.n_initial_points}/{self.n_calls-self.n_initial_points})")
            
            logger.info(f"{'='*60}")
            
            # Get next parameters to try
            if self.optimizer is None:
                raise RuntimeError("Optimizer not initialized")
            params = self.optimizer.ask()
            
            # Convert to dict for easier handling
            param_dict = {dim.name: params[j] for j, dim in enumerate(self.space)}
            
            # Log the parameters being tried for debugging
            logger.info(f"Parameters for iteration {iteration}: {param_dict}")
            
            # Run objective function
            score = self.objective(param_dict)
            
            # Tell optimizer about the result
            if self.optimizer is None:
                raise RuntimeError("Optimizer not initialized")
            self.optimizer.tell(params, score)
            
            logger.info(f"Iteration {iteration} completed. Score: {-score}")
            logger.info(f"Current best score: {self.best_score}")
            
            # Save state after each iteration
            self._save_state()
        
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"Total iterations: {len(self.results)}")
        logger.info(f"Best score: {self.best_score}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Save final results summary
        self._save_results_summary()
        
        return self.best_params, self.best_score
    
    def _save_results_summary(self):
        """Save human-readable results summary."""
        results_file = self.run_dir / "bo_results.json"
        
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
        
        summary = {
            'config_path': self.config_path,
            'run_dir': str(self.run_dir),
            'bo_session_id': self.bo_session_id,
            'total_iterations': len(self.results),
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'all_results': serializable_results,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results summary saved to {results_file}")
    
    def _cleanup_wandb_runs(self, config_name: str):
        """Clean up wandb runs for a specific config."""
        try:
            import wandb
            from wandb import Api
            
            # Load wandb configuration
            wandb_config_path = Path("src/state/configs/wandb/default.yaml")
            if wandb_config_path.exists():
                with open(wandb_config_path, 'r') as f:
                    wandb_config = yaml.safe_load(f)
                entity = wandb_config.get('entity')
                project = wandb_config.get('project')
                
                if entity and project:
                    api = Api()
                    runs = api.runs(f'{entity}/{project}')
                    
                    # Find and delete runs for this config
                    deleted_count = 0
                    for run in runs:
                        if hasattr(run, 'tags') and 'bo' in run.tags:
                            if run.name and run.name.startswith(config_name):
                                try:
                                    run.delete()
                                    deleted_count += 1
                                    logger.info(f"Deleted wandb run: {run.name}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete wandb run {run.name}: {e}")
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} wandb runs for config '{config_name}'")
                    else:
                        logger.info(f"No wandb runs found for config '{config_name}'")
                else:
                    logger.warning("wandb entity or project not found in config file")
            else:
                logger.warning("wandb config file not found at src/state/configs/wandb/default.yaml")
                
        except ImportError:
            logger.warning("wandb not available, skipping wandb cleanup")
        except Exception as e:
            logger.warning(f"Failed to cleanup wandb runs: {e}")


def main():
    """Main function to run enhanced Bayesian optimization."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python bayesian_optimization_v2.py <config_file>")
        print("Example: python src/bo/bayesian_optimization_v2.py src/state/configs/bo/bo_config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config to get parameters
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create optimizer
    optimizer = BayesianOptimizerV2(
        config_path=config_path,
        base_dir="bo_runs",
        max_steps=config['max_steps'],
        n_calls=config['n_calls'],
        n_initial_points=config['n_initial_points'],
        random_state=config['random_state']
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize()
    
    print(f"\nFinal Results:")
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    print(f"Results saved to: {optimizer.run_dir}")


if __name__ == "__main__":
    main()

