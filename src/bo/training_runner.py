#!/usr/bin/env python3
"""
Common training runner for Bayesian optimization and testing.
This module provides shared functionality for running training and evaluation.
"""

import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)


class TrainingRunner:
    """Common training runner for both testing and Bayesian optimization."""
    
    def __init__(
        self,
        base_config_path: str,
        perturbation_features_file: str,
        max_steps: int = 1000,
        num_workers: int = 8,
        base_output_dir: str = "./bo_runs",
        holdout_data_path: Optional[str] = None
    ):
        """
        Initialize the training runner.
        
        Args:
            base_config_path: Path to the TOML config file
            perturbation_features_file: Path to ESM2 perturbation features
            max_steps: Maximum training steps
            num_workers: Number of data loading workers
            base_output_dir: Base directory for all training outputs (on HDD)
            holdout_data_path: Path to holdout dataset for evaluation
        """
        self.base_config_path = base_config_path
        self.perturbation_features_file = perturbation_features_file
        self.max_steps = max_steps
        self.num_workers = num_workers
        self.base_output_dir = base_output_dir
        self.holdout_data_path = holdout_data_path
        
        logger.info(f"Initialized TrainingRunner with max_steps={max_steps}, num_workers={num_workers}")
        logger.info(f"Using HDD for outputs: {base_output_dir}")
    
    def preprocess_holdout_data(self, seed: int = 42) -> Optional[str]:
        """
        Preprocess the holdout dataset once and save with preprocessed name.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Path to the preprocessed holdout file
        """
        if not self.holdout_data_path:
            logger.warning("No holdout data path provided, skipping preprocessing")
            return None
            
        # Check if preprocessed file already exists
        holdout_path = Path(self.holdout_data_path)
        preprocessed_path = holdout_path.parent / f"{holdout_path.stem}_preprocessed.h5"
        
        if preprocessed_path.exists():
            logger.info(f"Preprocessed holdout file already exists: {preprocessed_path}")
            return str(preprocessed_path)
        
        logger.info(f"Preprocessing holdout dataset: {self.holdout_data_path}")
        logger.info(f"Output will be saved to: {preprocessed_path}")
        
        # Run preprocessing command
        cmd = [
            "/home/hackerman/anaconda3/envs/atlas/bin/python", "-m", "state", "tx", "preprocess_infer",
            "--adata", self.holdout_data_path,
            "--output", str(preprocessed_path),
            "--control_condition", "non-targeting",
            "--pert_col", "target_gene",
            "--seed", str(seed)
        ]
        
        try:
            logger.info(f"Running preprocessing: {' '.join(cmd)}")
            logger.info("This may take several minutes...")
            
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        logger.info(f"Preprocessing: {line.strip()}")
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                logger.info("Holdout preprocessing completed successfully!")
                return str(preprocessed_path)
            else:
                logger.error(f"Preprocessing failed with return code: {process.returncode}")
                return None
            
        except Exception as e:
            logger.error(f"Holdout preprocessing failed with error: {e}")
            return None
    
    def build_training_cmd(self, params: Dict[str, Any], output_dir: str, run_name: Optional[str] = None, bo_config_name: Optional[str] = None) -> list:
        """Build the training command with given parameters."""
        if run_name is None:
            run_name = "bo_run"
            
        # Create a safe name without colons for the training script
        safe_name = run_name.replace(":", "_")
        
        # Use 1 worker when basal_mapping_strategy is "batch" to avoid multiprocessing deadlocks
        effective_num_workers = 1 if params.get('basal_mapping_strategy') == 'batch' else self.num_workers
        if effective_num_workers != self.num_workers:
            logger.info(f"🔧 Using {effective_num_workers} worker(s) instead of {self.num_workers} due to basal_mapping_strategy='batch'")
        
        cmd = [
            "/home/hackerman/anaconda3/envs/atlas/bin/python", "-m", "state", "tx", "train",
            f"data.kwargs.toml_config_path={self.base_config_path}",
            f"data.kwargs.num_workers={effective_num_workers}",
            "data.kwargs.batch_col=batch_var",
            "data.kwargs.pert_col=target_gene",
            "data.kwargs.cell_type_key=cell_type",
            "data.kwargs.control_pert=non-targeting",
            f"data.kwargs.perturbation_features_file={self.perturbation_features_file}",
            "data.kwargs.esm_perts_only=true",
            f"training.max_steps={self.max_steps}",
            f"training.ckpt_every_n_steps=1000",  # Checkpoint every 1000 steps
            f"training.val_freq={max(10, self.max_steps//20)}",  # 20 validations with min 10 steps between
            "model=state_sm",
            f"output_dir={output_dir}",
            f"name={safe_name}",  # Set a safe name for the training script
            "validations.diff_exp.enable=true",
            "validations.perturbation.enable=true",
            
            # Wandb configuration with tags
            f"wandb.tags=[state_sm,bo,{run_name}{f',{bo_config_name}' if bo_config_name else ''}]",
            "use_wandb=true",
            
            # Hyperparameters
            f"training=with_warmup",  # Use the config that has warmup and lr_scheduler sections
            f"training.loss_fn={params['loss_fn']}",
            f"training.lr={params['learning_rate']}",  # Add learning rate
            f"model.kwargs.lr_policy_config.name={params['lr_policy_config_name']}",
            f"training.warmup.enabled={str(params['warmup_enabled']).lower()}",
            f"model.kwargs.input_dropout={params['input_dropout']}",  # Set input dropout in model
            f"data.kwargs.basal_mapping_strategy={params['basal_mapping_strategy']}",  # Add basal mapping strategy
            f"data.kwargs.phase={str(params['phase_enabled']).lower()}",  # Add phase enabled parameter to data
            f"model.kwargs.phase={str(params['phase_enabled']).lower()}"  # Set phase enabled parameter in model
        ]
        
        return cmd
    
    def run_training(self, params: Dict[str, Any], output_dir: str, run_name: Optional[str] = None, bo_config_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Run training with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            output_dir: Directory to save training outputs
            run_name: Name for this run (used in wandb tags)
            
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running training with parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"{'='*60}")
        
        cmd = self.build_training_cmd(params, output_dir, run_name, bo_config_name)
        logger.info(f"Training command: {' '.join(cmd)}")
        
        try:
            logger.info("Executing training command...")
            # Run with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[TRAINING] [{timestamp}] {line.rstrip()}")  # Show live output with prefix and timestamp
                        logger.info(f"{line.rstrip()}")
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                logger.info("Training completed successfully!")
                return True, "Training completed successfully"
            else:
                logger.error(f"Training failed with return code: {process.returncode}")
                return False, f"Training failed with return code: {process.returncode}"
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            return False, f"Training failed: {e}"
    
    def run_evaluation(self, model_dir: str, eval_output_dir: str) -> Tuple[bool, str]:
        """
        Run checkpoint evaluation.
        
        Args:
            model_dir: Directory containing the trained model
            
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Evaluating model in {model_dir}")
        
        # Find the actual checkpoint directory (nested structure)
        checkpoint_dirs = list(Path(model_dir).rglob("checkpoints"))
        if not checkpoint_dirs:
            logger.error(f"No checkpoints directory found in {model_dir}")
            return False, "No checkpoints found"
        
        # Use the first checkpoints directory found
        actual_model_dir = checkpoint_dirs[0].parent
        logger.info(f"Using checkpoint directory: {actual_model_dir}")
        
        # Determine which data file to use for evaluation
        if self.holdout_data_path:
            # Preprocess holdout data if needed
            preprocessed_holdout = self.preprocess_holdout_data(seed=42)
            if preprocessed_holdout:
                eval_data_path = preprocessed_holdout
                use_prepro = False  # Already preprocessed
                logger.info(f"Using preprocessed holdout data: {eval_data_path}")
            else:
                eval_data_path = self.holdout_data_path
                use_prepro = True  # Will preprocess during evaluation
                logger.info(f"Using original holdout data with preprocessing: {eval_data_path}")
        else:
            eval_data_path = self.base_config_path
            use_prepro = False
            logger.info(f"Using base config data: {eval_data_path}")
        
        eval_cmd = [
            "/home/hackerman/anaconda3/envs/atlas/bin/python", "run_checkpoint_evaluation.py",
            "--model_dir", str(actual_model_dir),
            "--adata", eval_data_path,
            "--eval_dir", eval_output_dir,  # Specify where to save results
            "--n_checkpoints", "1",  # Only evaluate the best checkpoint
            "--seed", "42",
            "--ctrl_pert_option", "replace"
        ]
        
        # Add preprocessing flag
        if use_prepro:
            eval_cmd.append("--prepro")
        else:
            eval_cmd.append("--no_prepro")
        
        try:
            # Log the full evaluation command prior to execution
            logger.info(f"Evaluation command: {' '.join(eval_cmd)}")
            logger.info("Running evaluation...")
            # Run with real-time output streaming
            process = subprocess.Popen(
                eval_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[EVALUATION] [{timestamp}] {line.rstrip()}")  # Show live output with prefix and timestamp
                        logger.info(f"{line.rstrip()}")
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                logger.info("Evaluation completed successfully!")
                
                # Validate that evaluation actually produced results
                # Look for agg_results.csv in the eval_output_dir or its subdirectories
                eval_dir = Path(eval_output_dir)
                agg_results_file = None
                
                # First check directly in eval_output_dir
                if (eval_dir / "agg_results.csv").exists():
                    agg_results_file = eval_dir / "agg_results.csv"
                else:
                    # Look in subdirectories (cell-eval creates timestamped subdirs)
                    for subdir in eval_dir.iterdir():
                        if subdir.is_dir() and (subdir / "agg_results.csv").exists():
                            agg_results_file = subdir / "agg_results.csv"
                            break
                
                if not agg_results_file:
                    logger.error(f"Evaluation completed but no agg_results.csv found in {eval_output_dir}")
                    logger.error(f"Expected results file: {eval_output_dir}/agg_results.csv")
                    logger.error(f"Contents of eval dir: {list(eval_dir.iterdir())}")
                    return False, "Evaluation completed but no results file found"
                
                logger.info(f"Evaluation results found: {agg_results_file}")
                return True, "Evaluation completed successfully"
            else:
                logger.error(f"Evaluation failed with return code: {process.returncode}")
                logger.error(f"Command that failed: {' '.join(eval_cmd)}")
                return False, f"Evaluation failed with return code: {process.returncode}"
            
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}")
            return False, f"Evaluation failed: {e}"
    
    def extract_score(self, eval_dir: str) -> float:
        """Extract discrimination_score_l1 from agg_results.csv."""
        
        # Find the agg_results.csv file
        agg_results_files = list(Path(eval_dir).rglob("agg_results.csv"))
        
        if not agg_results_files:
            error_msg = f"No agg_results.csv found in {eval_dir}. Evaluation pipeline failed to produce results."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Use the first one found
        agg_file = agg_results_files[0]
        logger.info(f"Reading results from {agg_file}")
        
        try:
            df = pd.read_csv(agg_file)
            logger.info(f"CSV contents:\n{df}")
            
            # Find the mean row for discrimination_score_l1
            mean_row = df[df['statistic'] == 'mean']
            if mean_row.empty:
                error_msg = "No 'mean' row found in agg_results.csv. Invalid evaluation results format."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            score = mean_row.iloc[0]['discrimination_score_l1']
            logger.info(f"Extracted discrimination_score_l1: {score}")
            
            # Check for invalid scores
            if np.isinf(score) or np.isnan(score):
                error_msg = f"Invalid score extracted from evaluation: {score}. Check evaluation pipeline."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            return float(score)
            
        except Exception as e:
            error_msg = f"Error reading agg_results.csv: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def run_full_pipeline(self, params: Dict[str, Any], output_dir: str, run_name: Optional[str] = None, bo_config_name: Optional[str] = None) -> Tuple[float, str]:
        """
        Run the complete training and evaluation pipeline.
        
        Args:
            params: Dictionary of hyperparameters
            output_dir: Directory to save outputs (relative to base_output_dir)
            run_name: Name for this run (used in wandb tags)
            
        Returns:
            Tuple of (score, message)
        """
        # Create full path on HDD
        full_output_dir = os.path.join(self.base_output_dir, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Run training
        success, msg = self.run_training(params, full_output_dir, run_name, bo_config_name)
        if not success:
            raise RuntimeError(f"Training failed: {msg}")
        
        # Run evaluation
        eval_dir = f"{full_output_dir}/eval_results"
        os.makedirs(eval_dir, exist_ok=True)  # Create eval results directory
        success, msg = self.run_evaluation(full_output_dir, eval_dir)
        if not success:
            raise RuntimeError(f"Evaluation failed: {msg}")
        
        # Extract score
        try:
            score = self.extract_score(eval_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to extract score: {e}")
        
        logger.info(f"Pipeline completed successfully! Score: {score}")
        return score, "Pipeline completed successfully" 
