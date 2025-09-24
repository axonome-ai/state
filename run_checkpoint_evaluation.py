#!/usr/bin/env python3
"""
Script to run evaluation on multiple checkpoints from a trained model.
Selects first, last, and evenly spaced checkpoints in between.
"""

import os
import re
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple
import shutil


def extract_step_number(checkpoint_name: str) -> int:
    """Extract step number from checkpoint filename."""
    # Handle different naming patterns
    if checkpoint_name == "last.ckpt":
        return float('inf')  # Sort last
    elif checkpoint_name == "final.ckpt":
        return float('inf') - 1  # Sort second to last
    elif checkpoint_name.startswith("step="):
        # Extract number from "step=XXXXX.ckpt"
        match = re.search(r'step=(\d+)', checkpoint_name)
        if match:
            return int(match.group(1))
    return 0  # Default for unrecognized patterns


def get_checkpoints(checkpoint_dir: str) -> List[Tuple[str, int]]:
    """Get all checkpoint files and their step numbers."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoints = []
    for file_path in checkpoint_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.ckpt':
            step_num = extract_step_number(file_path.name)
            checkpoints.append((file_path.name, step_num))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def select_checkpoints(checkpoints: List[Tuple[str, int]], n_checkpoints: int = 5) -> List[str]:
    """Select checkpoints for evaluation.
    
    Args:
        checkpoints: List of (checkpoint_name, step_number) tuples
        n_checkpoints: Number of checkpoints to select:
            - 0: Only the best (last) checkpoint
            - 1: Only the best (last) checkpoint  
            - 2+: First, last, and evenly spaced checkpoints in between
    """
    if len(checkpoints) == 0:
        return []
    
    if n_checkpoints <= 1:
        # Only evaluate the best (last) checkpoint
        return [checkpoints[-1][0]]
    
    if len(checkpoints) <= n_checkpoints:
        # If we have fewer checkpoints than requested, return all
        return [cp[0] for cp in checkpoints]
    
    # Select first, last, and evenly spaced checkpoints in between
    selected = [checkpoints[0][0]]  # First checkpoint
    
    if n_checkpoints > 2:
        # Calculate evenly spaced indices for middle checkpoints
        step_size = (len(checkpoints) - 1) / (n_checkpoints - 1)
        for i in range(1, n_checkpoints - 1):
            idx = int(round(i * step_size))
            if idx < len(checkpoints) - 1:  # Don't duplicate last
                selected.append(checkpoints[idx][0])
    
    selected.append(checkpoints[-1][0])  # Last (best) checkpoint
    return selected


def check_results_exist(output_dir: str) -> bool:
    """Check if evaluation results already exist and are complete."""
    if not os.path.exists(output_dir):
        return False
    
    # Check for key result files
    required_files = ["agg_results.csv", "pred_de.csv", "real_de.csv", "results.csv"]
    for file_name in required_files:
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
    
    return True


def run_evaluation(
    checkpoint_name: str,
    model_dir: str,
    adata_path: str,
    eval_dir: str,
    prepro: bool = True,
    seed: int = 42,
    ctrl_pert_option: str = None,
    force: bool = False
) -> str:
    """Run evaluation for a single checkpoint."""
    print(f"\n=== Running evaluation for checkpoint: {checkpoint_name} ===")
    
    # Set up paths
    data_stem = Path(adata_path).stem
    model_name = Path(model_dir).name.rstrip('/')
    
    if prepro:
        print("RUNNING WITH PREPROCESSING")
        prepro_dir = "/home/hackerman/Github/state/competition_support_set/validation_data"
        prepro_path = f"{prepro_dir}/{data_stem}_preprocessed2_s{seed}.h5"
        
        # Run preprocessing
        prepro_cmd = [
            "/home/hackerman/anaconda3/envs/atlas/bin/python", "-m", "state", "tx", "preprocess_infer",
            "--adata", adata_path,
            "--output", prepro_path,
            "--control_condition", "non-targeting",
            "--pert_col", "target_gene",
            "--seed", str(seed)
        ]
        
        print(f"Running preprocessing: {' '.join(prepro_cmd)}")
        subprocess.run(prepro_cmd, check=True)
        
        # Set up inference
        output_name = f"{model_name}_{data_stem}_s{seed}_with_replace"
        output_path = f"/home/hackerman/Github/state/competition/{output_name}.h5ad"
        
        infer_cmd = [
            "/home/hackerman/anaconda3/envs/atlas/bin/python", "-m", "state", "tx", "infer",
            "--adata", prepro_path,
            "--output", output_path,
            "--model_dir", model_dir,
            "--checkpoint", checkpoint_name,
            "--pert_col", "target_gene",
            "--ctrl_pert", "non-targeting"
        ]
        if ctrl_pert_option is not None:
            infer_cmd.extend(["--ctrl_pert_option", ctrl_pert_option])
        
    else:
        print("RUNNING WITHOUT PREPROCESSING")
        prepro_path = adata_path
        output_name = f"{model_name}_{data_stem}_dataloader"
        output_path = f"/home/hackerman/Github/state/competition/{output_name}.h5ad"
        
        infer_cmd = [
            "/home/hackerman/anaconda3/envs/atlas/bin/python", "-m", "state", "tx", "infer",
            "--adata", prepro_path,
            "--output", output_path,
            "--model_dir", model_dir,
            "--checkpoint", checkpoint_name,
            "--pert_col", "target_gene",
            "--ctrl_pert", "non-targeting"
        ]
        if ctrl_pert_option is not None:
            infer_cmd.extend(["--ctrl_pert_option", ctrl_pert_option])
    
    # Run inference
    print(f"Running inference: {' '.join(infer_cmd)}")
    subprocess.run(infer_cmd, check=True)
    
    # Set up cell evaluation
    output_dir = f"{eval_dir}/cell-eval-{output_name}_{checkpoint_name.replace('.ckpt', '')}"
    
    # Check if results already exist
    if not force and check_results_exist(output_dir):
        print(f"Results already exist for {checkpoint_name}, skipping evaluation")
        print(f"Existing results directory: {output_dir}")
        return output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    cell_eval_cmd = [
        "/home/hackerman/anaconda3/envs/atlas/bin/python", "-m", "cell_eval", "run",
        "--profile", "vcc",
        "-ar", adata_path,
        "-ap", output_path,
        "--num-threads", "12",
        "--outdir", output_dir
    ]
    
    # Run cell evaluation
    print(f"Running cell evaluation: {' '.join(cell_eval_cmd)}")
    subprocess.run(cell_eval_cmd, check=True)
    
    print(f"Evaluation completed. Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on multiple checkpoints")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--adata", required=True, help="Path to input data file")
    parser.add_argument("--eval_dir", default="./cell_eval_results", help="Base directory for evaluation results")
    parser.add_argument("--n_checkpoints", type=int, default=5, help="Number of checkpoints to evaluate (0 or 1 = best only, 2+ = first, last, and evenly spaced)")
    parser.add_argument("--prepro", action="store_true", default=True, help="Run with preprocessing")
    parser.add_argument("--no_prepro", dest="prepro", action="store_false", help="Run without preprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ctrl_pert_option", choices=["replace", None], default=None, help="Control perturbation option")
    parser.add_argument("--force", action="store_true", help="Force re-run even if results already exist")
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = f"{os.getcwd()}/src{os.environ.get('PYTHONPATH', '')}"
    
    # Get checkpoints
    checkpoint_dir = os.path.join(args.model_dir, "checkpoints")
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    
    try:
        all_checkpoints = get_checkpoints(checkpoint_dir)
        print(f"Found {len(all_checkpoints)} checkpoints:")
        for cp_name, step_num in all_checkpoints:
            print(f"  {cp_name} (step: {step_num})")
        
        # Select checkpoints to evaluate
        selected_checkpoints = select_checkpoints(all_checkpoints, args.n_checkpoints)
        print(f"\nSelected {len(selected_checkpoints)} checkpoints for evaluation:")
        for cp in selected_checkpoints:
            print(f"  {cp}")
        
        # Create evaluation directory
        os.makedirs(args.eval_dir, exist_ok=True)
        
        # Run evaluation for each selected checkpoint
        results = []
        skipped = []
        for checkpoint in selected_checkpoints:
            try:
                result_dir = run_evaluation(
                    checkpoint,
                    args.model_dir,
                    args.adata,
                    args.eval_dir,
                    args.prepro,
                    args.seed,
                    args.ctrl_pert_option,
                    args.force
                )
                results.append((checkpoint, result_dir))
            except subprocess.CalledProcessError as e:
                print(f"Error running evaluation for {checkpoint}: {e}")
                continue
        
        print(f"\n=== Evaluation Summary ===")
        print(f"Total checkpoints selected: {len(selected_checkpoints)}")
        print(f"Successfully evaluated: {len(results)}")
        print(f"Results:")
        for checkpoint, result_dir in results:
            print(f"  {checkpoint} -> {result_dir}")
        
        if not args.force:
            print(f"\nNote: Use --force to re-run checkpoints that already have results")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
