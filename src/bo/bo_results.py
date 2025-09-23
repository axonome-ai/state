#!/usr/bin/env python3
"""
BO Results Reporter - Get BO results with tunable parameters
"""

import os
import subprocess
import glob
import sys
import argparse
from pathlib import Path
import time
import yaml
import pandas as pd

def get_bo_results(config_name: str = "bo_config_example"):
    """Get comprehensive BO results report with tunable parameters"""
    
    print("=" * 80)
    print(f"📊 BAYESIAN OPTIMIZATION RESULTS - {config_name}")
    print("=" * 80)
    
    # Check BO run directory first
    bo_dir = Path(f"bo_runs/{config_name}")
    if not bo_dir.exists():
        print("❌ BO Dir: Not found")
        return
    
    # Load BO config to get tunable parameters
    config_path = Path(f"{config_name}.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        bo_config = yaml.safe_load(f)
    
    search_space = bo_config.get('search_space', {})
    tunable_params = list(search_space.keys())
    
    print(f"🔧 Tunable parameters: {', '.join(tunable_params)}")
    print()
    
    # Get wandb runs for this BO config
    wandb_runs = []
    
    try:
        import wandb
        
        # Load wandb configuration
        wandb_config_path = Path("src/state/configs/wandb/default.yaml")
        if wandb_config_path.exists():
            with open(wandb_config_path, 'r') as f:
                wandb_config = yaml.safe_load(f)
            entity = wandb_config.get('entity')
            project = wandb_config.get('project')
            
            if not entity or not project:
                print("Warning: wandb entity or project not found in config file")
                return
        else:
            print("Warning: wandb config file not found at src/state/configs/wandb/default.yaml")
            return
        
        api = wandb.Api()
        runs = api.runs(f'{entity}/{project}')
        
        # Filter runs for this BO config
        for run in runs:
            if hasattr(run, 'tags') and 'bo' in run.tags:
                if run.name and run.name.startswith(config_name):
                    wandb_runs.append((run.name, run))
        
        # Sort runs by name
        wandb_runs = sorted(wandb_runs, key=lambda x: x[0])
        
    except Exception as e:
        print(f"Warning: Could not fetch wandb runs: {e}")
        print("No runs found")
        return
    
    if not wandb_runs:
        print("No runs found")
        return
    
    # Load BO state to get scores and run types
    bo_state_file = bo_dir / "bo_state.json"
    bo_scores = {}
    bo_run_types = {}  # Track whether each run was random or Bayesian
    n_initial_points = 3  # Default, will be updated from config
    
    if bo_state_file.exists():
        try:
            import json
            import re
            with open(bo_state_file, 'r') as f:
                bo_state = json.load(f)
            
            # Get n_initial_points from config
            n_initial_points = bo_state.get('n_initial_points', 3)
            
            for i, result in enumerate(bo_state.get('results', [])):
                run_id_from_bo = result.get('run_id')
                score_from_bo = result.get('score')
                
                # Determine if this was a random or Bayesian run
                run_type = "Random" if i < n_initial_points else "Bayesian"
                
                if run_id_from_bo and score_from_bo is not None:
                    # Store both formats to handle both T and _ date formats
                    bo_scores[run_id_from_bo] = f"{float(score_from_bo):.4f}"
                    bo_run_types[run_id_from_bo] = run_type
                    
                    # Also store normalized version (T -> _)
                    normalized_run_id = re.sub(r'(\d{4}-\d{2}-\d{2})T(\d{2})', r'\1_\2', run_id_from_bo)
                    if normalized_run_id != run_id_from_bo:
                        bo_scores[normalized_run_id] = f"{float(score_from_bo):.4f}"
                        bo_run_types[normalized_run_id] = run_type
        except:
            pass
    
    # Prepare data for results table
    data = []
    for run_id, wandb_run in wandb_runs:
        # Get score from BO state, other metrics from evaluation results
        score = bo_scores.get(run_id, "Not found")
        perturbation_rank = "Not found"
        overlap_at_n = "Not found"
        mae = "Not found"
        
        try:
            # Convert run_id from underscore format to T format for local directory lookup
            import re
            local_run_id = re.sub(r'(\d{4}-\d{2}-\d{2})_(\d{2})', r'\1T\2', run_id)
            eval_dir = bo_dir / local_run_id / "eval_results"
            agg_results = None
            
            if eval_dir.exists():
                # Look for agg_results.csv in subdirectories
                agg_results_files = list(eval_dir.glob("*/agg_results.csv"))
                if agg_results_files:
                    agg_results = agg_results_files[0]  # Take the first one found
                else:
                    # Also check directly in eval_results
                    agg_results = eval_dir / "agg_results.csv"
            
            if agg_results and agg_results.exists():
                try:
                    df = pd.read_csv(agg_results)
                    if 'statistic' in df.columns:
                        # Find the mean row for all metrics
                        mean_row = df[df['statistic'] == 'mean']
                        if not mean_row.empty:
                            # Get perturbation rank from agg_results (discrimination_score_l1)
                            if 'discrimination_score_l1' in df.columns:
                                perturbation_rank = f"{float(mean_row.iloc[0]['discrimination_score_l1']):.4f}"
                            if 'overlap_at_N' in df.columns:
                                overlap_at_n = f"{float(mean_row.iloc[0]['overlap_at_N']):.4f}"
                            if 'mae' in df.columns:
                                mae = f"{float(mean_row.iloc[0]['mae']):.4f}"
                except:
                    pass
        except:
            pass
        
        
        # Check if run is currently active (include even without scores)
        is_currently_running = False
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            is_currently_running = run_id in result.stdout and 'python -m state tx train' in result.stdout
        except:
            pass
        
        # Skip runs without scores unless they're currently running or evaluating
        # Also skip runs that crashed early (step < 10) even if wandb state is "finished"
        if score == "Not found" and not is_currently_running:
            if wandb_run.state != "finished":
                continue
            # If wandb state is "finished" but step is very low, it likely crashed early
            # Get step from wandb history
            try:
                history = wandb_run.history()
                if not history.empty and 'trainer/global_step' in history.columns:
                    step_values = history['trainer/global_step'].dropna()
                    if not step_values.empty:
                        max_step = int(step_values.max())
                        if max_step < 10:
                            continue
            except:
                pass  # If we can't get step info, include the run
        
        # Get tunable parameters from wandb config
        param_values = {}
        try:
            if hasattr(wandb_run, 'config') and wandb_run.config:
                # Map BO parameter names to wandb config names
                param_mapping = {
                    'input_dropout': 'input_dropout',
                    'loss_fn': 'loss_fn',
                    'lr_policy_config_name': 'lr_policy_config',
                    'warmup_enabled': 'warmup',
                    'basal_mapping_strategy': 'data.kwargs.basal_mapping_strategy',
                    'learning_rate': 'lr',
                    'phase_enabled': 'data.kwargs.phase'
                }
                
                for param in tunable_params:
                    wandb_key = param_mapping.get(param, param)
                    
                    # Handle nested keys (e.g., 'data.kwargs.basal_mapping_strategy')
                    if '.' in wandb_key:
                        keys = wandb_key.split('.')
                        value = wandb_run.config
                        try:
                            for key in keys:
                                value = value.get(key, {})
                            if value == {}:
                                value = "N/A"
                        except:
                            value = "N/A"
                    else:
                        value = wandb_run.config.get(wandb_key, "N/A")
                    
                    # Special handling for different parameter types
                    if param == 'lr_policy_config_name' and isinstance(value, dict):
                        param_values[param] = value.get('name', 'N/A')
                    elif param == 'warmup_enabled' and isinstance(value, dict):
                        param_values[param] = value.get('enabled', 'N/A')
                    else:
                        param_values[param] = value
            else:
                # No config available, set all to N/A
                for param in tunable_params:
                    param_values[param] = "N/A"
        except Exception as e:
            # Log the error for debugging but still set params to N/A
            print(f"Warning: Could not get config for run {run_id}: {e}")
            for param in tunable_params:
                param_values[param] = "N/A"
        
        # Get run type (Random or Bayesian)
        run_type = bo_run_types.get(run_id, "Unknown")
        
        # If run type is Unknown but we have BO state info, determine it based on position
        if run_type == "Unknown" and bo_state_file.exists():
            try:
                # Count how many runs are already in BO state
                completed_runs = len(bo_state.get('results', []))
                # If this run would be the next one, determine its type
                if completed_runs < n_initial_points:
                    run_type = "Random"
                else:
                    run_type = "Bayesian"
            except:
                pass
        
        # Create row data
        row_data = {
            'Run Name': run_id,
            'Type': run_type,
            'Score': "Running" if is_currently_running and score == "Not found" else score,
            'Perturbation Rank': "Running" if is_currently_running and perturbation_rank == "Not found" else perturbation_rank,
            'Overlap at N': "Running" if is_currently_running and overlap_at_n == "Not found" else overlap_at_n,
            'MAE': "Running" if is_currently_running and mae == "Not found" else mae
        }
        
        # Add all tunable parameters
        for param in tunable_params:
            row_data[param] = param_values[param]
        
        data.append(row_data)
    
    # Find best score and highlight it using BO scores
    best_score = None
    best_run = None
    try:
        scores = []
        for run_id, score_str in bo_scores.items():
            scores.append(float(score_str))
        
        if scores:
            best_score = max(scores)
            for run_id, score_str in bo_scores.items():
                if float(score_str) == best_score:
                    best_run = run_id
                    break
    except:
        pass
    
    # Create and display pandas table
    df = pd.DataFrame(data)
    
    # Set pandas display options for prettier formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Use tabulate for even prettier formatting if available
    try:
        from tabulate import tabulate
        
        # Highlight the best run if we found one
        if best_run:
            # Add a highlight marker to the best run
            df_display = df.copy()
            df_display.loc[df_display['Run Name'] == best_run, 'Run Name'] = f"{best_run} 🏆"
            print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        # Fallback to pandas with borders
        if best_run:
            df_display = df.copy()
            df_display.loc[df_display['Run Name'] == best_run, 'Run Name'] = f"{best_run} 🏆"
            print(df_display.to_string(index=False))
        else:
            print(df.to_string(index=False))
    
    print(f"\nTotal runs: {len(wandb_runs)}")
    
    # Show best score summary
    if best_score is not None and best_run is not None:
        print(f"\n🏆 Best score: {best_score:.4f} (Run: {best_run})")
    
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BO Results Reporter')
    parser.add_argument('--name', default='bo_config_example', 
                       help='BO config name to check (default: bo_config_example)')
    
    args = parser.parse_args()
    get_bo_results(args.name)
