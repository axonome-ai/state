#!/usr/bin/env python3
"""
BO Status Reporter - Quick status check for Bayesian Optimization runs
"""

import os
import subprocess
import glob
import sys
import argparse
from pathlib import Path
import time

def is_run_currently_active(run_id: str, config_name: str) -> bool:
    """Check if a run is currently active by looking for running processes"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        # Look for the specific run name in the process command
        # The process command includes the run name with underscores
        return run_id in result.stdout and 'python -m state tx train' in result.stdout
    except:
        return False

def is_bo_process_running() -> bool:
    """Check if a BO process is currently running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'run_bo_v2.py' in result.stdout or 'bayesian_optimization' in result.stdout
    except:
        return False

def print_runs_table(bo_dir: Path, config_name: str):
    """Print a table of all runs associated with this BO config"""
    
    print("\n📊 BO RUNS TABLE")
    
    # Get wandb runs for this BO config
    wandb_runs = []
    
    try:
        import wandb
        import yaml
        
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
        
        # Use only wandb run.name for run IDsoki
        for run in runs:
            if hasattr(run, 'tags') and 'bo' in run.tags:
                if run.name and run.name.startswith(config_name):
                    wandb_runs.append((run.name, run))
        
        # Sort runs by wandb creation time (oldest first, newest last)
        from datetime import datetime
        wandb_runs = sorted(wandb_runs, key=lambda x: x[1].created_at, reverse=False)
        
    except Exception as e:
        print(f"Warning: Could not fetch wandb runs: {e}")
        print("No runs found")
        return
    
    if not wandb_runs:
        print("No runs found")
        return
    
    # Prepare data for pandas table
    data = []
    for run_id, wandb_run in wandb_runs:
        status = "Unknown"
        wandb_status = "N/A"
        step = "N/A"
        score = "Not found"
        train_loss = "N/A"
        val_loss = "N/A"
        created = "N/A"
        
        # Get creation time and calculate duration using wandb data
        try:
            created_time = wandb_run.created_at
            # Handle both datetime objects and string timestamps
            if isinstance(created_time, str):
                from datetime import datetime
                created_time = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
            created = created_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Try to get duration from wandb data first
            duration = "N/A"
            try:
                # Get runtime from wandb summary
                if hasattr(wandb_run, 'summary') and '_runtime' in wandb_run.summary:
                    duration_seconds = wandb_run.summary['_runtime']
                    
                    # Format duration
                    if duration_seconds < 60:
                        duration = f"{int(duration_seconds)}s"
                    elif duration_seconds < 3600:
                        duration = f"{int(duration_seconds/60)}m {int(duration_seconds%60)}s"
                    elif duration_seconds < 86400:
                        hours = int(duration_seconds/3600)
                        minutes = int((duration_seconds%3600)/60)
                        duration = f"{hours}h {minutes}m"
                    else:
                        days = int(duration_seconds/86400)
                        hours = int((duration_seconds%86400)/3600)
                        duration = f"{days}d {hours}h"
                else:
                    # Fallback: calculate from created_at to now for running runs
                    start_time = wandb_run.created_at.timestamp()
                    end_time = time.time()
                    duration_seconds = end_time - start_time
                    
                    if duration_seconds < 60:
                        duration = f"{int(duration_seconds)}s"
                    elif duration_seconds < 3600:
                        duration = f"{int(duration_seconds/60)}m {int(duration_seconds%60)}s"
                    elif duration_seconds < 86400:
                        hours = int(duration_seconds/3600)
                        minutes = int((duration_seconds%3600)/60)
                        duration = f"{hours}h {minutes}m"
                    else:
                        days = int(duration_seconds/86400)
                        hours = int((duration_seconds%86400)/3600)
                        duration = f"{days}d {hours}h"
            except:
                pass  # Keep duration as "N/A" if wandb data not available
        except:
            created = "N/A"
            duration = "N/A"
        
        # We already have wandb data for this run
        wandb_status = wandb_run.state.title()
        
        # Get step and losses from wandb
        try:
            history = wandb_run.history()
            if not history.empty:
                # Get the maximum global_step value instead of just the last row
                # This handles cases where validation steps happen after training ends
                if 'trainer/global_step' in history.columns:
                    step_values = history['trainer/global_step'].dropna()
                    if not step_values.empty:
                        step = int(step_values.max())
                    else:
                        step = 'N/A'
                else:
                    step = 'N/A'
                
                # Extract train and val loss - get last non-NaN values
                train_loss_series = history['train_loss'].dropna()
                val_loss_series = history['val_loss'].dropna()
                
                if not train_loss_series.empty:
                    train_loss = f"{float(train_loss_series.iloc[-1]):.4f}"
                else:
                    train_loss = 'N/A'
                    
                if not val_loss_series.empty:
                    val_loss = f"{float(val_loss_series.iloc[-1]):.4f}"
                else:
                    val_loss = 'N/A'
        except:
            pass
        
        # Check if this run is currently active first
        if is_run_currently_active(run_id, config_name):
            status = "Training"
        else:
            # Use wandb status and step count to determine local status
            if wandb_status == "Finished":
                # Check if it actually completed training (should have many steps)
                if step != "N/A" and isinstance(step, int) and step < 100:
                    # Very few steps means it crashed early
                    status = "Crashed"
                else:
                    status = "Completed"
            elif wandb_status == "Crashed":
                status = "Crashed"
            elif wandb_status == "Failed":
                status = "Failed"
            else:
                # If no BO process is running, these are failed/incomplete runs
                if not is_bo_process_running():
                    status = "Failed"
                else:
                    status = "Queued"
        
        # Check for evaluation results to get score
        # This is the only part that still needs file system access
        try:
            # Convert run_id from underscore format to T format for local directory lookup
            # Replace the underscore after the date (YYYY-MM-DD_HH_MM_SS -> YYYY-MM-DDTHH_MM_SS)
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
                import pandas as pd
                try:
                    df = pd.read_csv(agg_results)
                    if 'discrimination_score_l1' in df.columns and not df['discrimination_score_l1'].isna().all():
                        # Get the mean of discrimination_score_l1 as the BO score
                        # Find the mean row for all metrics
                        mean_row = df[df['statistic'] == 'mean']
                        if not mean_row.empty:
                            mean_score = mean_row.iloc[0]['discrimination_score_l1']
                            score = f"{float(mean_score):.4f}"
                except:
                    pass
        except:
            pass
        
        # Update status based on score availability
        if status == "Completed" and score == "Not found":
            status = "Evaluating"
        
        data.append({
            'Run ID': run_id,
            'Status': status,
            'Wandb Status': wandb_status,
            'Step': step,
            'Score': score,
            'Train Loss': train_loss,
            'Val Loss': val_loss,
            'Duration': duration,
            'Created': created
        })
    
    # Create and display pandas table
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Set pandas display options for prettier formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Use tabulate for even prettier formatting if available
    try:
        from tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        # Fallback to pandas with borders
        print(df.to_string(index=False))
    
    print(f"\nTotal runs: {len(wandb_runs)}")
    
    print()

def get_bo_status(config_name: str = "bo_config_example"):
    """Get comprehensive BO status report"""
    
    print("=" * 60)
    print(f"🤖 BAYESIAN OPTIMIZATION STATUS REPORT - {config_name}")
    print("=" * 60)
    
    # Check BO run directory first
    bo_dir = Path(f"bo_runs/{config_name}")
    if not bo_dir.exists():
        print("❌ BO Dir: Not found")
        return
    
    # Print runs table
    print_runs_table(bo_dir, config_name)
    
    # Check if BO process is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        bo_running = 'run_bo_v2.py' in result.stdout and 'grep' not in result.stdout
        if bo_running:
            print("🟢 Status: BO PROCESS RUNNING")
        else:
            print("🔴 Status: BO PROCESS NOT RUNNING")
    except:
        print("❓ Status: Could not check process")
    
    # Check disk space
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '/dev/sdc1' in line:
                parts = line.split()
                total = parts[1]
                used = parts[2]
                free = parts[3]
                usage = parts[4]
                print(f"💾 Disk: {free} free ({usage} used)")
                break
    except:
        print("❓ Disk: Could not check")
    
    print(f"📁 BO Dir: {bo_dir} exists")
    
    
    # Check log file
    log_file = bo_dir / "bo_log.txt"
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"📝 Log: {len(lines)} lines")
                print(f"📁 Log path: {log_file.absolute()}")
                
                # Get last few lines
                last_lines = lines[-3:] if len(lines) >= 3 else lines
                print("📋 Recent log:")
                for line in last_lines:
                    print(f"   {line.strip()}")
        except:
            print("❓ Log: Could not read")
    
    # Check checkpoints
    ckpt_dir = bo_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        total_size = sum(f.stat().st_size for f in ckpt_files) / (1024**3)  # GB
        print(f"💾 Checkpoints: {len(ckpt_files)} files, {total_size:.1f}GB")
    
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BO Status Reporter')
    parser.add_argument('--name', default='bo_config_example', 
                       help='BO config name to check (default: bo_config_example)')
    
    args = parser.parse_args()
    get_bo_status(args.name)
