#!/usr/bin/env python3
"""
BO Run Manager - Utility for managing Bayesian Optimization runs.

This script provides commands for:
- Listing active runs
- Showing run details
- Cleaning up failed runs
- Resuming interrupted runs
"""

import sys
import json
from pathlib import Path
from bo_config_manager import BOConfigManager


def list_runs():
    """List all active BO runs."""
    manager = BOConfigManager()
    runs = manager.list_active_runs()
    
    if not runs:
        print("No active BO runs found.")
        return
    
    print("Active BO runs:")
    print("="*80)
    print(f"{'Config Name':<25} {'Status':<12} {'Iterations':<12} {'Best Score':<12} {'Modified':<20}")
    print("-"*80)
    
    for name, info in runs.items():
        status = info.get('status', 'unknown')
        iterations = info.get('n_iterations', 0)
        best_score = info.get('best_score', 'N/A')
        modified = info.get('modified', 'N/A')[:19]  # Truncate timestamp
        
        print(f"{name:<25} {status:<12} {iterations:<12} {best_score:<12} {modified:<20}")


def show_run_details(config_name: str):
    """Show detailed information about a specific run."""
    manager = BOConfigManager()
    runs = manager.list_active_runs()
    
    if config_name not in runs:
        print(f"Error: Run '{config_name}' not found.")
        print("Available runs:")
        for name in runs.keys():
            print(f"  {name}")
        return
    
    info = runs[config_name]
    run_dir = Path(info['run_dir'])
    
    print(f"Run Details: {config_name}")
    print("="*60)
    print(f"Run directory: {run_dir}")
    print(f"Status: {info.get('status', 'unknown')}")
    print(f"Iterations: {info.get('n_iterations', 0)}")
    print(f"Best score: {info.get('best_score', 'N/A')}")
    print(f"Created: {info.get('created', 'N/A')}")
    print(f"Modified: {info.get('modified', 'N/A')}")
    
    # Load and show recent results
    results_file = run_dir / "bo_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\nRecent evaluations (last 5):")
        print("-"*60)
        recent_results = results.get('all_results', [])[-5:]
        
        for i, result in enumerate(recent_results, 1):
            score = result.get('score', 'N/A')
            params = result.get('params', {})
            timestamp = result.get('timestamp', 'N/A')[:19]
            
            print(f"  {i}. Score: {score} ({timestamp})")
            for param, value in params.items():
                print(f"     {param}: {value}")
            print()
    
    # Show training runs
    runs_dir = run_dir / "runs"
    if runs_dir.exists():
        training_runs = list(runs_dir.iterdir())
        print(f"Training runs: {len(training_runs)}")
        if training_runs:
            print("  Recent runs:")
            for run_path in sorted(training_runs)[-3:]:  # Show last 3
                print(f"    {run_path.name}")


def cleanup_failed_runs(config_name: str = None):
    """Clean up failed or incomplete runs."""
    manager = BOConfigManager()
    
    if config_name:
        # Clean specific run
        runs = manager.list_active_runs()
        if config_name not in runs:
            print(f"Error: Run '{config_name}' not found.")
            return
        
        run_dir = Path(runs[config_name]['run_dir'])
        print(f"Cleaning up failed runs in {run_dir}...")
        manager.cleanup_failed_runs(run_dir)
        print("Cleanup completed.")
    else:
        # Clean all runs
        runs = manager.list_active_runs()
        if not runs:
            print("No runs to clean up.")
            return
        
        print("Cleaning up failed runs in all active runs...")
        for name, info in runs.items():
            run_dir = Path(info['run_dir'])
            print(f"  Cleaning {name}...")
            manager.cleanup_failed_runs(run_dir)
        print("Cleanup completed.")


def show_help():
    """Show help message."""
    print("BO Run Manager")
    print("="*40)
    print("Commands:")
    print("  list                    - List all active BO runs")
    print("  show <config_name>      - Show details for a specific run")
    print("  cleanup [config_name]   - Clean up failed runs (optionally for specific config)")
    print("  help                    - Show this help message")
    print()
    print("Examples:")
    print("  python bo_manager.py list")
    print("  python bo_manager.py show bo_config_small_test")
    print("  python bo_manager.py cleanup")
    print("  python bo_manager.py cleanup bo_config_small_test")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_runs()
    elif command == "show":
        if len(sys.argv) < 3:
            print("Error: Please specify a config name.")
            print("Usage: python bo_manager.py show <config_name>")
            sys.exit(1)
        show_run_details(sys.argv[2])
    elif command == "cleanup":
        config_name = sys.argv[2] if len(sys.argv) > 2 else None
        cleanup_failed_runs(config_name)
    elif command == "help":
        show_help()
    else:
        print(f"Error: Unknown command '{command}'")
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

