#!/usr/bin/env python3
"""
Enhanced runner script for config-based Bayesian optimization.

Usage:
    python run_bo_v2.py <config_file>
    
Examples:
    python src/bo/run_bo_v2.py src/state/configs/bo/bo_config.yaml
    python src/bo/run_bo_v2.py src/state/configs/bo/bo_config_small_test.yaml
"""

import sys
import yaml
from pathlib import Path
from bayesian_optimization_v2 import BayesianOptimizerV2
from bo_config_manager import BOConfigManager


def main():
    """Main function to run enhanced Bayesian optimization."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_bo_v2.py <config_file> [--new|--continue]")
        print("\nExamples:")
        print("  python src/bo/run_bo_v2.py src/state/configs/bo/bo_config.yaml")
        print("  python src/bo/run_bo_v2.py src/state/configs/bo/bo_config.yaml --new")
        print("  python src/bo/run_bo_v2.py src/state/configs/bo/bo_config.yaml --continue")
        print("\nFlags:")
        print("  --new      Force start a new run (overwrites existing)")
        print("  --continue Force continue with config changes")
        sys.exit(1)
    
    config_path = sys.argv[1]
    force_new = "--new" in sys.argv
    force_continue = "--continue" in sys.argv
    
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load and display config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("BAYESIAN OPTIMIZATION RUNNER V2")
    print("="*60)
    print(f"Config file: {config_path}")
    print(f"Base config: {config['base_config_path']}")
    print(f"Max steps: {config['max_steps']}")
    print(f"N calls: {config['n_calls']}")
    print(f"N initial points: {config['n_initial_points']}")
    print(f"Target metric: {config['target_metric']}")
    print("="*60)
    
    # Show active runs
    manager = BOConfigManager()
    active_runs = manager.list_active_runs()
    
    if active_runs:
        print("\nActive BO runs:")
        for name, info in active_runs.items():
            status = info.get('status', 'unknown')
            iterations = info.get('n_iterations', 0)
            best_score = info.get('best_score', 'N/A')
            print(f"  {name}: {status} ({iterations} iterations, best: {best_score})")
        print()
    
    # Create optimizer
    print("Initializing Bayesian optimizer...")
    try:
        optimizer = BayesianOptimizerV2(
            config_path=config_path,
            base_dir="bo_runs",
            max_steps=config['max_steps'],
            n_calls=config['n_calls'],
            n_initial_points=config['n_initial_points'],
            random_state=config['random_state'],
            force_new=force_new,
            force_continue=force_continue
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo resolve this:")
        print("  Use --new to start fresh (overwrites existing run)")
        print("  Use --continue to continue with config changes")
        sys.exit(1)
    
    if optimizer.is_continuation:
        print(f"Continuing existing run: {optimizer.run_dir}")
        print(f"Previous iterations: {len(optimizer.results)}")
        print(f"Previous best score: {optimizer.best_score}")
    else:
        print(f"Starting new run: {optimizer.run_dir}")
    
    # Run optimization
    print("\nStarting optimization...")
    try:
        best_params, best_score = optimizer.optimize()
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETED!")
        print(f"{'='*60}")
        print(f"Best score: {best_score}")
        print(f"Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nResults saved to: {optimizer.run_dir}")
        print(f"Log file: {optimizer.run_dir}/bo_log.txt")
        print(f"Results file: {optimizer.run_dir}/bo_results.json")
        
    except KeyboardInterrupt:
        print(f"\n\nOptimization interrupted by user.")
        print(f"Progress saved to: {optimizer.run_dir}")
        print(f"To continue, run the same command again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during optimization: {e}")
        print(f"Progress saved to: {optimizer.run_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()

