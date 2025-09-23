#!/usr/bin/env python3
"""
Simple runner script for Bayesian optimization.
"""

import yaml
import sys
from pathlib import Path
from bayesian_optimization import BayesianOptimizer


def load_config(config_path: str = "src/state/configs/bo/bo_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function to run Bayesian optimization."""
    
    # Load configuration
    config = load_config()
    
    print("Bayesian Optimization Configuration:")
    print(f"  Base config: {config['base_config_path']}")
    print(f"  Output dir: {config['output_dir']}")
    print(f"  Max steps: {config['max_steps']}")
    print(f"  N calls: {config['n_calls']}")
    print(f"  Target metric: {config['target_metric']}")
    
    # Verify base config exists
    if not Path(config['base_config_path']).exists():
        print(f"Error: Base config file not found: {config['base_config_path']}")
        sys.exit(1)
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        base_config_path=config['base_config_path'],
        output_dir=config['output_dir'],
        max_steps=config['max_steps'],
        n_calls=config['n_calls'],
        random_state=config['random_state']
    )
    
    # Run optimization
    print("\nStarting Bayesian optimization...")
    best_params, best_score = optimizer.optimize()
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETED!")
    print(f"{'='*60}")
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    print(f"Results saved to: {config['output_dir']}/bo_results.json")


if __name__ == "__main__":
    main()

